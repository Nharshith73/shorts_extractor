"""Persistent Single-Stream Supervisor and Lifecycle State Machine."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
from enum import Enum
import logging
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Callable, Iterator

from app.ingestion.rolling_buffer import utc_now
from app.ingestion.stream_reader import StreamConfig, StreamReader
from .config import AuthorizationError, LiveSourceConfig, MissingMediaInputError

LOGGER = logging.getLogger(__name__)


class LiveState(str, Enum):
    STOPPED = "STOPPED"
    DISCOVERING = "DISCOVERING"
    WAITING_FOR_LIVE = "WAITING_FOR_LIVE"
    CONNECTING = "CONNECTING"
    LIVE = "LIVE"
    RECONNECTING = "RECONNECTING"
    ENDED = "ENDED"
    FAILED = "FAILED"
    DISABLED = "DISABLED"
    AUTHORIZATION_REVOKED = "AUTHORIZATION_REVOKED"


class SupervisorStateStore:
    """SQLite persistence layer for supervisor state transitions."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 10000")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS live_supervisor_state (
                    source_id TEXT PRIMARY KEY,
                    current_state TEXT NOT NULL,
                    previous_state TEXT,
                    session_id TEXT,
                    media_url TEXT,
                    reconnect_attempts INTEGER NOT NULL DEFAULT 0,
                    started_at TEXT,
                    updated_at TEXT NOT NULL,
                    error_message TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS live_supervisor_transitions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id TEXT NOT NULL,
                    from_state TEXT NOT NULL,
                    to_state TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    reason TEXT
                )
                """
            )

    def get_state(self, source_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM live_supervisor_state WHERE source_id = ?",
                (source_id,),
            ).fetchone()
            return dict(row) if row else None

    def record_transition(
        self,
        source_id: str,
        to_state: LiveState,
        session_id: str | None = None,
        media_url: str | None = None,
        error_message: str | None = None,
        reason: str = "",
    ) -> LiveState:
        now_iso = utc_now().isoformat()
        with self.connect() as conn:
            existing = conn.execute(
                "SELECT current_state, started_at, reconnect_attempts FROM live_supervisor_state WHERE source_id = ?",
                (source_id,),
            ).fetchone()

            from_state = existing["current_state"] if existing else LiveState.STOPPED.value
            started_at = (existing["started_at"] if existing else now_iso) or now_iso
            reconnect_attempts = existing["reconnect_attempts"] if existing else 0

            if to_state == LiveState.RECONNECTING:
                reconnect_attempts += 1
            elif to_state == LiveState.LIVE:
                reconnect_attempts = 0

            conn.execute(
                """
                INSERT INTO live_supervisor_state (
                    source_id, current_state, previous_state, session_id,
                    media_url, reconnect_attempts, started_at, updated_at, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_id) DO UPDATE SET
                    previous_state = excluded.previous_state,
                    current_state = excluded.current_state,
                    session_id = COALESCE(excluded.session_id, live_supervisor_state.session_id),
                    media_url = COALESCE(excluded.media_url, live_supervisor_state.media_url),
                    reconnect_attempts = excluded.reconnect_attempts,
                    updated_at = excluded.updated_at,
                    error_message = excluded.error_message
                """,
                (
                    source_id,
                    to_state.value,
                    from_state,
                    session_id,
                    media_url,
                    reconnect_attempts,
                    started_at,
                    now_iso,
                    error_message,
                ),
            )

            conn.execute(
                """
                INSERT INTO live_supervisor_transitions (
                    source_id, from_state, to_state, timestamp, reason
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (source_id, from_state, to_state.value, now_iso, reason),
            )

        LOGGER.info(
            "Supervisor [%s]: %s -> %s (reason: %s)",
            source_id,
            from_state,
            to_state.value,
            reason or "none",
        )
        return to_state


class LiveSupervisor:
    """Manages the lifecycle of a single-source stream ingestion process."""

    def __init__(
        self,
        config: LiveSourceConfig,
        *,
        ffmpeg_binary: str = "ffmpeg",
        ffprobe_binary: str = "ffprobe",
    ) -> None:
        self.config = config
        self.ffmpeg_binary = ffmpeg_binary
        self.ffprobe_binary = ffprobe_binary

        manifest_db = self.config.capture.segments_dir / "manifest.sqlite3"
        self.state_store = SupervisorStateStore(manifest_db)

        self._state = LiveState.STOPPED
        self._media_url: str | None = None
        self._reader: StreamReader | None = None
        self._monitor_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # Check existing state after restart
        self._recover_on_startup()

    @property
    def state(self) -> LiveState:
        with self._lock:
            return self._state

    @property
    def reader(self) -> StreamReader | None:
        with self._lock:
            return self._reader

    def _recover_on_startup(self) -> None:
        existing = self.state_store.get_state(self.config.id)
        if existing:
            prev = existing.get("current_state")
            if prev in (LiveState.LIVE.value, LiveState.CONNECTING.value, LiveState.RECONNECTING.value):
                # Process crashed or was restarted while live
                LOGGER.info("Recovering supervisor state from previous run: was %s", prev)
                self._transition(LiveState.STOPPED, reason="Recovered after application restart")

    def _transition(
        self,
        to_state: LiveState,
        session_id: str | None = None,
        media_url: str | None = None,
        error_message: str | None = None,
        reason: str = "",
    ) -> LiveState:
        with self._lock:
            self._state = to_state
            effective_media_url = media_url or self._media_url
            self.state_store.record_transition(
                source_id=self.config.id,
                to_state=to_state,
                session_id=session_id,
                media_url=effective_media_url,
                error_message=error_message,
                reason=reason,
            )
            return self._state

    def start(self, media_override: str | None = None) -> None:
        """Start single-source supervisor."""
        with self._lock:
            if self._state in (LiveState.LIVE, LiveState.CONNECTING):
                raise RuntimeError(f"LiveSupervisor is already active in state {self._state.value}")

        self._stop_event.clear()

        # 1. Authorization validation
        self._transition(LiveState.DISCOVERING, reason="Validating stream authorization")
        try:
            self.config.authorization.validate()
        except AuthorizationError as exc:
            self._transition(
                LiveState.AUTHORIZATION_REVOKED,
                error_message=str(exc),
                reason="Authorization validation failed",
            )
            raise

        # 2. Media URL resolution
        self._transition(LiveState.WAITING_FOR_LIVE, reason="Resolving authorized media feed")
        try:
            if media_override:
                resolved_url = media_override
            else:
                resolved_url = self.config.acquisition.resolve_media_url(self.config.watch_url)
            self._media_url = resolved_url
        except MissingMediaInputError as exc:
            self._transition(
                LiveState.FAILED,
                error_message=str(exc),
                reason="Missing authorized media input feed",
            )
            raise

        # 3. Initialize StreamReader and transition to CONNECTING
        self._transition(
            LiveState.CONNECTING,
            media_url=self._media_url,
            reason=f"Connecting to feed: {self._media_url[:60]}...",
        )

        from app.ingestion.stream_reader import is_local_file

        is_live_source = not is_local_file(self._media_url)
        stream_cfg = StreamConfig(
            source=self._media_url,
            segment_seconds=self.config.capture.segment_seconds,
            retention_seconds=self.config.capture.retention_seconds,
            segments_dir=self.config.capture.segments_dir,
            is_live=is_live_source,
        )

        self._reader = StreamReader(
            config=stream_cfg,
            ffmpeg_binary=self.ffmpeg_binary,
            ffprobe_binary=self.ffprobe_binary,
        )
        self._reader.start()

        # 4. Launch supervision monitor thread
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="live-supervisor-monitor",
            daemon=True,
        )
        self._monitor_thread.start()

    def _monitor_loop(self) -> None:
        """Continuously monitor StreamReader and FFmpeg process health."""
        LOGGER.info("Supervisor monitor thread started for source %s", self.config.id)
        last_completed = 0
        stall_counter = 0

        while not self._stop_event.is_set():
            time.sleep(1.0)
            if self._stop_event.is_set():
                break

            reader = self.reader
            if not reader:
                break

            # Check if reader is alive
            if not reader.is_running:
                proc = reader.current_process
                exit_code = proc.poll() if proc else None
                if exit_code == 0:
                    self._transition(LiveState.ENDED, reason="Stream terminated normally (EOF)")
                else:
                    self._transition(
                        LiveState.FAILED,
                        error_message=f"StreamReader exited unexpectedly with code {exit_code}",
                        reason="StreamReader terminated unexpectedly",
                    )
                break

            # Check segment production
            completed = reader.completed_segment_count
            restarts = reader.restart_count

            if completed > 0:
                if self.state != LiveState.LIVE:
                    self._transition(
                        LiveState.LIVE,
                        session_id=reader.source_id,
                        reason=f"Active segment production verified ({completed} segments produced)",
                    )

            if restarts > 0 and self.state == LiveState.LIVE:
                proc = reader.current_process
                if proc and proc.poll() is not None:
                    self._transition(
                        LiveState.RECONNECTING,
                        reason=f"FFmpeg process disconnected, reconnecting (restarts: {restarts})",
                    )

        LOGGER.info("Supervisor monitor thread finished for source %s", self.config.id)

    def stop(self, timeout_seconds: float = 15.0) -> None:
        """Gracefully shut down StreamReader and FFmpeg without leaving orphans."""
        self._stop_event.set()

        reader = self.reader
        if reader and reader.is_running:
            LOGGER.info("Stopping StreamReader and terminating FFmpeg processes...")
            reader.stop()

        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=3.0)

        self._transition(LiveState.STOPPED, reason="Graceful operator stop")
        LOGGER.info("Supervisor stopped successfully for source %s", self.config.id)

    def get_summary(self) -> dict[str, Any]:
        """Return diagnostic status summary for CLI and monitoring."""
        reader = self.reader
        proc = reader.current_process if reader else None
        return {
            "source_id": self.config.id,
            "creator_name": self.config.creator_name,
            "platform": self.config.platform,
            "state": self.state.value,
            "media_url": self._media_url or "(none)",
            "ffmpeg_pid": proc.pid if proc and proc.poll() is None else None,
            "segments_completed": reader.completed_segment_count if reader else 0,
            "reconnect_count": reader.restart_count if reader else 0,
            "retention_seconds": self.config.capture.retention_seconds,
            "segments_dir": str(self.config.capture.segments_dir),
        }
