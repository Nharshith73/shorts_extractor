"""FFmpeg-supervised stream ingestion into timestamped MPEG-TS segments."""

from __future__ import annotations

import csv
import hashlib
import logging
import shutil
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import yaml

from .rolling_buffer import RollingBuffer, utc_now

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class StreamConfig:
    """The supported Stage 1 configuration values."""

    source: str
    segment_seconds: float = 6.0
    retention_seconds: float = 900.0
    segments_dir: Path = Path("data/segments")
    reconnect_max_backoff_seconds: float = 60.0
    reanchor_threshold_seconds: float = 2.0
    is_live: bool = False

    def validate(self) -> None:
        if not self.source.strip():
            raise ValueError("source must be a non-empty URL or local file path")
        if self.segment_seconds <= 0:
            raise ValueError("segment_seconds must be positive")
        if self.retention_seconds <= 0:
            raise ValueError("retention_seconds must be positive")
        if self.reconnect_max_backoff_seconds <= 0:
            raise ValueError("reconnect_max_backoff_seconds must be positive")
        if self.reanchor_threshold_seconds <= 0:
            raise ValueError("reanchor_threshold_seconds must be positive")

    @property
    def stall_seconds(self) -> float:
        """A conservative timeout derived from the requested segment duration."""
        return max(15.0, self.segment_seconds * 3.0)


def load_config(config_path: Path, source_override: str | None = None) -> StreamConfig:
    """Load and validate the small YAML configuration used by the CLI."""
    path = config_path.resolve()
    with path.open("r", encoding="utf-8") as config_file:
        raw = yaml.safe_load(config_file) or {}
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")

    allowed_keys = {
        "source",
        "segment_seconds",
        "retention_seconds",
        "segments_dir",
        "reconnect_max_backoff_seconds",
        "reanchor_threshold_seconds",
        "is_live",
    }
    unknown_keys = set(raw) - allowed_keys
    if unknown_keys:
        raise ValueError(f"unsupported configuration keys: {', '.join(sorted(unknown_keys))}")

    source = source_override if source_override is not None else raw.get("source", "")
    if not isinstance(source, str):
        raise ValueError("source must be a string URL or local file path")
    segments_dir_value = raw.get("segments_dir", "data/segments")
    if not isinstance(segments_dir_value, str):
        raise ValueError("segments_dir must be a string path")
    segments_dir = Path(segments_dir_value)
    if not segments_dir.is_absolute():
        segments_dir = path.parent / segments_dir

    config = StreamConfig(
        source=str(source),
        segment_seconds=float(raw.get("segment_seconds", 6)),
        retention_seconds=float(raw.get("retention_seconds", 900)),
        segments_dir=segments_dir,
        reconnect_max_backoff_seconds=float(raw.get("reconnect_max_backoff_seconds", 60)),
        reanchor_threshold_seconds=float(raw.get("reanchor_threshold_seconds", 2.0)),
        is_live=raw.get("is_live", False),
    )
    if not isinstance(config.is_live, bool):
        raise ValueError("is_live must be true or false")
    config.validate()
    return config


def is_local_file(source: str) -> bool:
    """Return whether a source denotes an existing local regular file."""
    return Path(source).expanduser().is_file()


def command_is_available(command: str) -> bool:
    """Check either a PATH command or a direct executable path."""
    return shutil.which(command) is not None or Path(command).is_file()


def probe_duration(path: Path, ffprobe_binary: str = "ffprobe") -> float:
    """Use ffprobe to verify a closed segment is playable and measure it."""
    result = subprocess.run(
        [
            ffprobe_binary,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    duration = float(result.stdout.strip())
    if duration <= 0:
        raise ValueError(f"ffprobe reported a non-positive duration for {path}")
    return duration


@dataclass(slots=True)
class SessionTimestampTracker:
    """Build a contiguous wall-clock timeline for one FFmpeg session."""

    last_wall_start_utc: datetime | None = None
    last_duration: float | None = None

    def resolve(
        self,
        fs_created_utc: datetime,
        duration: float,
        reanchor_threshold_seconds: float,
    ) -> tuple[datetime, bool, float]:
        """Choose a start time from the prior media duration or filesystem time."""
        filesystem_time = fs_created_utc.astimezone(UTC)
        if self.last_wall_start_utc is None or self.last_duration is None:
            resolved = filesystem_time
            timing_reanchored = False
            drift_seconds = 0.0
        else:
            expected = self.last_wall_start_utc + timedelta(seconds=self.last_duration)
            drift_seconds = (filesystem_time - expected).total_seconds()
            timing_reanchored = abs(drift_seconds) > reanchor_threshold_seconds
            resolved = filesystem_time if timing_reanchored else expected
        self.last_wall_start_utc = resolved
        self.last_duration = duration
        return resolved, timing_reanchored, drift_seconds


@dataclass(slots=True)
class _Session:
    """FFmpeg process state for a single connection attempt."""

    process: subprocess.Popen[str]
    list_path: Path
    finished_paths: set[Path]
    session_id: str
    timestamp_tracker: SessionTimestampTracker


class StreamReader:
    """Continuously segment one URL or file and reconnect after interruption.

    FFmpeg's segment muxer is used in stream-copy mode.  It therefore cuts at
    *existing* video keyframes (``-break_non_keyframes 0``), avoiding corrupt
    MPEG-TS files but allowing durations to vary with the source GOP length.
    """

    def __init__(
        self,
        config: StreamConfig,
        *,
        ffmpeg_binary: str = "ffmpeg",
        ffprobe_binary: str = "ffprobe",
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        config.validate()
        self.config = config
        self.ffmpeg_binary = ffmpeg_binary
        self.ffprobe_binary = ffprobe_binary
        self._clock = clock
        self.buffer = RollingBuffer(config.segments_dir, config.retention_seconds)
        self.source_id = hashlib.sha256(config.source.encode("utf-8")).hexdigest()[:16]
        self._stop_event = threading.Event()
        self._supervisor: threading.Thread | None = None
        self._process_lock = threading.Lock()
        self._current_process: subprocess.Popen[str] | None = None
        self._restart_count = 0
        self._completed_segment_count = 0

    @property
    def current_process(self) -> subprocess.Popen[str] | None:
        """Expose the active process for diagnostics and controlled tests."""
        with self._process_lock:
            return self._current_process

    @property
    def restart_count(self) -> int:
        """Number of reconnect attempts completed in this reader run."""
        return self._restart_count

    @property
    def completed_segment_count(self) -> int:
        """Completed, playable segments produced during this reader run."""
        return self._completed_segment_count

    @property
    def is_running(self) -> bool:
        return bool(self._supervisor and self._supervisor.is_alive())

    def _validate_binaries(self) -> None:
        missing = [
            name
            for name in (self.ffmpeg_binary, self.ffprobe_binary)
            if not command_is_available(name)
        ]
        if missing:
            raise RuntimeError(
                "Required executable(s) were not found on PATH: " + ", ".join(missing)
            )

    def start(self) -> None:
        """Start background capture and periodic retention cleanup."""
        if self.is_running:
            raise RuntimeError("StreamReader is already running")
        self._validate_binaries()
        self._stop_event.clear()
        # A prior unclean process cannot still be writing once this sole
        # reader is starting.  Preserve the file for inspection, but ensure it
        # is never considered a valid segment.
        self.buffer.manifest.abandon_writing(self.source_id)
        self.buffer.start_cleanup()
        self._supervisor = threading.Thread(
            target=self._supervise,
            name="streamclip-ingestion-supervisor",
            daemon=True,
        )
        self._supervisor.start()

    def run(self) -> None:
        """Run until a finite file completes or :meth:`stop` is called."""
        self.start()
        self.join()

    def run_forever(self) -> None:
        """Run with SIGINT/SIGTERM cleanup; intended for the module CLI."""
        previous_handlers: dict[int, Any] = {}

        def request_stop(signum: int, _frame: Any) -> None:
            LOGGER.info("Received signal %s; stopping ingestion", signum)
            self.stop()

        for signal_number in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signal_number] = signal.signal(signal_number, request_stop)
        try:
            self.run()
        except KeyboardInterrupt:
            self.stop()
        finally:
            for signal_number, previous_handler in previous_handlers.items():
                signal.signal(signal_number, previous_handler)

    def join(self, timeout: float | None = None) -> None:
        """Wait for the capture supervisor."""
        if self._supervisor:
            self._supervisor.join(timeout)

    def stop(self) -> None:
        """Terminate FFmpeg and mark any active partial files abandoned."""
        self._stop_event.set()
        process = self.current_process
        if process is not None:
            self._terminate_process(process)
        if self._supervisor and self._supervisor is not threading.current_thread():
            self._supervisor.join(timeout=10.0)
        self.buffer.manifest.abandon_writing(self.source_id)
        self.buffer.stop_cleanup()

    def close(self) -> None:
        """Alias for :meth:`stop` for context-manager-style callers."""
        self.stop()

    def _supervise(self) -> None:
        backoff_seconds = 1.0
        try:
            while not self._stop_event.is_set():
                session = self._launch_session()
                if session is None:
                    self._record_reconnect("could not launch ffmpeg")
                    if self._stop_event.wait(backoff_seconds):
                        break
                    backoff_seconds = min(
                        backoff_seconds * 2, self.config.reconnect_max_backoff_seconds
                    )
                    continue

                reason, return_code, made_progress = self._monitor_session(session)
                self._finalize_session(session)
                self.buffer.manifest.abandon_writing(self.source_id, session.session_id)
                self._set_current_process(None)

                if reason == "stopped" or self._stop_event.is_set():
                    break

                normal_file_completion = (
                    not self.config.is_live and reason == "exited" and return_code == 0
                )
                if normal_file_completion:
                    LOGGER.info("Finished local source %s", self.config.source)
                    break

                self._record_reconnect(
                    f"ffmpeg {reason}" + (f" (exit code {return_code})" if return_code is not None else "")
                )
                if made_progress:
                    backoff_seconds = 1.0
                if self._stop_event.wait(backoff_seconds):
                    break
                backoff_seconds = min(
                    backoff_seconds * 2, self.config.reconnect_max_backoff_seconds
                )
        except Exception:
            LOGGER.exception("Ingestion supervisor stopped because of an unexpected error")
        finally:
            process = self.current_process
            if process is not None:
                self._terminate_process(process)
            self.buffer.manifest.abandon_writing(self.source_id)
            self._set_current_process(None)
            self.buffer.stop_cleanup()

    def _record_reconnect(self, reason: str) -> None:
        self._restart_count += 1
        LOGGER.warning("Reconnect %d: %s", self._restart_count, reason)

    def _launch_session(self) -> _Session | None:
        session_id = uuid.uuid4().hex
        list_path = self.buffer.segments_dir / f".segment-list-{session_id}.csv"
        # %%06d remains a segment index after FFmpeg's strftime expansion.  A
        # per-session UUID prevents collisions when a reconnect occurs inside
        # the same clock second.
        output_pattern = self.buffer.segments_dir / (
            f"segment_{session_id}_%Y%m%dT%H%M%S_%%06d.ts"
        )
        command = self._ffmpeg_command(output_pattern, list_path)
        LOGGER.info(
            "Starting FFmpeg capture for source_id=%s session_id=%s",
            self.source_id,
            session_id,
        )
        LOGGER.debug("FFmpeg command: %s", command)
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except OSError as exc:
            LOGGER.error("Could not start ffmpeg: %s", exc)
            return None
        self._set_current_process(process)
        self._drain_process_output(process)
        return _Session(
            process=process,
            list_path=list_path,
            finished_paths=set(),
            session_id=session_id,
            timestamp_tracker=SessionTimestampTracker(),
        )

    def _ffmpeg_command(self, output_pattern: Path, list_path: Path) -> list[str]:
        command = [
            self.ffmpeg_binary,
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "warning",
            "-progress",
            "pipe:1",
        ]
        # Files are intentionally rate-limited during local testing; live URLs
        # are already real-time and do not need this input option.
        if not self.config.is_live and is_local_file(self.config.source):
            command.append("-re")
        command.extend(
            [
                "-i",
                self.config.source,
                "-map",
                "0:v?",
                "-map",
                "0:a?",
                "-c",
                "copy",
                "-f",
                "segment",
                "-segment_format",
                "mpegts",
                "-segment_time",
                str(self.config.segment_seconds),
                "-segment_time_delta",
                "0.05",
                "-break_non_keyframes",
                "0",
                "-reset_timestamps",
                "1",
                "-strftime",
                "1",
                "-segment_list",
                str(list_path),
                "-segment_list_type",
                "csv",
                str(output_pattern),
            ]
        )
        return command

    def _monitor_session(self, session: _Session) -> tuple[str, int | None, bool]:
        last_segment_at = time.monotonic()
        made_progress = False
        while not self._stop_event.is_set():
            self._discover_open_segment_files(session.session_id)
            newly_completed = self._collect_completed_segments(session)
            if newly_completed:
                last_segment_at = time.monotonic()
                made_progress = True

            return_code = session.process.poll()
            if return_code is not None:
                return "exited", return_code, made_progress
            if time.monotonic() - last_segment_at > self.config.stall_seconds:
                LOGGER.warning(
                    "No completed segment for %.1f seconds; restarting FFmpeg",
                    self.config.stall_seconds,
                )
                self._terminate_process(session.process)
                return "stalled", session.process.poll(), made_progress
            self._stop_event.wait(0.25)

        self._terminate_process(session.process)
        return "stopped", session.process.poll(), made_progress

    def _discover_open_segment_files(self, session_id: str) -> None:
        """Track every new file as ``writing`` before FFmpeg closes it.

        FFmpeg's CSV segment list is the authoritative commit record, but
        discovering files independently prevents the cleanup worker from ever
        treating an in-progress or crash-truncated file as a completed one.
        """
        for path in self.buffer.segments_dir.glob("segment_*.ts"):
            if self.buffer.manifest.get_by_path(path) is None:
                self._register_writing_file(path, session_id)

    def _finalize_session(self, session: _Session) -> None:
        """Read the final FFmpeg segment-list entries after process exit."""
        for _ in range(8):
            self._collect_completed_segments(session)
            if session.process.poll() is not None:
                time.sleep(0.05)
        try:
            session.list_path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            LOGGER.warning("Could not remove temporary segment list %s: %s", session.list_path, exc)

    def _collect_completed_segments(self, session: _Session) -> int:
        """Use FFmpeg's list (written only for closed files) as the commit log."""
        if not session.list_path.exists():
            return 0
        completed = 0
        try:
            with session.list_path.open("r", encoding="utf-8", newline="") as segment_list:
                rows = list(csv.reader(segment_list))
        except OSError as exc:
            LOGGER.debug("Could not read segment list %s: %s", session.list_path, exc)
            return 0

        for row in rows:
            if not row:
                continue
            path = Path(row[0])
            if not path.is_absolute():
                path = session.list_path.parent / path
            path = path.resolve()
            if path in session.finished_paths or not path.exists():
                continue
            self._register_writing_file(path, session.session_id)
            try:
                duration = probe_duration(path, self.ffprobe_binary)
            except (OSError, subprocess.SubprocessError, ValueError) as exc:
                LOGGER.warning("Closed segment is not playable yet %s: %s", path, exc)
                continue
            record = self.buffer.manifest.get_by_path(path)
            if record is None:
                LOGGER.warning("Manifest entry disappeared before completion: %s", path)
                continue
            wall_start_utc, timing_reanchored, drift_seconds = session.timestamp_tracker.resolve(
                record.fs_created_utc,
                duration,
                self.config.reanchor_threshold_seconds,
            )
            if timing_reanchored:
                LOGGER.warning(
                    "Re-anchoring segment %s by %+0.3fs (threshold %.3fs)",
                    path.name,
                    drift_seconds,
                    self.config.reanchor_threshold_seconds,
                )
            if self.buffer.manifest.complete(
                path,
                duration,
                wall_start_utc=wall_start_utc,
                timing_reanchored=timing_reanchored,
            ):
                completed += 1
                self._completed_segment_count += 1
                LOGGER.info("Completed segment %s (%.3fs)", path.name, duration)
            session.finished_paths.add(path)
        return completed

    def _register_writing_file(self, path: Path, session_id: str) -> None:
        """Insert the capture-start wall time before transitioning to complete."""
        if self.buffer.manifest.get_by_path(path) is not None:
            return
        try:
            stat = path.stat()
            # Windows exposes creation time as st_ctime; on Unix it is the
            # metadata-change time and still tracks segment file creation here.
            wall_start = datetime.fromtimestamp(stat.st_ctime, UTC)
        except OSError:
            wall_start = self._clock()
        self.buffer.manifest.add_writing(path, wall_start, self.source_id, session_id)

    def _drain_process_output(self, process: subprocess.Popen[str]) -> None:
        """Drain pipes so verbose FFmpeg output cannot block the subprocess."""
        def drain_stdout() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                if line.startswith("progress="):
                    LOGGER.debug("FFmpeg %s", line.strip())

        def drain_stderr() -> None:
            assert process.stderr is not None
            for line in process.stderr:
                LOGGER.debug("FFmpeg: %s", line.strip())

        for target, name in ((drain_stdout, "ffmpeg-progress"), (drain_stderr, "ffmpeg-stderr")):
            threading.Thread(target=target, name=name, daemon=True).start()

    @staticmethod
    def _terminate_process(process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        try:
            process.terminate()
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            LOGGER.warning("FFmpeg did not terminate cleanly; killing it")
            process.kill()
            process.wait(timeout=5)
        except OSError as exc:
            LOGGER.debug("Could not terminate FFmpeg process: %s", exc)

    def _set_current_process(self, process: subprocess.Popen[str] | None) -> None:
        with self._process_lock:
            self._current_process = process

