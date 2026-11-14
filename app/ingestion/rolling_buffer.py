"""SQLite-backed timestamped segment manifest and retention buffer."""

from __future__ import annotations

import logging
import os
import sqlite3
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Iterable, Sequence

LOGGER = logging.getLogger(__name__)

PLAYABLE_SEGMENT_STATUSES = ("complete",)


def utc_now() -> datetime:
    """Return an aware UTC time in one place for easy testing."""
    return datetime.now(UTC)


def ensure_utc(value: datetime) -> datetime:
    """Normalise a datetime to an aware UTC datetime."""
    if value.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware UTC values")
    return value.astimezone(UTC)


def to_utc_text(value: datetime) -> str:
    """Store UTC datetimes as sortable ISO-8601 text."""
    return ensure_utc(value).isoformat(timespec="microseconds").replace("+00:00", "Z")


def from_utc_text(value: str) -> datetime:
    """Read ISO-8601 UTC text stored by :func:`to_utc_text`."""
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


@dataclass(frozen=True, slots=True)
class SegmentRecord:
    """One manifest entry.

    ``path`` is ``None`` only for a legacy ``discontinuity`` marker.
    ``wall_start_utc`` is the real-world capture time rather than a source PTS
    or filename counter.
    """

    id: int
    path: Path | None
    wall_start_utc: datetime
    fs_created_utc: datetime
    duration: float
    status: str
    source_id: str
    pinned: bool
    session_id: str
    timing_reanchored: bool

    @property
    def wall_end_utc(self) -> datetime:
        return self.wall_start_utc + timedelta(seconds=self.duration)


@dataclass(frozen=True, slots=True)
class SessionBoundary:
    """A point where adjacent captured segments belong to different sessions."""

    boundary_utc: datetime
    source_id: str
    previous_session_id: str | None
    next_session_id: str | None
    previous_segment_id: int | None
    next_segment_id: int | None


class SegmentManifest:
    """Small SQLite data-access layer for the rolling segment manifest.

    Connections are intentionally short-lived.  That keeps it safe for the
    reader, cleanup thread, and future worker processes without sharing a
    SQLite connection between threads.
    """

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._create_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def _create_schema(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS segments (
                    id INTEGER PRIMARY KEY,
                    path TEXT UNIQUE,
                    wall_start_utc TEXT NOT NULL,
                    fs_created_utc TEXT NOT NULL,
                    duration REAL NOT NULL DEFAULT 0,
                    status TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    pinned INTEGER NOT NULL DEFAULT 0,
                    session_id TEXT NOT NULL,
                    timing_reanchored INTEGER NOT NULL DEFAULT 0,
                    UNIQUE(path)
                )
                """
            )
            self._migrate_schema(connection)
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_segments_time
                ON segments (wall_start_utc)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_segments_source_time
                ON segments (source_id, wall_start_utc)
                """
            )
            # Older versions had ``path TEXT UNIQUE`` while some manually
            # created manifests did not.  This explicit index covers both.
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_segments_path_unique ON segments (path)"
            )

    @staticmethod
    def _migrate_schema(connection: sqlite3.Connection) -> None:
        """Upgrade a six-column Stage 1 manifest in place without dropping rows."""
        columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(segments)").fetchall()
        }
        if "pinned" not in columns:
            LOGGER.info("Migrating manifest: adding pinned column")
            connection.execute(
                "ALTER TABLE segments ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0"
            )
            # Earlier releases represented pinning as a status value.  Keep
            # those rows playable while retaining their protection.
            connection.execute(
                "UPDATE segments SET pinned = 1, status = 'complete' WHERE status = 'pinned'"
            )
        if "session_id" not in columns:
            LOGGER.info("Migrating manifest: adding session_id column")
            connection.execute(
                "ALTER TABLE segments ADD COLUMN session_id TEXT NOT NULL DEFAULT ''"
            )
        if "fs_created_utc" not in columns:
            LOGGER.info("Migrating manifest: adding fs_created_utc column")
            connection.execute(
                "ALTER TABLE segments ADD COLUMN fs_created_utc TEXT NOT NULL DEFAULT ''"
            )
        if "timing_reanchored" not in columns:
            LOGGER.info("Migrating manifest: adding timing_reanchored column")
            connection.execute(
                "ALTER TABLE segments ADD COLUMN timing_reanchored INTEGER NOT NULL DEFAULT 0"
            )
        connection.execute(
            """
            UPDATE segments
            SET session_id = 'legacy-' || source_id
            WHERE session_id IS NULL OR session_id = ''
            """
        )
        connection.execute(
            """
            UPDATE segments
            SET fs_created_utc = wall_start_utc
            WHERE fs_created_utc IS NULL OR fs_created_utc = ''
            """
        )

    @staticmethod
    def _record(row: sqlite3.Row) -> SegmentRecord:
        raw_path = row["path"]
        return SegmentRecord(
            id=int(row["id"]),
            path=Path(raw_path) if raw_path else None,
            wall_start_utc=from_utc_text(str(row["wall_start_utc"])),
            fs_created_utc=from_utc_text(str(row["fs_created_utc"])),
            duration=float(row["duration"]),
            status=str(row["status"]),
            source_id=str(row["source_id"]),
            pinned=bool(row["pinned"]),
            session_id=str(row["session_id"]),
            timing_reanchored=bool(row["timing_reanchored"]),
        )

    def add_writing(
        self,
        path: Path,
        wall_start_utc: datetime,
        source_id: str,
        session_id: str,
        fs_created_utc: datetime | None = None,
    ) -> None:
        """Record an in-progress file with a provisional filesystem timestamp."""
        filesystem_time = fs_created_utc or wall_start_utc
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO segments
                (path, wall_start_utc, fs_created_utc, duration, status, source_id,
                 pinned, session_id, timing_reanchored)
                VALUES (?, ?, ?, 0, 'writing', ?, 0, ?, 0)
                """,
                (
                    str(path.resolve()),
                    to_utc_text(wall_start_utc),
                    to_utc_text(filesystem_time),
                    source_id,
                    session_id,
                ),
            )

    def complete(
        self,
        path: Path,
        duration: float,
        *,
        wall_start_utc: datetime | None = None,
        timing_reanchored: bool = False,
    ) -> bool:
        """Atomically mark an observed file playable once it has closed."""
        if duration <= 0:
            raise ValueError("a completed segment must have a positive duration")
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE segments
                SET duration = ?, status = 'complete',
                    wall_start_utc = COALESCE(?, wall_start_utc),
                    timing_reanchored = ?
                WHERE path = ? AND status = 'writing'
                """,
                (
                    duration,
                    to_utc_text(wall_start_utc) if wall_start_utc is not None else None,
                    int(timing_reanchored),
                    str(path.resolve()),
                ),
            )
        return cursor.rowcount == 1

    def abandon_writing(self, source_id: str, session_id: str | None = None) -> int:
        """Flag unclosed files as unusable after a process stop/restart."""
        condition = "WHERE source_id = ? AND status = 'writing'"
        parameters: tuple[object, ...] = (source_id,)
        if session_id is not None:
            condition += " AND session_id = ?"
            parameters += (session_id,)
        with self._connect() as connection:
            cursor = connection.execute(
                f"UPDATE segments SET status = 'abandoned' {condition}",
                parameters,
            )
        return cursor.rowcount

    def add_discontinuity(
        self, wall_start_utc: datetime, source_id: str, session_id: str | None = None
    ) -> None:
        """Add a legacy zero-duration gap marker to the same manifest table.

        New captures represent reconnects by changing ``session_id`` on real
        segments.  This remains for pre-migration records and explicit repair
        tooling.
        """
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO segments
                (path, wall_start_utc, fs_created_utc, duration, status, source_id,
                 pinned, session_id, timing_reanchored)
                VALUES (NULL, ?, ?, 0, 'discontinuity', ?, 0, ?, 0)
                """,
                (
                    to_utc_text(wall_start_utc),
                    to_utc_text(wall_start_utc),
                    source_id,
                    session_id or f"marker-{source_id}",
                ),
            )

    def get_by_path(self, path: Path) -> SegmentRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM segments WHERE path = ?", (str(path.resolve()),)
            ).fetchone()
        return self._record(row) if row else None

    def all_records(self, source_id: str | None = None) -> list[SegmentRecord]:
        query = "SELECT * FROM segments"
        parameters: tuple[object, ...] = ()
        if source_id is not None:
            query += " WHERE source_id = ?"
            parameters = (source_id,)
        query += " ORDER BY wall_start_utc, id"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._record(row) for row in rows]

    def records_with_status(self, statuses: Sequence[str]) -> list[SegmentRecord]:
        if not statuses:
            return []
        placeholders = ", ".join("?" for _ in statuses)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM segments WHERE status IN ({placeholders}) "
                "ORDER BY wall_start_utc, id",
                tuple(statuses),
            ).fetchall()
        return [self._record(row) for row in rows]

    def pin(self, segment_ids: Iterable[int]) -> int:
        """Protect completed segment IDs from automatic retention cleanup."""
        ids = tuple(dict.fromkeys(segment_ids))
        if not ids:
            return 0
        placeholders = ", ".join("?" for _ in ids)
        with self._connect() as connection:
            cursor = connection.execute(
                f"UPDATE segments SET pinned = 1 "
                f"WHERE id IN ({placeholders}) AND status = 'complete'",
                ids,
            )
        return cursor.rowcount

    def unpin(self, segment_ids: Iterable[int]) -> int:
        """Return pinned segments to normal retention behaviour."""
        ids = tuple(dict.fromkeys(segment_ids))
        if not ids:
            return 0
        placeholders = ", ".join("?" for _ in ids)
        with self._connect() as connection:
            cursor = connection.execute(
                f"UPDATE segments SET pinned = 0 "
                f"WHERE id IN ({placeholders}) AND status = 'complete'",
                ids,
            )
        return cursor.rowcount

    def delete_record(self, segment_id: int) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM segments WHERE id = ?", (segment_id,))


class RollingBuffer:
    """A recent on-disk window of MPEG-TS segments and their metadata."""

    def __init__(self, segments_dir: Path, retention_seconds: float) -> None:
        if retention_seconds <= 0:
            raise ValueError("retention_seconds must be positive")
        self.segments_dir = segments_dir.resolve()
        self.segments_dir.mkdir(parents=True, exist_ok=True)
        self.retention_seconds = retention_seconds
        self.manifest = SegmentManifest(self.segments_dir / "manifest.sqlite3")
        self._cleanup_stop = threading.Event()
        self._cleanup_thread: threading.Thread | None = None
        self._cleanup_interval_seconds = self._default_cleanup_interval_seconds()
        self._stats_lock = threading.Lock()
        self._total_deleted_rows = 0

    def _default_cleanup_interval_seconds(self) -> float:
        return max(1.0, min(30.0, self.retention_seconds / 10))

    @property
    def cleanup_interval_seconds(self) -> float:
        """The active interval used by the background cleanup worker."""
        return self._cleanup_interval_seconds

    @property
    def total_deleted_rows(self) -> int:
        """Rows removed by this buffer instance since it was created."""
        with self._stats_lock:
            return self._total_deleted_rows

    def pin(self, segment_ids: Iterable[int]) -> int:
        """Pin complete segments so cleanup will never remove them."""
        return self.manifest.pin(segment_ids)

    def unpin(self, segment_ids: Iterable[int]) -> int:
        """Unpin segments after downstream processing has released them."""
        return self.manifest.unpin(segment_ids)

    def cleanup_once(self, now: datetime | None = None) -> int:
        """Delete expired non-writing, non-pinned segment files and rows.

        The file removal deliberately occurs before its database row is removed:
        if deletion fails, the manifest remains an accurate record that needs a
        later retry.  ``writing`` and ``pinned`` records are never selected.
        """
        current_time = ensure_utc(now) if now is not None else utc_now()
        cutoff = current_time - timedelta(seconds=self.retention_seconds)
        deletable = self.manifest.records_with_status(("complete", "abandoned", "discontinuity"))
        removed = 0
        for record in deletable:
            if record.pinned or record.wall_end_utc >= cutoff:
                continue
            try:
                if record.path is not None and record.path.exists():
                    record.path.unlink()
                self.manifest.delete_record(record.id)
                removed += 1
                LOGGER.debug("Removed expired manifest entry %s", record.id)
            except OSError as exc:
                LOGGER.warning("Could not delete expired segment %s: %s", record.path, exc)
        if removed:
            with self._stats_lock:
                self._total_deleted_rows += removed
        return removed

    def start_cleanup(self, interval_seconds: float | None = None) -> None:
        """Start periodic retention cleanup in a daemon thread."""
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            return
        interval = interval_seconds if interval_seconds is not None else self._default_cleanup_interval_seconds()
        if interval <= 0:
            raise ValueError("cleanup interval must be positive")
        self._cleanup_interval_seconds = interval
        self._cleanup_stop.clear()
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            args=(interval,),
            name="streamclip-retention-cleanup",
            daemon=True,
        )
        self._cleanup_thread.start()

    def _cleanup_loop(self, interval_seconds: float) -> None:
        while not self._cleanup_stop.wait(interval_seconds):
            try:
                self.cleanup_once()
            except Exception:  # pragma: no cover - defensive logging in a daemon
                LOGGER.exception("Rolling-buffer cleanup failed")

    def stop_cleanup(self) -> None:
        """Stop and join the background cleanup worker."""
        self._cleanup_stop.set()
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5.0)
            self._cleanup_thread = None

    def get_segments(self, start_utc: datetime, end_utc: datetime) -> list[SegmentRecord]:
        """Return playable segments that overlap the requested UTC interval."""
        start = ensure_utc(start_utc)
        end = ensure_utc(end_utc)
        if end <= start:
            raise ValueError("end_utc must be later than start_utc")
        candidates = self.manifest.records_with_status(PLAYABLE_SEGMENT_STATUSES)
        return [
            record
            for record in candidates
            if record.wall_start_utc < end and record.wall_end_utc > start
        ]

    def get_discontinuities(
        self, start_utc: datetime, end_utc: datetime
    ) -> list[SessionBoundary]:
        """Return legacy markers and real session changes within a UTC range.

        A new FFmpeg process receives a new session ID.  Therefore the first
        complete segment after a reconnect forms a boundary even if the stream
        resumes quickly enough that timestamp arithmetic alone appears
        contiguous.
        """
        start = ensure_utc(start_utc)
        end = ensure_utc(end_utc)
        if end <= start:
            raise ValueError("end_utc must be later than start_utc")

        records = self.manifest.all_records()
        boundaries: list[SessionBoundary] = []
        previous_by_source: dict[str, SegmentRecord] = {}
        for record in records:
            if record.status != "complete":
                continue
            previous = previous_by_source.get(record.source_id)
            if previous is not None and previous.session_id != record.session_id:
                if start <= record.wall_start_utc < end:
                    boundaries.append(
                        SessionBoundary(
                            boundary_utc=record.wall_start_utc,
                            source_id=record.source_id,
                            previous_session_id=previous.session_id,
                            next_session_id=record.session_id,
                            previous_segment_id=previous.id,
                            next_segment_id=record.id,
                        )
                    )
            previous_by_source[record.source_id] = record

        for record in records:
            if record.status == "discontinuity" and start <= record.wall_start_utc < end:
                boundaries.append(
                    SessionBoundary(
                        boundary_utc=record.wall_start_utc,
                        source_id=record.source_id,
                        previous_session_id=None,
                        next_session_id=None,
                        previous_segment_id=None,
                        next_segment_id=None,
                    )
                )
        return sorted(boundaries, key=lambda boundary: (boundary.boundary_utc, boundary.source_id))

    def concatenate(
        self,
        start_utc: datetime,
        end_utc: datetime,
        output_path: Path,
        *,
        ffmpeg_binary: str = "ffmpeg",
    ) -> list[SegmentRecord]:
        """Concatenate segments covering a time range using FFmpeg's concat demuxer.

        Stream-copying is fast and preserves source quality.  As with all
        copy-mode concat jobs, the inputs must have compatible codecs and time
        bases; callers can choose an ``.ts`` or ``.mp4`` output filename.
        """
        records = self.get_segments(start_utc, end_utc)
        if not records:
            raise ValueError("no playable segments cover the requested time range")
        if any(record.path is None for record in records):
            raise ValueError("cannot concatenate a discontinuity")

        output = output_path.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        list_file_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", suffix=".ffconcat", delete=False
            ) as list_file:
                list_file_name = list_file.name
                for record in records:
                    assert record.path is not None
                    # The concat format escapes literal single quotes this way.
                    escaped = record.path.resolve().as_posix().replace("'", r"'\\''")
                    list_file.write(f"file '{escaped}'\n")
            subprocess.run(
                [
                    ffmpeg_binary,
                    "-hide_banner",
                    "-n",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    list_file_name,
                    "-c",
                    "copy",
                    str(output),
                ],
                check=True,
            )
        finally:
            if list_file_name:
                try:
                    os.unlink(list_file_name)
                except FileNotFoundError:
                    pass
        return records

    def close(self) -> None:
        """Release background resources."""
        self.stop_cleanup()

