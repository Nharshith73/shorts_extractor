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

PLAYABLE_SEGMENT_STATUSES = ("complete", "pinned")


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

    ``path`` is ``None`` only for a ``discontinuity`` entry.  ``wall_start_utc``
    is the real-world capture time rather than a source PTS or filename counter.
    """

    id: int
    path: Path | None
    wall_start_utc: datetime
    duration: float
    status: str
    source_id: str

    @property
    def wall_end_utc(self) -> datetime:
        return self.wall_start_utc + timedelta(seconds=self.duration)


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
                    duration REAL NOT NULL DEFAULT 0,
                    status TEXT NOT NULL,
                    source_id TEXT NOT NULL
                )
                """
            )
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

    @staticmethod
    def _record(row: sqlite3.Row) -> SegmentRecord:
        raw_path = row["path"]
        return SegmentRecord(
            id=int(row["id"]),
            path=Path(raw_path) if raw_path else None,
            wall_start_utc=from_utc_text(str(row["wall_start_utc"])),
            duration=float(row["duration"]),
            status=str(row["status"]),
            source_id=str(row["source_id"]),
        )

    def add_writing(
        self, path: Path, wall_start_utc: datetime, source_id: str
    ) -> None:
        """Record a file as in-progress, without replacing a known row."""
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO segments
                (path, wall_start_utc, duration, status, source_id)
                VALUES (?, ?, 0, 'writing', ?)
                """,
                (str(path.resolve()), to_utc_text(wall_start_utc), source_id),
            )

    def complete(self, path: Path, duration: float) -> bool:
        """Atomically mark an observed file playable once it has closed."""
        if duration <= 0:
            raise ValueError("a completed segment must have a positive duration")
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE segments
                SET duration = ?, status = 'complete'
                WHERE path = ? AND status = 'writing'
                """,
                (duration, str(path.resolve())),
            )
        return cursor.rowcount == 1

    def abandon_writing(self, source_id: str) -> int:
        """Flag unclosed files as unusable after a process stop/restart."""
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE segments SET status = 'abandoned'
                WHERE source_id = ? AND status = 'writing'
                """,
                (source_id,),
            )
        return cursor.rowcount

    def add_discontinuity(self, wall_start_utc: datetime, source_id: str) -> None:
        """Add a zero-duration gap marker to the same manifest table."""
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO segments
                (path, wall_start_utc, duration, status, source_id)
                VALUES (NULL, ?, 0, 'discontinuity', ?)
                """,
                (to_utc_text(wall_start_utc), source_id),
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
                f"UPDATE segments SET status = 'pinned' "
                f"WHERE id IN ({placeholders}) AND status IN ('complete', 'pinned')",
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
                f"UPDATE segments SET status = 'complete' "
                f"WHERE id IN ({placeholders}) AND status = 'pinned'",
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
            if record.wall_end_utc >= cutoff:
                continue
            try:
                if record.path is not None and record.path.exists():
                    record.path.unlink()
                self.manifest.delete_record(record.id)
                removed += 1
                LOGGER.debug("Removed expired manifest entry %s", record.id)
            except OSError as exc:
                LOGGER.warning("Could not delete expired segment %s: %s", record.path, exc)
        return removed

    def start_cleanup(self, interval_seconds: float | None = None) -> None:
        """Start periodic retention cleanup in a daemon thread."""
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            return
        interval = interval_seconds or max(1.0, min(30.0, self.retention_seconds / 10))
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

