from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.ingestion.rolling_buffer import RollingBuffer


def add_complete(
    buffer: RollingBuffer,
    path: Path,
    start: datetime,
    duration: float,
    *,
    source_id: str = "test-source",
    session_id: str = "session-a",
) -> int:
    path.write_bytes(b"synthetic-segment")
    buffer.manifest.add_writing(path, start, source_id, session_id)
    assert buffer.manifest.complete(path, duration)
    record = buffer.manifest.get_by_path(path)
    assert record is not None
    return record.id


def test_pins_persist_across_rolling_buffer_reinstantiation(tmp_path: Path) -> None:
    segments_dir = tmp_path / "segments"
    origin = datetime(2026, 1, 1, tzinfo=UTC)
    first_buffer = RollingBuffer(segments_dir, retention_seconds=60)
    segment_id = add_complete(first_buffer, segments_dir / "pinned.ts", origin, 6)
    assert first_buffer.pin([segment_id]) == 1

    second_buffer = RollingBuffer(segments_dir, retention_seconds=60)
    record = second_buffer.manifest.get_by_path(segments_dir / "pinned.ts")
    assert record is not None
    assert record.status == "complete"
    assert record.pinned is True


def test_retention_deletes_abandoned_but_not_pinned_or_writing(tmp_path: Path) -> None:
    buffer = RollingBuffer(tmp_path / "segments", retention_seconds=10)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    old = now - timedelta(seconds=40)

    expired_path = buffer.segments_dir / "expired.ts"
    add_complete(buffer, expired_path, old, 6)

    pinned_path = buffer.segments_dir / "pinned.ts"
    pinned_id = add_complete(buffer, pinned_path, old, 6)
    assert buffer.pin([pinned_id]) == 1

    writing_path = buffer.segments_dir / "writing.ts"
    writing_path.write_bytes(b"partial")
    buffer.manifest.add_writing(writing_path, old, "test-source", "session-a")

    abandoned_path = buffer.segments_dir / "abandoned.ts"
    abandoned_path.write_bytes(b"partial")
    buffer.manifest.add_writing(abandoned_path, old, "other-source", "session-b")
    assert buffer.manifest.abandon_writing("other-source", "session-b") == 1

    assert buffer.cleanup_once(now) == 2
    assert buffer.total_deleted_rows == 2
    assert not expired_path.exists()
    assert not abandoned_path.exists()
    assert pinned_path.exists()
    assert writing_path.exists()

    remaining = {record.path: record for record in buffer.manifest.all_records()}
    assert remaining[pinned_path.resolve()].pinned is True
    assert remaining[writing_path.resolve()].status == "writing"


def test_retention_removes_expired_row_when_file_is_already_missing(tmp_path: Path) -> None:
    buffer = RollingBuffer(tmp_path / "segments", retention_seconds=10)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    path = buffer.segments_dir / "missing.ts"
    add_complete(buffer, path, now - timedelta(seconds=40), 6)
    path.unlink()

    assert buffer.cleanup_once(now) == 1
    assert buffer.manifest.get_by_path(path) is None
    assert buffer.total_deleted_rows == 1


def test_migrates_old_manifest_without_losing_rows(tmp_path: Path) -> None:
    segments_dir = tmp_path / "segments"
    segments_dir.mkdir()
    database_path = segments_dir / "manifest.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE segments (
                id INTEGER PRIMARY KEY,
                path TEXT,
                wall_start_utc TEXT NOT NULL,
                duration REAL NOT NULL DEFAULT 0,
                status TEXT NOT NULL,
                source_id TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO segments (id, path, wall_start_utc, duration, status, source_id)
            VALUES (7, ?, '2026-01-01T00:00:00.000000Z', 6.5, 'pinned', 'old-source')
            """,
            (str((segments_dir / "old.ts").resolve()),),
        )

    buffer = RollingBuffer(segments_dir, retention_seconds=60)
    record = buffer.manifest.get_by_path(segments_dir / "old.ts")
    assert record is not None
    assert record.id == 7
    assert record.duration == 6.5
    assert record.source_id == "old-source"
    assert record.status == "complete"
    assert record.pinned is True
    assert record.session_id == "legacy-old-source"
    assert record.fs_created_utc == record.wall_start_utc
    assert record.timing_reanchored is False

    with sqlite3.connect(database_path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(segments)")}
        assert {"pinned", "session_id", "fs_created_utc", "timing_reanchored"} <= columns
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO segments
                (path, wall_start_utc, duration, status, source_id, pinned, session_id)
                VALUES (?, '2026-01-01T00:00:10.000000Z', 6, 'complete', 'old-source', 0, 'legacy-old-source')
                """,
                (str((segments_dir / "old.ts").resolve()),),
            )


def test_manifest_range_query_and_session_discontinuities(tmp_path: Path) -> None:
    buffer = RollingBuffer(tmp_path / "segments", retention_seconds=60)
    origin = datetime(2026, 1, 1, tzinfo=UTC)
    paths = [buffer.segments_dir / f"segment-{index}.ts" for index in range(3)]
    for index, path in enumerate(paths):
        add_complete(
            buffer,
            path,
            origin + timedelta(seconds=index * 10),
            10,
            session_id="session-a" if index < 2 else "session-b",
        )

    selected = buffer.get_segments(
        origin + timedelta(seconds=5), origin + timedelta(seconds=25)
    )
    assert [record.path for record in selected] == paths

    boundaries = buffer.get_discontinuities(origin, origin + timedelta(seconds=30))
    assert len(boundaries) == 1
    boundary = boundaries[0]
    assert boundary.boundary_utc == origin + timedelta(seconds=20)
    assert boundary.previous_session_id == "session-a"
    assert boundary.next_session_id == "session-b"

    boundary_selected = buffer.get_segments(
        origin + timedelta(seconds=10), origin + timedelta(seconds=20)
    )
    assert [record.path for record in boundary_selected] == [paths[1]]


def test_manifest_preserves_filesystem_time_when_timing_reanchors(tmp_path: Path) -> None:
    buffer = RollingBuffer(tmp_path / "segments", retention_seconds=60)
    filesystem_time = datetime(2026, 1, 1, tzinfo=UTC)
    corrected_time = filesystem_time + timedelta(seconds=3)
    path = buffer.segments_dir / "reanchored.ts"
    path.write_bytes(b"synthetic-segment")

    buffer.manifest.add_writing(
        path,
        filesystem_time,
        "test-source",
        "session-a",
        fs_created_utc=filesystem_time,
    )
    assert buffer.manifest.complete(
        path,
        6,
        wall_start_utc=corrected_time,
        timing_reanchored=True,
    )

    record = buffer.manifest.get_by_path(path)
    assert record is not None
    assert record.fs_created_utc == filesystem_time
    assert record.wall_start_utc == corrected_time
    assert record.timing_reanchored is True
