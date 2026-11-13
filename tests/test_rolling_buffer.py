from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.ingestion.rolling_buffer import RollingBuffer


def add_complete(buffer: RollingBuffer, path: Path, start: datetime, duration: float) -> int:
    path.write_bytes(b"synthetic-segment")
    buffer.manifest.add_writing(path, start, "test-source")
    assert buffer.manifest.complete(path, duration)
    record = buffer.manifest.get_by_path(path)
    assert record is not None
    return record.id


def test_cleanup_respects_writing_and_pinned_segments(tmp_path: Path) -> None:
    buffer = RollingBuffer(tmp_path / "segments", retention_seconds=10)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    old = now - timedelta(seconds=40)

    expired_path = buffer.segments_dir / "expired.ts"
    expired_id = add_complete(buffer, expired_path, old, 6)

    pinned_path = buffer.segments_dir / "pinned.ts"
    pinned_id = add_complete(buffer, pinned_path, old, 6)
    assert buffer.pin([pinned_id]) == 1

    writing_path = buffer.segments_dir / "writing.ts"
    writing_path.write_bytes(b"partial")
    buffer.manifest.add_writing(writing_path, old, "test-source")

    abandoned_path = buffer.segments_dir / "abandoned.ts"
    abandoned_path.write_bytes(b"partial")
    buffer.manifest.add_writing(abandoned_path, old, "other-source")
    assert buffer.manifest.abandon_writing("other-source") == 1
    buffer.manifest.add_discontinuity(old, "test-source")

    assert buffer.cleanup_once(now) == 3
    assert not expired_path.exists()
    assert not abandoned_path.exists()
    assert pinned_path.exists()
    assert writing_path.exists()

    remaining = {record.status for record in buffer.manifest.all_records()}
    assert remaining == {"pinned", "writing"}
    assert buffer.manifest.get_by_path(expired_path) is None
    assert buffer.manifest.get_by_path(pinned_path) is not None
    assert expired_id != pinned_id


def test_manifest_range_query_is_ordered_and_includes_overlaps(tmp_path: Path) -> None:
    buffer = RollingBuffer(tmp_path / "segments", retention_seconds=60)
    origin = datetime(2026, 1, 1, tzinfo=UTC)
    paths = [buffer.segments_dir / f"segment-{index}.ts" for index in range(3)]
    for index, path in enumerate(paths):
        add_complete(buffer, path, origin + timedelta(seconds=index * 10), 10)

    selected = buffer.get_segments(
        origin + timedelta(seconds=5), origin + timedelta(seconds=25)
    )
    assert [record.path for record in selected] == paths

    boundary_selected = buffer.get_segments(
        origin + timedelta(seconds=10), origin + timedelta(seconds=20)
    )
    assert [record.path for record in boundary_selected] == [paths[1]]
