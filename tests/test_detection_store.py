from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from app.detection.models import FeatureBin
from app.detection.store import DetectionStore
from app.ingestion.rolling_buffer import RollingBuffer


def test_feature_persistence_marks_segment_and_allows_replay(tmp_path: Path) -> None:
    buffer = RollingBuffer(tmp_path / "segments", retention_seconds=60)
    path = buffer.segments_dir / "segment.ts"
    path.write_bytes(b"segment")
    start = datetime(2026, 1, 1, tzinfo=UTC)
    buffer.manifest.add_writing(path, start, "source", "session")
    assert buffer.manifest.complete(path, 6)
    segment = buffer.manifest.get_by_path(path)
    assert segment is not None

    store = DetectionStore(buffer.manifest.database_path)
    assert [item.id for item in store.pending_segments()] == [segment.id]
    feature = FeatureBin(
        segment_id=segment.id,
        source_id="source",
        session_id="session",
        wall_utc=start,
        bin_seconds=1,
        audio_rms_db=-20,
        voice_energy_db=-25,
        onset_strength=1,
        motion=2,
        cut_change=0.2,
    )
    store.save_features_and_mark_analyzed(segment, [feature])
    assert store.pending_segments() == []
    restored = store.session_features("source", "session")
    assert restored == [feature]
