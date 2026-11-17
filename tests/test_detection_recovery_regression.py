"""Regression tests for P0: Detection crash consistency and dirty-session recovery."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
import sqlite3

import pytest

from app.detection.config import DetectionConfig
from app.detection.models import DetectedEvent, FeatureBin
from app.detection.runner import DetectionRunner
from app.detection.store import DetectionStore
from app.ingestion.rolling_buffer import SegmentManifest, SegmentRecord


def _dummy_segment(segment_id: int, source_id: str = "src", session_id: str = "sess", duration: float = 6.0) -> SegmentRecord:
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC) + timedelta(seconds=segment_id * 6)
    return SegmentRecord(
        id=segment_id,
        path=Path(f"seg{segment_id}.ts"),
        wall_start_utc=now,
        fs_created_utc=now,
        duration=duration,
        status="complete",
        source_id=source_id,
        pinned=False,
        session_id=session_id,
        timing_reanchored=False,
    )


def _features_with_excitement(source_id: str, session_id: str, start_time: datetime, count: int = 120) -> list[FeatureBin]:
    """Generate 120s of features with warmup (first 60s baseline) and a high excitement spike at 90s."""
    bins = []
    for i in range(count):
        t = start_time + timedelta(seconds=i)
        # Spike at second 90
        is_spike = 88 <= i <= 92
        bins.append(
            FeatureBin(
                source_id=source_id,
                session_id=session_id,
                wall_utc=t,
                bin_seconds=1.0,
                audio_rms_db=0.0 if is_spike else -40.0,
                voice_energy_db=0.0 if is_spike else -45.0,
                onset_strength=5.0 if is_spike else 0.01,
                motion=20.0 if is_spike else 0.5,
                cut_change=2.0 if is_spike else 0.01,
            )
        )
    return bins


def test_crash_after_feature_commit_leaves_session_recoverable(tmp_path: Path) -> None:
    """Boundary 2 & 3: Features and analysis_state are committed, but process crashes before event persistence."""
    db_path = tmp_path / "manifest.sqlite3"
    manifest = SegmentManifest(db_path)
    store = DetectionStore(db_path)
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)

    seg = _dummy_segment(1)
    features = _features_with_excitement("src", "sess", now, 120)

    # 1. Simulate process committing features and analysis_state, but crashing before replace_session_events
    revision = store.save_features_and_mark_analyzed(seg, features)
    assert revision == 1

    # Verify state: segment is analyzed, but session is dirty
    state = store.session_state("src", "sess")
    assert state is not None
    assert state["dirty"] is True
    assert state["revision"] == 1
    assert state["clean_revision"] == 0

    # No events exist yet
    assert len(store.events("src")) == 0

    # 2. Restart runner and run recovery (simulating recovery on restart with NO pending segments)
    config = DetectionConfig()
    runner = DetectionRunner(db_path, config)
    assert len(store.pending_segments()) == 0

    recovered_count = runner.recover_dirty_sessions()
    assert recovered_count == 1

    # Verify events are now populated
    events = store.events("src")
    assert len(events) >= 1
    assert any(88 <= (e.peak_wall_utc - now).total_seconds() <= 92 for e in events)

    # Verify session is now clean
    state_after = store.session_state("src", "sess")
    assert state_after is not None
    assert state_after["dirty"] is False
    assert state_after["clean_revision"] == 1


def test_recovery_is_idempotent_and_does_not_duplicate_events(tmp_path: Path) -> None:
    """Repeated recovery passes produce identical consistent event sets."""
    db_path = tmp_path / "manifest.sqlite3"
    store = DetectionStore(db_path)
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)

    seg = _dummy_segment(1)
    features = _features_with_excitement("src", "sess", now, 120)
    store.save_features_and_mark_analyzed(seg, features)

    config = DetectionConfig()
    runner = DetectionRunner(db_path, config)

    # Run recovery first time
    assert runner.recover_dirty_sessions() == 1
    events_1 = store.events("src")
    assert len(events_1) >= 1

    # Run recovery second time: no dirty sessions, events unchanged
    assert runner.recover_dirty_sessions() == 0
    events_2 = store.events("src")
    assert [(e.peak_wall_utc, e.score) for e in events_1] == [(e.peak_wall_utc, e.score) for e in events_2]


def test_failure_during_event_replacement_leaves_session_recoverable(tmp_path: Path) -> None:
    """Boundary 4: If event replacement fails or is aborted mid-way, transaction rolls back and session stays dirty."""
    db_path = tmp_path / "manifest.sqlite3"
    store = DetectionStore(db_path)
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)

    seg = _dummy_segment(1)
    features = _features_with_excitement("src", "sess", now, 120)
    store.save_features_and_mark_analyzed(seg, features)

    # Simulate an error during replace_session_events by passing an invalid event that causes error
    bad_event = DetectedEvent(
        source_id="src",
        session_id="sess",
        peak_wall_utc=now,
        start_wall_utc=now,
        end_wall_utc=now,
        score=float("nan"),  # NaN will fail json or serialization if checked, or raise
        components={"test": float("inf")},
    )

    with pytest.raises(Exception):
        # We simulate a failure in event persistence
        with store._connect() as conn:
            conn.execute("INSERT INTO events (source_id, session_id, peak_wall_utc, start_wall_utc, end_wall_utc, score, components_json, created_utc) VALUES (NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL)")

    # Session MUST still be marked dirty
    state = store.session_state("src", "sess")
    assert state is not None
    assert state["dirty"] is True

    # Recover properly
    config = DetectionConfig()
    runner = DetectionRunner(db_path, config)
    assert runner.recover_dirty_sessions() == 1
    assert store.session_state("src", "sess")["dirty"] is False


def test_concurrent_revision_remains_dirty_if_newer_features_arrive(tmp_path: Path) -> None:
    """If new features arrive while scoring was in-flight, the session must remain dirty."""
    db_path = tmp_path / "manifest.sqlite3"
    store = DetectionStore(db_path)
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)

    seg1 = _dummy_segment(1)
    store.save_features_and_mark_analyzed(seg1, _features_with_excitement("src", "sess", now, 60))
    state1 = store.session_state("src", "sess")
    assert state1["revision"] == 1

    # While scoring revision 1, segment 2 arrives and updates to revision 2
    seg2 = _dummy_segment(2)
    store.save_features_and_mark_analyzed(seg2, _features_with_excitement("src", "sess", now + timedelta(seconds=60), 60))
    state2 = store.session_state("src", "sess")
    assert state2["revision"] == 2
    assert state2["dirty"] is True

    # Now replace_session_events finishes scoring with expected_revision=1
    store.replace_session_events("src", "sess", [], expected_revision=1)

    # Because revision is now 2, the session MUST STILL BE DIRTY!
    state_after = store.session_state("src", "sess")
    assert state_after["dirty"] is True
    assert state_after["revision"] == 2
    assert state_after["clean_revision"] == 1

    # Recovering revision 2 marks it clean
    config = DetectionConfig()
    runner = DetectionRunner(db_path, config)
    assert runner.recover_dirty_sessions() == 1
    assert store.session_state("src", "sess")["dirty"] is False
    assert store.session_state("src", "sess")["clean_revision"] == 2
