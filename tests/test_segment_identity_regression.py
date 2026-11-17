"""Regression tests for P0: Segment identity uniqueness and migration safety."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
import sqlite3

import pytest

from app.detection.models import FeatureBin
from app.detection.store import DetectionStore
from app.ingestion.rolling_buffer import RollingBuffer, SegmentManifest, to_utc_text


def _sample_feature(source_id: str, session_id: str, wall_utc: datetime) -> FeatureBin:
    return FeatureBin(
        source_id=source_id,
        session_id=session_id,
        wall_utc=wall_utc,
        bin_seconds=1.0,
        audio_rms_db=-20.0,
        voice_energy_db=-25.0,
        onset_strength=0.1,
        motion=2.0,
        cut_change=0.05,
    )


def test_segment_id_not_reused_after_retention_deletion(tmp_path: Path) -> None:
    """Cover requirements 1-6: Capture, analyze, delete, capture new, verify identity & pending."""
    buffer = RollingBuffer(tmp_path / "segments", retention_seconds=10)
    store = DetectionStore(buffer.manifest.database_path)
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)

    # 1. Capture segment 1
    p1 = tmp_path / "segments" / "seg1.ts"
    p1.touch()
    buffer.manifest.add_writing(p1, now, "src1", "sess1")
    assert buffer.manifest.complete(p1, 6.0)

    pending = store.pending_segments()
    assert len(pending) == 1
    seg1 = pending[0]
    seg1_id = seg1.id

    # Analyze segment 1
    store.save_features_and_mark_analyzed(seg1, [_sample_feature("src1", "sess1", now)])
    assert len(store.pending_segments()) == 0

    # 2. Retain/delete segment 1
    deleted = buffer.cleanup_once(now + timedelta(seconds=20))
    assert deleted == 1
    assert buffer.manifest.get_by_path(p1) is None

    # 3. Capture new segment 2
    p2 = tmp_path / "segments" / "seg2.ts"
    p2.touch()
    buffer.manifest.add_writing(p2, now + timedelta(seconds=30), "src1", "sess1")
    assert buffer.manifest.complete(p2, 6.0)

    # 4. Verify the new identity differs from previous identity
    seg2 = buffer.manifest.get_by_path(p2)
    assert seg2 is not None
    assert seg2.id != seg1_id
    assert seg2.id > seg1_id

    # 5. Verify the new segment appears in pending_segments()
    pending_after = store.pending_segments()
    assert len(pending_after) == 1
    assert pending_after[0].id == seg2.id

    # 6. Verify historical analysis_state record does not suppress the new segment
    with store._connect() as conn:
        state_rows = conn.execute("SELECT * FROM analysis_state").fetchall()
        assert len(state_rows) == 1
        assert state_rows[0]["segment_id"] == seg1_id


def test_completely_emptied_segments_table_does_not_reuse_id(tmp_path: Path) -> None:
    """Cover requirement 7: A completely emptied table never reuses an allocated ID."""
    manifest = SegmentManifest(tmp_path / "manifest.sqlite3")
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)

    # Insert 3 segments
    for i in range(1, 4):
        p = tmp_path / f"seg{i}.ts"
        p.touch()
        manifest.add_writing(p, now + timedelta(seconds=i * 6), "src1", "sess1")
        manifest.complete(p, 6.0)

    records = manifest.all_records()
    assert [r.id for r in records] == [1, 2, 3]

    # Delete all rows from segments
    for r in records:
        manifest.delete_record(r.id)
    assert len(manifest.all_records()) == 0

    # Insert a new segment into the completely empty table
    p4 = tmp_path / "seg4.ts"
    p4.touch()
    manifest.add_writing(p4, now + timedelta(seconds=24), "src1", "sess1")
    manifest.complete(p4, 6.0)

    rec4 = manifest.get_by_path(p4)
    assert rec4 is not None
    # Must be 4, NOT 1
    assert rec4.id == 4


def test_database_migration_preserves_existing_rows_and_metadata(tmp_path: Path) -> None:
    """Cover requirement 8: Migration from pre-autoincrement schema preserves all data and constraints."""
    db_path = tmp_path / "legacy.sqlite3"
    now_str = to_utc_text(datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC))

    # Create pre-migration table without AUTOINCREMENT
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE segments (
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
        # Insert historical row 42
        conn.execute(
            """
            INSERT INTO segments (id, path, wall_start_utc, fs_created_utc, duration, status, source_id, pinned, session_id, timing_reanchored)
            VALUES (42, 'path/to/seg42.ts', ?, ?, 6.0, 'complete', 'src-legacy', 1, 'sess-legacy', 0)
            """,
            (now_str, now_str),
        )

    # Open with SegmentManifest to trigger migration
    manifest = SegmentManifest(db_path)
    records = manifest.all_records()
    assert len(records) == 1
    r = records[0]
    assert r.id == 42
    assert r.path == Path("path/to/seg42.ts")
    assert r.duration == 6.0
    assert r.status == "complete"
    assert r.source_id == "src-legacy"
    assert r.pinned is True
    assert r.session_id == "sess-legacy"
    assert r.timing_reanchored is False

    # Check AUTOINCREMENT in schema
    with manifest._connect() as conn:
        sql = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='segments'").fetchone()[0]
        assert "AUTOINCREMENT" in sql.upper()

    # Capture next segment and ensure ID is > 42
    p_new = tmp_path / "seg_new.ts"
    p_new.touch()
    manifest.add_writing(p_new, datetime(2026, 1, 1, 12, 1, 0, tzinfo=UTC), "src-legacy", "sess-legacy")
    manifest.complete(p_new, 6.0)
    rec_new = manifest.get_by_path(p_new)
    assert rec_new is not None
    assert rec_new.id == 43


def test_migration_idempotent_running_twice_does_not_damage(tmp_path: Path) -> None:
    """Cover requirement 9: Running migration repeatedly is safe and idempotent."""
    manifest = SegmentManifest(tmp_path / "manifest.sqlite3")
    p = tmp_path / "seg.ts"
    p.touch()
    manifest.add_writing(p, datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC), "src", "sess")
    manifest.complete(p, 6.0)

    # Call _migrate_schema explicitly multiple times
    with manifest._connect() as conn:
        manifest._migrate_schema(conn)
        manifest._migrate_schema(conn)

    records = manifest.all_records()
    assert len(records) == 1
    assert records[0].id == 1
    assert records[0].path == p.resolve()
