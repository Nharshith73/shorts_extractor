from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.ingestion.rolling_buffer import SegmentRecord
from scripts.soak_test import (
    build_retention_summary,
    build_source_session_summaries,
    build_timing_summary,
)


def make_record(path: Path, start: datetime) -> SegmentRecord:
    return SegmentRecord(
        id=1,
        path=path.resolve(),
        wall_start_utc=start,
        fs_created_utc=start,
        duration=6.0,
        status="complete",
        source_id="test-source",
        pinned=False,
        session_id="test-session",
        timing_reanchored=False,
    )


def test_soak_retention_allows_cleanup_and_segment_grace(tmp_path: Path) -> None:
    observed = datetime(2026, 1, 1, tzinfo=UTC)
    segment_path = tmp_path / "segment.ts"
    segment_path.write_bytes(b"segment")
    record = make_record(segment_path, observed - timedelta(seconds=930))

    summary = build_retention_summary(
        [record],
        tmp_path,
        observed_utc=observed,
        retention_seconds=900,
        cleanup_interval_seconds=30,
        segment_seconds=6,
        total_segments_produced=300,
        total_deleted=145,
    )

    assert summary.oldest_retained_segment_age_seconds == 930
    assert summary.maximum_allowed_age_seconds == 936
    assert summary.files_match_manifest_rows is True
    assert summary.retention_confirmed is True


def test_soak_retention_rejects_old_or_untracked_files(tmp_path: Path) -> None:
    observed = datetime.now(UTC)
    segment_path = tmp_path / "segment.ts"
    segment_path.write_bytes(b"segment")
    record = make_record(segment_path, observed - timedelta(seconds=937))

    old_summary = build_retention_summary(
        [record],
        tmp_path,
        observed_utc=observed,
        retention_seconds=900,
        cleanup_interval_seconds=30,
        segment_seconds=6,
        total_segments_produced=1,
        total_deleted=0,
    )
    assert old_summary.retention_confirmed is False

    untracked_path = tmp_path / "untracked.ts"
    untracked_path.write_bytes(b"segment")
    unmatched_summary = build_retention_summary(
        [],
        tmp_path,
        observed_utc=observed,
        retention_seconds=900,
        cleanup_interval_seconds=30,
        segment_seconds=6,
        total_segments_produced=1,
        total_deleted=0,
    )
    assert unmatched_summary.files_match_manifest_rows is False
    assert unmatched_summary.retention_confirmed is False


def test_soak_retention_rejects_manifest_row_without_its_file(tmp_path: Path) -> None:
    observed = datetime(2026, 1, 1, tzinfo=UTC)
    segment_path = tmp_path / "missing.ts"
    segment_path.write_bytes(b"segment")
    record = make_record(segment_path, observed - timedelta(seconds=10))
    segment_path.unlink()

    summary = build_retention_summary(
        [record],
        tmp_path,
        observed_utc=observed,
        retention_seconds=900,
        cleanup_interval_seconds=30,
        segment_seconds=6,
        total_segments_produced=1,
        total_deleted=0,
    )

    assert summary.manifest_rows_missing_files == 1
    assert summary.files_match_manifest_rows is False
    assert summary.retention_confirmed is False


def test_timing_and_source_session_summaries_use_persisted_fields(tmp_path: Path) -> None:
    origin = datetime(2026, 1, 1, tzinfo=UTC)
    path = tmp_path / "segment.ts"
    reanchored_path = tmp_path / "reanchored.ts"
    path.write_bytes(b"segment")
    reanchored_path.write_bytes(b"segment")
    record = make_record(path, origin)
    reanchored_record = SegmentRecord(
        id=2,
        path=reanchored_path.resolve(),
        wall_start_utc=origin + timedelta(seconds=8),
        fs_created_utc=origin + timedelta(seconds=6),
        duration=6.0,
        status="complete",
        source_id="test-source",
        pinned=False,
        session_id="test-session",
        timing_reanchored=True,
    )

    timing = build_timing_summary([record, reanchored_record], origin + timedelta(seconds=15))
    assert timing.max_wall_vs_filesystem_delta_seconds == 2.0
    assert timing.last_computed_wall_end_utc == origin + timedelta(seconds=14)
    assert timing.last_wall_end_to_capture_end_gap_seconds == 1.0

    source_sessions = build_source_session_summaries([record, reanchored_record])
    assert source_sessions[0].manifest_rows == 2
    assert source_sessions[0].ts_files_on_disk == 2
