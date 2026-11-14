"""Run the Stage 1 30-minute local-source acceptance test.

The default is deliberately long.  It creates a deterministic local test video,
captures it in real time, and prints a compact report when finished.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.ingestion.rolling_buffer import SegmentRecord  # noqa: E402
from app.ingestion.stream_reader import StreamConfig, StreamReader, probe_duration  # noqa: E402


@dataclass(frozen=True, slots=True)
class RetentionSummary:
    """Observable retention state at the end of one soak capture."""

    retention_seconds: float
    cleanup_interval_seconds: float
    total_segments_produced: int
    total_deleted: int
    manifest_rows: int
    ts_files_on_disk: int
    manifest_rows_missing_files: int
    untracked_ts_files: int
    oldest_retained_segment_age_seconds: float | None
    maximum_allowed_age_seconds: float
    files_match_manifest_rows: bool
    retention_confirmed: bool


@dataclass(frozen=True, slots=True)
class TimingSummary:
    """Timing diagnostics based on persisted segment timestamps."""

    max_wall_vs_filesystem_delta_seconds: float
    last_computed_wall_end_utc: datetime | None
    actual_capture_end_utc: datetime
    last_wall_end_to_capture_end_gap_seconds: float | None


@dataclass(frozen=True, slots=True)
class SourceSessionSummary:
    """Manifest and disk counts for one source/session pair."""

    source_id: str
    session_id: str
    manifest_rows: int
    ts_files_on_disk: int


def build_retention_summary(
    records: Sequence[SegmentRecord],
    segments_dir: Path,
    *,
    observed_utc: datetime,
    retention_seconds: float,
    cleanup_interval_seconds: float,
    segment_seconds: float,
    total_segments_produced: int,
    total_deleted: int,
) -> RetentionSummary:
    """Measure retention with the same wall-clock basis used by the manifest.

    A buffer is allowed to retain one full cleanup interval of expired work and
    one still-closing segment.  Files represented by a manifest row use that
    row's capture-start timestamp; untracked files use their filesystem
    creation timestamp as a conservative diagnostic fallback.
    """
    observed = observed_utc.astimezone(UTC)
    manifest_paths = {
        record.path.resolve()
        for record in records
        if record.path is not None
    }
    files_on_disk = {path.resolve() for path in segments_dir.glob("*.ts")}
    records_by_path = {
        record.path.resolve(): record
        for record in records
        if record.path is not None
    }

    row_ages = [
        max(0.0, (observed - record.wall_start_utc).total_seconds())
        for record in records
    ]
    file_ages: list[float] = []
    for path in files_on_disk:
        record = records_by_path.get(path)
        if record is not None:
            file_ages.append(max(0.0, (observed - record.wall_start_utc).total_seconds()))
            continue
        file_start = datetime.fromtimestamp(path.stat().st_ctime, UTC)
        file_ages.append(max(0.0, (observed - file_start).total_seconds()))

    maximum_allowed_age = retention_seconds + cleanup_interval_seconds + segment_seconds
    all_ages = [*row_ages, *file_ages]
    no_expired_row_or_file = all(age <= maximum_allowed_age for age in all_ages)
    files_match_manifest_rows = files_on_disk == manifest_paths
    manifest_rows_missing_files = sum(
        1 for record in records if record.path is not None and not record.path.exists()
    )
    return RetentionSummary(
        retention_seconds=retention_seconds,
        cleanup_interval_seconds=cleanup_interval_seconds,
        total_segments_produced=total_segments_produced,
        total_deleted=total_deleted,
        manifest_rows=len(records),
        ts_files_on_disk=len(files_on_disk),
        manifest_rows_missing_files=manifest_rows_missing_files,
        untracked_ts_files=len(files_on_disk - manifest_paths),
        oldest_retained_segment_age_seconds=max(file_ages, default=None),
        maximum_allowed_age_seconds=maximum_allowed_age,
        files_match_manifest_rows=files_match_manifest_rows,
        retention_confirmed=(
            no_expired_row_or_file
            and files_match_manifest_rows
            and manifest_rows_missing_files == 0
        ),
    )


def build_timing_summary(
    records: Sequence[SegmentRecord], actual_capture_end_utc: datetime
) -> TimingSummary:
    """Summarise correction size and end-of-capture wall-clock alignment."""
    completed = [record for record in records if record.status == "complete"]
    max_correction = max(
        (abs((record.wall_start_utc - record.fs_created_utc).total_seconds()) for record in completed),
        default=0.0,
    )
    last_record = max(completed, key=lambda record: record.wall_start_utc, default=None)
    capture_end = actual_capture_end_utc.astimezone(UTC)
    if last_record is None:
        return TimingSummary(max_correction, None, capture_end, None)
    computed_end = last_record.wall_end_utc
    return TimingSummary(
        max_correction,
        computed_end,
        capture_end,
        (capture_end - computed_end).total_seconds(),
    )


def build_source_session_summaries(
    records: Sequence[SegmentRecord],
) -> list[SourceSessionSummary]:
    """Count manifest rows and existing files for each source/session pair."""
    grouped: dict[tuple[str, str], list[SegmentRecord]] = {}
    for record in records:
        grouped.setdefault((record.source_id, record.session_id), []).append(record)
    return [
        SourceSessionSummary(
            source_id=source_id,
            session_id=session_id,
            manifest_rows=len(group_records),
            ts_files_on_disk=sum(
                1
                for record in group_records
                if record.path is not None and record.path.exists()
            ),
        )
        for (source_id, session_id), group_records in sorted(grouped.items())
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1 local 30-minute soak test")
    parser.add_argument("--duration-seconds", type=int, default=30 * 60)
    parser.add_argument("--segment-seconds", type=float, default=6)
    parser.add_argument("--retention-seconds", type=float, default=15 * 60)
    parser.add_argument("--work-dir", type=Path, default=ROOT / "data" / "soak-test")
    parser.add_argument("--source", type=Path, help="Use an existing local source instead of generating one")
    return parser.parse_args()


def require_ffmpeg() -> None:
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        raise RuntimeError("Install and add to PATH before running soak test: " + ", ".join(missing))


def recreate_work_dir(work_dir: Path) -> Path:
    """Remove prior soak artifacts, limited to this repository's data folder."""
    data_dir = (ROOT / "data").resolve()
    target = work_dir.resolve()
    if target == data_dir or not target.is_relative_to(data_dir):
        raise ValueError(f"work_dir must be a child of {data_dir}")
    if work_dir.is_symlink():
        work_dir.unlink()
    elif work_dir.exists():
        shutil.rmtree(work_dir)
    target.mkdir(parents=True, exist_ok=True)
    return target


def make_source(path: Path, duration_seconds: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=640x360:rate=30",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=880:sample_rate=48000",
            "-t",
            str(duration_seconds),
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-g",
            "30",
            "-keyint_min",
            "30",
            "-sc_threshold",
            "0",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ],
        check=True,
    )


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    require_ffmpeg()
    if args.duration_seconds <= 0 or args.retention_seconds <= 0:
        raise ValueError("duration and retention must be positive")

    requested_work_dir = args.work_dir.resolve()
    source = args.source.resolve() if args.source else None
    if source is not None and source.is_relative_to(requested_work_dir):
        raise ValueError("--source cannot be inside --work-dir because the work directory is reset")
    work_dir = recreate_work_dir(requested_work_dir)
    source = source or work_dir / "synthetic-source.mp4"
    if not args.source:
        logging.info("Generating %ss local source at %s", args.duration_seconds, source)
        make_source(source, args.duration_seconds)
    if not source.is_file():
        raise FileNotFoundError(source)

    segments_dir = work_dir / "segments"
    started_utc = datetime.now(UTC)
    reader = StreamReader(
        StreamConfig(
            source=str(source),
            segment_seconds=args.segment_seconds,
            retention_seconds=args.retention_seconds,
            segments_dir=segments_dir,
            reconnect_max_backoff_seconds=30,
            is_live=False,
        )
    )
    reader.run()
    finished_utc = datetime.now(UTC)

    manifest_records = reader.buffer.manifest.all_records()
    records = [record for record in manifest_records if record.status == "complete"]
    corrupt_or_missing: list[str] = []
    for record in records:
        if record.path is None or not record.path.exists():
            corrupt_or_missing.append(str(record.path))
            continue
        try:
            probe_duration(record.path)
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            corrupt_or_missing.append(f"{record.path}: {exc}")

    retention_summary = build_retention_summary(
        manifest_records,
        segments_dir,
        observed_utc=finished_utc,
        retention_seconds=args.retention_seconds,
        cleanup_interval_seconds=reader.buffer.cleanup_interval_seconds,
        segment_seconds=args.segment_seconds,
        total_segments_produced=reader.completed_segment_count,
        total_deleted=reader.buffer.total_deleted_rows,
    )
    timing_summary = build_timing_summary(manifest_records, finished_utc)
    source_session_summaries = build_source_session_summaries(manifest_records)

    logging.info("Stage 1 soak-test report")
    logging.info("  capture wall time: %.1fs", (finished_utc - started_utc).total_seconds())
    logging.info("  retention seconds: %.1f", retention_summary.retention_seconds)
    logging.info("  cleanup interval: %.1fs", retention_summary.cleanup_interval_seconds)
    logging.info("  total segments produced: %d", retention_summary.total_segments_produced)
    logging.info("  total deleted: %d", retention_summary.total_deleted)
    logging.info("  rows in manifest: %d", retention_summary.manifest_rows)
    logging.info("  .ts files on disk: %d", retention_summary.ts_files_on_disk)
    logging.info(
        "  manifest rows missing files: %d",
        retention_summary.manifest_rows_missing_files,
    )
    logging.info("  untracked .ts files: %d", retention_summary.untracked_ts_files)
    for summary in source_session_summaries:
        logging.info(
            "  source_id=%s session_id=%s rows=%d .ts_files=%d",
            summary.source_id,
            summary.session_id,
            summary.manifest_rows,
            summary.ts_files_on_disk,
        )
    if retention_summary.oldest_retained_segment_age_seconds is None:
        logging.info("  oldest retained segment age: none")
    else:
        logging.info(
            "  oldest retained segment age: %.3fs",
            retention_summary.oldest_retained_segment_age_seconds,
        )
    logging.info(
        "  maximum allowed retained age: %.3fs",
        retention_summary.maximum_allowed_age_seconds,
    )
    logging.info(
        "  files match manifest rows: %s",
        retention_summary.files_match_manifest_rows,
    )
    logging.info("  missing/corrupt segments: %d", len(corrupt_or_missing))
    for problem in corrupt_or_missing:
        logging.info("    - %s", problem)
    logging.info(
        "  max |wall_start_utc - fs_created_utc|: %.3fs",
        timing_summary.max_wall_vs_filesystem_delta_seconds,
    )
    if timing_summary.last_computed_wall_end_utc is None:
        logging.info("  last computed wall end: none")
        logging.info("  computed wall end to capture end gap: none")
    else:
        logging.info("  last computed wall end: %s", timing_summary.last_computed_wall_end_utc)
        logging.info(
            "  computed wall end to actual capture end gap: %+0.3fs",
            timing_summary.last_wall_end_to_capture_end_gap_seconds,
        )
    logging.info("  retention cleanup confirmed: %s", retention_summary.retention_confirmed)
    return 1 if corrupt_or_missing or not retention_summary.retention_confirmed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        logging.getLogger(__name__).error("Soak test failed: %s", exc)
        raise SystemExit(2) from exc
