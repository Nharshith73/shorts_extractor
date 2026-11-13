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
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.ingestion.stream_reader import StreamConfig, StreamReader, probe_duration  # noqa: E402


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

    work_dir = args.work_dir.resolve()
    source = args.source.resolve() if args.source else work_dir / "synthetic-source.mp4"
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

    records = [
        record
        for record in reader.buffer.manifest.all_records(reader.source_id)
        if record.status in {"complete", "pinned"}
    ]
    corrupt_or_missing: list[str] = []
    for record in records:
        if record.path is None or not record.path.exists():
            corrupt_or_missing.append(str(record.path))
            continue
        try:
            probe_duration(record.path)
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            corrupt_or_missing.append(f"{record.path}: {exc}")

    drift_values = []
    for previous, current in zip(records, records[1:]):
        drift_values.append(abs((current.wall_start_utc - previous.wall_end_utc).total_seconds()))
    max_drift = max(drift_values, default=0.0)
    cleanup_cutoff = finished_utc.timestamp() - args.retention_seconds
    retention_ok = (
        bool(records)
        and all(record.wall_end_utc.timestamp() >= cleanup_cutoff - args.segment_seconds for record in records)
        and (args.duration_seconds <= args.retention_seconds or len(records) * args.segment_seconds < args.duration_seconds)
    )

    logging.info("Stage 1 soak-test report")
    logging.info("  capture wall time: %.1fs", (finished_utc - started_utc).total_seconds())
    logging.info("  retained segment count: %d", len(records))
    logging.info("  missing/corrupt segments: %d", len(corrupt_or_missing))
    for problem in corrupt_or_missing:
        logging.info("    - %s", problem)
    logging.info("  max timestamp drift vs wall clock: %.3fs", max_drift)
    logging.info("  retention cleanup confirmed: %s", retention_ok)
    return 1 if corrupt_or_missing or not retention_ok else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        logging.getLogger(__name__).error("Soak test failed: %s", exc)
        raise SystemExit(2) from exc
