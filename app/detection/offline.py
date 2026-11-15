"""Offline Stage 2 tuning runner for a single local video file."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from app.ingestion.stream_reader import probe_duration

from .config import DetectionConfig, load_detection_config
from .features import FEATURE_NAMES, extract_segment_features
from .models import DetectedEvent, FeatureBin
from .scoring import detect_events

LOGGER = logging.getLogger(__name__)


def _event_json(event: DetectedEvent, origin: datetime) -> dict[str, object]:
    return {
        "peak_time_seconds": (event.peak_wall_utc - origin).total_seconds(),
        "start_time_seconds": (event.start_wall_utc - origin).total_seconds(),
        "end_time_seconds": (event.end_wall_utc - origin).total_seconds(),
        "score": event.score,
        "components": event.components,
    }


def run_offline(
    video_path: Path,
    config: DetectionConfig,
    *,
    ffmpeg_binary: str = "ffmpeg",
    ffprobe_binary: str = "ffprobe",
) -> tuple[list[FeatureBin], list[DetectedEvent], float, float]:
    """Extract and score one file, returning features, events, duration, and RTF."""
    duration = probe_duration(video_path, ffprobe_binary)
    origin = datetime(1970, 1, 1, tzinfo=UTC)
    source_id = hashlib.sha256(str(video_path.resolve()).encode("utf-8")).hexdigest()[:16]
    started = time.monotonic()
    features = extract_segment_features(
        video_path,
        segment_id=None,
        source_id=source_id,
        session_id="offline",
        wall_start_utc=origin,
        duration_seconds=duration,
        config=config,
        ffmpeg_binary=ffmpeg_binary,
    )
    _, events = detect_events(features, config)
    processing_seconds = time.monotonic() - started
    realtime_factor = duration / processing_seconds if processing_seconds else float("inf")
    return features, events, duration, realtime_factor


def write_features_csv(path: Path, features: list[FeatureBin], origin: datetime) -> None:
    """Write raw features for spreadsheet or plotting-based threshold tuning."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["time_seconds", "bin_seconds", *FEATURE_NAMES]
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for feature in features:
            writer.writerow(
                {
                    "time_seconds": (feature.wall_utc - origin).total_seconds(),
                    "bin_seconds": feature.bin_seconds,
                    **{name: getattr(feature, name) for name in FEATURE_NAMES},
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lightweight detection on one video file.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Output event timeline JSON")
    parser.add_argument("--features", type=Path, required=True, help="Output raw feature CSV")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        if not args.video.is_file():
            raise FileNotFoundError(args.video)
        config = load_detection_config(args.config)
        features, events, duration, realtime_factor = run_offline(args.video, config)
        origin = datetime(1970, 1, 1, tzinfo=UTC)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "duration_seconds": duration,
                    "real_time_factor": realtime_factor,
                    "events": [_event_json(event, origin) for event in events],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        write_features_csv(args.features, features, origin)
        LOGGER.info(
            "Processed %.1fs in %.2fx real time (%d events)",
            duration,
            realtime_factor,
            len(events),
        )
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        LOGGER.error("Offline detection failed: %s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
