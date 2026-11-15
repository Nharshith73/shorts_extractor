"""CLI entry point for streaming Stage 2 detection."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import load_detection_config, load_segments_dir
from .runner import DetectionRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze complete Stage 1 segments for excitement.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--once", action="store_true", help="Analyze current pending segments then exit")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        detection_config = load_detection_config(args.config)
        runner = DetectionRunner(load_segments_dir(args.config) / "manifest.sqlite3", detection_config)
        if args.once:
            runner.run_once()
        else:
            runner.follow()
    except (OSError, RuntimeError, ValueError) as exc:
        logging.getLogger(__name__).error("Detection failed: %s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
