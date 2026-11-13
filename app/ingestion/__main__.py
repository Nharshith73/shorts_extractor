"""CLI entry point for Stage 1 ingestion."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .stream_reader import load_config, StreamReader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a source into a rolling MPEG-TS buffer.")
    parser.add_argument("--source", help="Permitted live-stream URL or local video file")
    parser.add_argument(
        "--config", type=Path, default=Path("config.yaml"), help="YAML configuration path"
    )
    parser.add_argument("--log-level", default="INFO", help="Python logging level (default: INFO)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        config = load_config(args.config, source_override=args.source)
        reader = StreamReader(config)
        reader.run_forever()
    except (OSError, RuntimeError, ValueError) as exc:
        logging.getLogger(__name__).error("Ingestion failed: %s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
