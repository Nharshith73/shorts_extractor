"""Command-line candidate worker."""

import argparse
import logging
from pathlib import Path
import time

from .config import load_config
from .candidate_extractor import CandidateExtractor


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract candidates from Stage 2 events")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        config, segments, retention = load_config(args.config)
        worker = CandidateExtractor(segments, config, retention)
        while True:
            worker.run_once()
            if args.once:
                return 0
            time.sleep(config.poll_seconds)
    except KeyboardInterrupt:
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        logging.getLogger(__name__).error("Candidate worker failed: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
