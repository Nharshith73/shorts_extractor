"""VOD run, review, and evaluate commands."""

import argparse
import logging
from pathlib import Path
import subprocess

from .pipeline import run, rescore
from .review import review, summary
from .evaluate import evaluate
from .settings import parse_timestamp


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline StreamClip AI VOD pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--video", type=Path, required=True)
    run_parser.add_argument("--chat", type=Path)
    run_parser.add_argument("--out", type=Path, required=True)
    run_parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    run_parser.add_argument("--start", type=parse_timestamp, default=0.0)
    run_parser.add_argument("--end", type=parse_timestamp)
    run_parser.add_argument("--streamer", help="Source streamer handle for final credit attribution")
    rescore_parser = commands.add_parser("rescore")
    rescore_parser.add_argument("--out", type=Path, required=True)
    rescore_parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    rescore_parser.add_argument("--extract", action="store_true")
    for command in (run_parser, rescore_parser):
        ranking = command.add_mutually_exclusive_group()
        ranking.add_argument("--top-k", type=int)
        ranking.add_argument("--clips-per-hour", type=float)
    summary_parser = commands.add_parser("summary")
    summary_parser.add_argument("--out", type=Path, required=True)
    review_parser = commands.add_parser("review")
    review_parser.add_argument("--out", type=Path, required=True)
    render_parser = commands.add_parser("render", help="Render reviewed clips as captioned vertical Shorts")
    render_parser.add_argument("--out", type=Path, required=True)
    render_parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    render_parser.add_argument("--streamer", help="Override inferred/configured credit handle")
    from .render import parse_ranks
    render_selection = render_parser.add_mutually_exclusive_group()
    render_selection.add_argument("--all", action="store_true", help="Render all reviewed preview rows")
    render_selection.add_argument("--ranks", type=parse_ranks, help="Render these ranks regardless of approval, e.g. 1,3,5")
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--out", type=Path, required=True)
    evaluation.add_argument("--labels", type=Path, required=True)
    evaluation.add_argument("--tolerance", type=float, default=10)
    evaluation.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        if args.command == "run":
            run(args.video, args.out, args.config, args.chat, args.start, args.end,
                top_k=args.top_k, clips_per_hour=args.clips_per_hour, streamer=args.streamer)
        elif args.command == "rescore":
            rescore(args.out, args.config, args.extract, top_k=args.top_k, clips_per_hour=args.clips_per_hour)
        elif args.command == "summary":
            summary(args.out)
        elif args.command == "review":
            review(args.out)
        elif args.command == "render":
            from .render import render_reviewed
            report = render_reviewed(args.out, args.config, all_clips=args.all, ranks=args.ranks, streamer=args.streamer)
            return 2 if report["failed"] else 0
        else:
            evaluate(args.out, args.labels, args.tolerance, args.top_k)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        logging.getLogger(__name__).error("VOD %s failed: %s", args.command, exc)
        return 2
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
