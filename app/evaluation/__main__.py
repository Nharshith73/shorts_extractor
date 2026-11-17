"""CLI entrypoint for Stage 4A evaluation, annotation, and context generation."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
import sys

from .annotate import annotate_interval
from .context import FrameSamplerConfig, build_candidate_context
from .dataset import DEFAULT_MANIFEST_PATH, EvaluationDataset
from .evaluate import evaluate_candidates, export_evaluation_artifacts

LOGGER = logging.getLogger("app.evaluation")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.evaluation",
        description="StreamClip AI — Gameplay Intelligence Foundation & Evaluation CLI",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. annotate
    ann_parser = subparsers.add_parser("annotate", help="Annotate a ground-truth gameplay moment")
    ann_parser.add_argument("--video", type=Path, required=True, help="Path to source video file")
    ann_parser.add_argument("--start", type=float, required=True, help="Moment start timestamp in seconds")
    ann_parser.add_argument("--end", type=float, required=True, help="Moment end timestamp in seconds")
    ann_parser.add_argument("--category", default="unknown", help="Event category (e.g. funny, clutch_kill, fail)")
    ann_parser.add_argument("--rating", type=int, default=3, help="Editorial rating (0 to 5)")
    ann_parser.add_argument("--setup", action=argparse.BooleanOptionalAction, default=True, help="Setup is present")
    ann_parser.add_argument("--payoff", action=argparse.BooleanOptionalAction, default=True, help="Payoff is present")
    ann_parser.add_argument("--context-complete", action=argparse.BooleanOptionalAction, default=True, help="Context is complete")
    ann_parser.add_argument("--approved", choices=["y", "n", ""], default="y", help="Approval decision (y/n)")
    ann_parser.add_argument("--rejection-reason", default="", help="Reason for rejection if approved=n")
    ann_parser.add_argument("--notes", default="", help="Reviewer notes")
    ann_parser.add_argument("--synthetic", action="store_true", help="Mark annotation as synthetic")
    ann_parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH, help="Path to dataset manifest.json")
    ann_parser.add_argument("--preview", action="store_true", help="Generate an MP4 preview clip")

    # 2. context
    ctx_parser = subparsers.add_parser("context", help="Extract multimodal context for a candidate clip")
    ctx_parser.add_argument("--video", type=Path, required=True, help="Path to source video file")
    ctx_parser.add_argument("--candidate-json", type=Path, required=True, help="Path to candidate metadata JSON or clips.json")
    ctx_parser.add_argument("--out", type=Path, required=True, help="Output directory for context package")
    ctx_parser.add_argument("--features-csv", type=Path, default=None, help="Optional features.csv path")
    ctx_parser.add_argument("--transcripts-dir", type=Path, default=None, help="Optional .transcripts directory")
    ctx_parser.add_argument("--chat", type=Path, default=None, help="Optional Twitch chat JSON path")
    ctx_parser.add_argument("--max-frames", type=int, default=16, help="Maximum visual frames to sample")

    # 3. evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate candidate predictions against ground-truth manifest")
    eval_parser.add_argument("--candidates", type=Path, required=True, help="Path to clips.json or review.csv")
    eval_parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH, help="Path to dataset manifest.json")
    eval_parser.add_argument("--out", type=Path, required=True, help="Output directory for evaluation reports")
    eval_parser.add_argument("--top-k", type=int, default=10, help="Evaluate on top-K candidates (default: 10)")
    eval_parser.add_argument("--min-iou", type=float, default=0.20, help="Minimum temporal IoU for candidate-label match")

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.command == "annotate":
        record = annotate_interval(
            video=args.video,
            start=args.start,
            end=args.end,
            category=args.category,
            rating=args.rating,
            setup=args.setup,
            payoff=args.payoff,
            context_complete=args.context_complete,
            approved=args.approved,
            rejection_reason=args.rejection_reason,
            notes=args.notes,
            synthetic=args.synthetic,
            manifest_path=args.manifest,
            preview=args.preview,
        )
        print(f"Annotation saved: sample_id={record.sample_id} v{record.annotation_version} ({record.event_category}, rating={record.editorial_rating})")
        return 0

    elif args.command == "context":
        raw = json.loads(args.candidate_json.read_text(encoding="utf-8"))
        candidates = raw if isinstance(raw, list) else [raw]
        sampler = FrameSamplerConfig(max_frames=args.max_frames)
        for c in candidates:
            build_candidate_context(
                candidate=c,
                source_video=args.video,
                output_dir=args.out,
                features_csv=args.features_csv,
                transcripts_dir=args.transcripts_dir,
                chat_path=args.chat,
                sampler_config=sampler,
            )
        print(f"Generated context packages for {len(candidates)} candidate(s) under {args.out}")
        return 0

    elif args.command == "evaluate":
        if not args.candidates.is_file():
            LOGGER.error("Candidates file not found: %s", args.candidates)
            return 1
        if not args.manifest.is_file():
            LOGGER.error("Ground-truth manifest not found: %s", args.manifest)
            return 1

        # Load candidates from JSON or CSV
        if args.candidates.suffix == ".csv":
            with args.candidates.open(encoding="utf-8") as f:
                candidates = list(csv.DictReader(f))
        else:
            candidates = json.loads(args.candidates.read_text(encoding="utf-8"))

        dataset = EvaluationDataset(args.manifest)
        labels = dataset.records
        if not labels:
            LOGGER.error("Manifest at %s contains 0 annotations.", args.manifest)
            return 1

        report = evaluate_candidates(candidates, labels, top_k=args.top_k, min_iou=args.min_iou)
        json_path, csv_path, summary_path = export_evaluation_artifacts(report, args.out)

        print("\n" + summary_path.read_text(encoding="utf-8"))
        print(f"Artifacts saved:\n - JSON: {json_path}\n - CSV:  {csv_path}\n - MD:   {summary_path}\n")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
