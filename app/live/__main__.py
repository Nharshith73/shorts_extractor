"""Operator CLI for Single-Stream Live Operations."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

from app.evaluation.inference import MockHighlightJudgeProvider
from app.evaluation.providers.qwen import QwenProviderConfig, QwenVisionLanguageProvider
from .config import AuthorizationError, LiveSourceConfig, MissingMediaInputError, load_source_config
from .pipeline import SingleStreamLivePipeline
from .queue import LiveJudgmentQueue
from .review import LiveReviewStore
from .supervisor import LiveSupervisor, SupervisorStateStore
from .youtube import inspect_youtube_source

LOGGER = logging.getLogger("app.live")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.live",
        description="StreamClip AI — Single-Stream Live Intelligence Operator CLI",
    )
    parser.add_argument("--config", type=Path, default=Path("source.yaml"), help="Path to source.yaml (default: source.yaml)")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. start
    start_parser = subparsers.add_parser("start", help="Start continuous live capture and AI pipeline")
    start_parser.add_argument("--media-override", default=None, help="Direct media feed URL override (e.g. rtmp://... or test video)")
    start_parser.add_argument("--mock-ai", action="store_true", help="Use MockHighlightJudgeProvider instead of real local Qwen VLM")
    start_parser.add_argument("--cpu-ai", action="store_true", help="Run Qwen inference on CPU (n_gpu_layers=0)")
    start_parser.add_argument("--foreground", action="store_true", help="Run synchronously in foreground (Ctrl+C to stop)")

    # 2. status
    subparsers.add_parser("status", help="Inspect live supervisor, capture, and queue status")

    # 3. stop
    subparsers.add_parser("stop", help="Gracefully stop live stream capture and pipeline")

    # 4. candidates
    cand_parser = subparsers.add_parser("candidates", help="List extracted candidates and dual rankings")
    cand_parser.add_argument("--limit", type=int, default=25, help="Maximum candidates to display (default: 25)")

    # 5. queue
    q_parser = subparsers.add_parser("queue", help="Inspect persistent AI judgment queue")
    q_parser.add_argument("--limit", type=int, default=25, help="Maximum queue jobs to display")
    q_parser.add_argument("--status-filter", choices=["PENDING", "RUNNING", "COMPLETED", "RETRY_WAIT", "FAILED"], default=None)

    # 6. approve
    app_parser = subparsers.add_parser("approve", help="Approve an extracted candidate for publishing")
    app_parser.add_argument("candidate_id", help="Candidate ID to approve")
    app_parser.add_argument("--notes", default="", help="Optional editorial notes")

    # 7. reject
    rej_parser = subparsers.add_parser("reject", help="Reject an extracted candidate")
    rej_parser.add_argument("candidate_id", help="Candidate ID to reject")
    rej_parser.add_argument("--reason", default="Rejected by operator", help="Rejection reason")

    # 8. validate-source
    subparsers.add_parser("validate-source", help="Validate source authorization and media feed availability")

    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Load source.yaml
    try:
        config = load_source_config(args.config)
    except Exception as exc:
        LOGGER.error("Failed to load source configuration from %s: %s", args.config, exc)
        return 1

    manifest_db = config.capture.segments_dir / "manifest.sqlite3"

    if args.command == "validate-source":
        print("\n=== StreamClip AI Source Authorization & Media Feed Audit ===")
        print(f"Source ID:      {config.id}")
        print(f"Creator:        {config.creator_name}")
        print(f"Platform:       {config.platform}")
        print(f"Watch URL:      {config.watch_url or '(none)'}")
        print(f"Permission Ref: {config.authorization.permission_reference}")

        # Validate authorization gates
        try:
            config.authorization.validate()
            print("Authorization:  VALID (All gates permitted)")
        except AuthorizationError as exc:
            print(f"Authorization:  BLOCKED ({exc})")
            return 1

        # Inspect YouTube metadata if watch_url provided
        if config.watch_url:
            yt_meta = inspect_youtube_source(config.watch_url)
            print(f"\nYouTube API Status: {'Available' if yt_meta.api_available else 'Offline/Unauthenticated'}")
            print(f"Broadcast State:    {yt_meta.live_broadcast_content}")
            print(f"Is Live:            {yt_meta.is_live}")
            print(f"Chat Enabled:       {yt_meta.chat_enabled} (Active Chat ID: {yt_meta.active_live_chat_id or 'none'})")
            if yt_meta.error_message:
                print(f"Note:               {yt_meta.error_message}")

        # Resolve media input feed
        print("\nMedia Acquisition Check:")
        try:
            media_url = config.acquisition.resolve_media_url(config.watch_url)
            print(f"Media Input:    RESOLVED -> {media_url[:60]}...")
            return 0
        except MissingMediaInputError as exc:
            print(f"Media Input:    BLOCKED -> {exc}")
            return 2

    elif args.command == "status":
        state_store = SupervisorStateStore(manifest_db)
        state_record = state_store.get_state(config.id)
        queue = LiveJudgmentQueue(manifest_db)
        q_stats = queue.get_stats()
        review_store = LiveReviewStore(manifest_db)
        candidates = review_store.list_candidates()

        print("\n=== StreamClip AI Live Status ===")
        print(f"Source ID:          {config.id} ({config.creator_name})")
        print(f"Supervisor State:   {state_record.get('current_state') if state_record else 'STOPPED'}")
        print(f"Previous State:     {state_record.get('previous_state') if state_record else 'none'}")
        print(f"Active Media URL:   {state_record.get('media_url') if state_record else '(none)'}")
        print(f"Reconnect Attempts: {state_record.get('reconnect_attempts') if state_record else 0}")
        print(f"Last State Update:  {state_record.get('updated_at') if state_record else 'never'}")
        if state_record and state_record.get("error_message"):
            print(f"Last Error:         {state_record['error_message']}")

        print(f"\nAI Judgment Queue (Total depth: {queue.get_queue_depth()}):")
        for st, count in q_stats.items():
            print(f"  - {st:<12}: {count}")

        print(f"\nExtracted Candidates: {len(candidates)}")
        approved = sum(1 for c in candidates if c.human_approval == "approved")
        rejected = sum(1 for c in candidates if c.human_approval == "rejected")
        pending = sum(1 for c in candidates if c.human_approval == "pending")
        print(f"  - Approved: {approved} | Rejected: {rejected} | Pending Review: {pending}\n")
        return 0

    elif args.command == "start":
        provider = (
            MockHighlightJudgeProvider()
            if args.mock_ai
            else QwenVisionLanguageProvider(QwenProviderConfig(n_gpu_layers=0 if args.cpu_ai else 99))
        )
        pipeline = SingleStreamLivePipeline(config=config, provider=provider)

        try:
            print(f"Starting StreamClip AI live pipeline for source '{config.id}'...")
            pipeline.start(media_override=args.media_override)
            print(f"Supervisor state: {pipeline.supervisor.state.value}")

            if args.foreground:
                print("Running in foreground. Press Ctrl+C to stop.\n")
                try:
                    while True:
                        import time
                        time.sleep(2.0)
                        summary = pipeline.supervisor.get_summary()
                        print(
                            f"\r[Live Monitor] State: {summary['state']:<12} | Segments: {summary['segments_completed']:<5} | Reconnects: {summary['reconnect_count']}",
                            end="",
                            flush=True,
                        )
                except KeyboardInterrupt:
                    print("\nReceived stop signal. Shutting down gracefully...")
                finally:
                    pipeline.stop()
            return 0
        except Exception as exc:
            LOGGER.exception("Live pipeline startup failed: %s", exc)
            return 1

    elif args.command == "stop":
        supervisor = LiveSupervisor(config=config)
        supervisor.stop()
        print(f"Stopped live supervisor for source '{config.id}'.")
        return 0

    elif args.command == "candidates":
        review_store = LiveReviewStore(manifest_db)
        items = review_store.list_candidates(limit=args.limit)
        if not items:
            print("\nNo extracted candidates found in live review store.\n")
            return 0

        print(f"\n{'Sig#':<5} {'AI#':<5} {'Candidate ID':<26} {'Score':<7} {'Rating':<7} {'Category':<14} {'AI App':<7} {'Human Status'}")
        print("-" * 85)
        for it in items:
            sig_r = f"#{it.signal_rank}"
            ai_r = f"#{it.ai_rank}" if it.ai_rank else "-"
            cid = it.candidate_id[:24]
            sc = f"{it.signal_score:.2f}"
            rt = f"{it.ai_editorial_rating}/5" if it.ai_editorial_rating is not None else "-"
            cat = it.ai_category or "-"
            ai_app = it.ai_approved or "-"
            h_stat = it.human_approval.upper()
            print(f"{sig_r:<5} {ai_r:<5} {cid:<26} {sc:<7} {rt:<7} {cat:<14} {ai_app:<7} {h_stat}")
        print()
        return 0

    elif args.command == "queue":
        queue = LiveJudgmentQueue(manifest_db)
        jobs = queue.list_jobs(limit=args.limit, status=args.status_filter)
        if not jobs:
            print("\nAI Judgment queue is empty.\n")
            return 0

        print(f"\n{'Job ID':<16} {'Candidate ID':<26} {'Priority':<9} {'Status':<12} {'Attempts':<9} {'Updated'}")
        print("-" * 85)
        for j in jobs:
            print(f"{j.job_id:<16} {j.candidate_id[:24]:<26} {j.priority:<9.2f} {j.status:<12} {j.attempts}/{j.max_attempts:<7} {j.updated_at[:19]}")
        print()
        return 0

    elif args.command == "approve":
        review_store = LiveReviewStore(manifest_db)
        success = review_store.set_approval(args.candidate_id, "approved", notes=args.notes)
        if success:
            print(f"Candidate '{args.candidate_id}' APPROVED successfully.")
            return 0
        else:
            print(f"Candidate '{args.candidate_id}' not found in review store.")
            return 1

    elif args.command == "reject":
        review_store = LiveReviewStore(manifest_db)
        success = review_store.set_approval(args.candidate_id, "rejected", rejection_reason=args.reason)
        if success:
            print(f"Candidate '{args.candidate_id}' REJECTED ({args.reason}).")
            return 0
        else:
            print(f"Candidate '{args.candidate_id}' not found in review store.")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
