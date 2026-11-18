"""Candidate evaluation and editorial judging workflow for Stage 4B."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
import time
from typing import Any, Sequence

from .context import FrameSamplerConfig, build_candidate_context
from .inference import HighlightJudgeProvider
from .models import CandidateContext
from .schema import AIJudgmentRecord, EvidenceItem

LOGGER = logging.getLogger(__name__)


def load_candidate_context(
    candidate_id: str,
    vod_dir: Path,
    candidate_dict: dict[str, Any] | None = None,
    source_video: Path | None = None,
    max_frames: int = 8,
) -> CandidateContext:
    """Load existing CandidateContext package or generate one on demand."""
    ctx_file = vod_dir / "context" / candidate_id / "context.json"
    if ctx_file.is_file():
        try:
            data = json.loads(ctx_file.read_text(encoding="utf-8"))
            return CandidateContext.from_dict(data)
        except Exception as exc:
            LOGGER.warning("Could not read existing context package %s: %s", ctx_file, exc)

    # Generate context if source video and candidate metadata are available
    if candidate_dict is not None and source_video is not None and source_video.is_file():
        features_csv = vod_dir / "features.csv"
        transcripts_dir = vod_dir / ".transcripts"
        chat_json = vod_dir / "chat.json"

        sampler = FrameSamplerConfig(max_frames=max_frames)
        ctx = build_candidate_context(
            candidate=candidate_dict,
            source_video=source_video,
            output_dir=vod_dir / "context",
            features_csv=features_csv if features_csv.is_file() else None,
            transcripts_dir=transcripts_dir if transcripts_dir.is_dir() else None,
            chat_path=chat_json if chat_json.is_file() else None,
            sampler_config=sampler,
        )
        return ctx

    raise FileNotFoundError(
        f"Context package not found at {ctx_file} and insufficient metadata to generate it"
    )


def judge_candidates(
    vod_dir: Path,
    provider: HighlightJudgeProvider,
    candidate_id: str | None = None,
    top_k: int | None = 10,
    force: bool = False,
    max_frames: int = 8,
    timeout_seconds: float = 60.0,
) -> tuple[list[AIJudgmentRecord], Path]:
    """Evaluate candidates using HighlightJudgeProvider and export dual-ranked comparison."""
    vod_dir = vod_dir.resolve()
    clips_json = vod_dir / "clips.json"
    if not clips_json.is_file():
        raise FileNotFoundError(f"clips.json not found in {vod_dir}")

    candidates_raw: list[dict[str, Any]] = json.loads(clips_json.read_text(encoding="utf-8"))
    if not candidates_raw:
        LOGGER.warning("No candidates found in %s", clips_json)
        return [], vod_dir / "ai_judgments.csv"

    # Sort candidates by canonical signal score (Signal Rank)
    from app.vod.selection import canonical_ranking_key

    sorted_candidates = sorted(
        candidates_raw,
        key=lambda c: canonical_ranking_key(
            c.get("score", 0.0),
            c.get("unclipped_score"),
            c.get("peak_seconds", c.get("start_seconds", 0.0)),
        ),
    )

    # Filter target candidates
    if candidate_id:
        targets = [c for c in sorted_candidates if c.get("window_id") == candidate_id]
        if not targets:
            raise ValueError(f"Candidate ID '{candidate_id}' not found in {clips_json}")
    elif top_k is not None and top_k > 0:
        targets = sorted_candidates[:top_k]
    else:
        targets = sorted_candidates

    judgments_dir = vod_dir / "judgments"
    judgments_dir.mkdir(parents=True, exist_ok=True)

    records: list[AIJudgmentRecord] = []
    source_video_path = Path(targets[0].get("video", "")) if targets else None

    for cand in targets:
        w_id = str(cand.get("window_id"))
        judgment_file = judgments_dir / f"{w_id}.json"

        # Check existing judgment
        if judgment_file.is_file() and not force:
            try:
                cached_data = json.loads(judgment_file.read_text(encoding="utf-8"))
                # Check model name matches to reuse
                cached_model = cached_data.get("model_metadata", {}).get("model")
                if cached_model == provider.model_name:
                    record = AIJudgmentRecord.model_validate(cached_data)
                    LOGGER.info("Reusing existing judgment for %s (v%s, rating=%d)", w_id, record.schema_version, record.editorial_rating)
                    records.append(record)
                    continue
            except Exception as exc:
                LOGGER.warning("Could not read cached judgment %s: %s", judgment_file, exc)

        # Load or generate context
        ctx = load_candidate_context(
            candidate_id=w_id,
            vod_dir=vod_dir,
            candidate_dict=cand,
            source_video=source_video_path,
            max_frames=max_frames,
        )

        LOGGER.info("Evaluating candidate %s with provider %s...", w_id, provider.model_name)
        if records and getattr(provider, "config", None):
            inter_delay = getattr(provider.config, "inter_request_delay_seconds", 0.0)
            if isinstance(inter_delay, (int, float)) and inter_delay > 0:
                time.sleep(inter_delay)

        # Call provider
        if hasattr(provider, "judge_candidate_record"):
            record = provider.judge_candidate_record(ctx, timeout_seconds=timeout_seconds)
        else:
            hj = provider.evaluate_candidate(ctx, timeout_seconds=timeout_seconds)
            record = AIJudgmentRecord(
                schema_version="1.0",
                candidate_id=hj.window_id,
                event_category=hj.event_category,
                summary=hj.suggested_title or hj.reasoning[:100],
                editorial_rating=hj.editorial_rating,
                confidence=hj.confidence,
                context_complete=True,
                setup_present=True,
                payoff_present=True,
                entertainment_factors=["high_motion_action"] if hj.editorial_rating >= 3 else ["routine_transit"],
                evidence=[
                    EvidenceItem(
                        timestamp=ctx.peak_seconds,
                        source="visual" if ctx.components.get("motion", 0) > 1.0 else "audio",
                        observation=hj.reasoning,
                    )
                ],
                hook_suggestion=hj.hook_text,
                recommended_start_seconds=ctx.start_seconds,
                recommended_end_seconds=ctx.end_seconds,
                recommended_layout=hj.recommended_layout,
                rejection_reason=None if hj.editorial_rating >= 3 else "Low editorial score",
                model_metadata=hj.model_metadata,
            )

        # Save judgment JSON atomically
        temp_file = judgment_file.with_suffix(".tmp.json")
        temp_file.write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")
        temp_file.replace(judgment_file)
        records.append(record)
        LOGGER.info(
            "Saved AI judgment for %s: category=%s, rating=%d/5, approved=%s (latency: %.2fs)",
            w_id,
            record.event_category,
            record.editorial_rating,
            "y" if record.is_approved else "n",
            record.model_metadata.get("latency_seconds", 0.0),
        )

    # Dual-ranking comparison table
    # 1. Map candidates by window_id
    cand_by_id = {str(c.get("window_id")): (idx + 1, c) for idx, c in enumerate(sorted_candidates)}
    
    # 2. Sort by AI editorial rank (rating desc, confidence desc)
    ai_ranked_records = sorted(
        records,
        key=lambda r: (r.editorial_rating, r.confidence),
        reverse=True,
    )

    csv_path = vod_dir / "ai_judgments.csv"
    fieldnames = [
        "signal_rank",
        "candidate_id",
        "start_seconds",
        "end_seconds",
        "duration_seconds",
        "signal_score",
        "ai_rank",
        "ai_editorial_rating",
        "ai_confidence",
        "ai_category",
        "ai_approved",
        "ai_rejection_reason",
        "ai_hook_suggestion",
        "ai_recommended_layout",
        "ai_summary",
        "latency_seconds",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for ai_rank, rec in enumerate(ai_ranked_records, 1):
            sig_info = cand_by_id.get(rec.candidate_id)
            sig_rank = sig_info[0] if sig_info else ""
            c_dict = sig_info[1] if sig_info else {}
            writer.writerow({
                "signal_rank": sig_rank,
                "candidate_id": rec.candidate_id,
                "start_seconds": c_dict.get("start_seconds", rec.recommended_start_seconds),
                "end_seconds": c_dict.get("end_seconds", rec.recommended_end_seconds),
                "duration_seconds": rec.recommended_duration,
                "signal_score": round(float(c_dict.get("score", 0.0)), 3),
                "ai_rank": ai_rank,
                "ai_editorial_rating": rec.editorial_rating,
                "ai_confidence": round(rec.confidence, 3),
                "ai_category": rec.event_category,
                "ai_approved": "y" if rec.is_approved else "n",
                "ai_rejection_reason": rec.rejection_reason or "",
                "ai_hook_suggestion": rec.hook_suggestion,
                "ai_recommended_layout": rec.recommended_layout,
                "ai_summary": rec.summary,
                "latency_seconds": rec.model_metadata.get("latency_seconds", 0.0),
            })

    LOGGER.info("Exported dual-ranked AI judgment comparison to %s", csv_path)
    return records, csv_path
