"""Operator annotation workflow for ground-truth gameplay moment labeling."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from app.ingestion.stream_reader import probe_duration
from .dataset import DEFAULT_MANIFEST_PATH, EvaluationDataset, compute_source_fingerprint, generate_sample_id
from .models import STANDARD_CATEGORIES, AnnotationRecord

LOGGER = logging.getLogger(__name__)


def generate_preview_clip(
    source_video: Path,
    start_seconds: float,
    duration_seconds: float,
    output_dir: Path,
    sample_id: str,
) -> Path:
    """Extract a fast preview clip for operator visual inspection."""
    import subprocess
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_file = output_dir / f"{sample_id}_preview.mp4"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        str(start_seconds),
        "-i",
        str(source_video),
        "-t",
        str(duration_seconds),
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        str(preview_file),
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=60)
    return preview_file


def annotate_interval(
    video: Path,
    start: float,
    end: float,
    *,
    category: str = "unknown",
    rating: int = 3,
    setup: bool = True,
    payoff: bool = True,
    context_complete: bool = True,
    approved: str = "y",
    rejection_reason: str = "",
    notes: str = "",
    synthetic: bool = False,
    sample_id_override: str | None = None,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    preview: bool = False,
    preview_dir: Path = Path("data/evaluation/previews"),
    author: str = "operator",
) -> AnnotationRecord:
    """Create or update an annotation record in the evaluation manifest."""
    video = video.resolve()
    if not video.is_file():
        raise FileNotFoundError(f"Source video not found: {video}")

    duration = probe_duration(video)
    if start < 0 or end <= start or end > duration + 0.25:
        raise ValueError(
            f"Invalid interval [{start:.2f}, {end:.2f}] for video with duration {duration:.2f}s"
        )

    fingerprint = compute_source_fingerprint(video)
    source_id = video.stem
    sample_id = sample_id_override or generate_sample_id(source_id, start, end)

    if preview:
        try:
            preview_path = generate_preview_clip(
                video, start, end - start, preview_dir, sample_id
            )
            LOGGER.info("Generated preview clip at %s", preview_path)
        except Exception as exc:
            LOGGER.warning("Could not generate preview clip: %s", exc)

    dataset = EvaluationDataset(manifest_path)
    record = AnnotationRecord(
        sample_id=sample_id,
        source_fingerprint=fingerprint,
        source_id=source_id,
        start_seconds=round(start, 6),
        end_seconds=round(end, 6),
        event_category=category,
        editorial_rating=rating,
        context_completeness=context_complete,
        setup_presence=setup,
        payoff_presence=payoff,
        approved=approved,
        rejection_reason=rejection_reason,
        notes=notes,
        is_synthetic=synthetic,
    )

    saved_record = dataset.add_or_update(record, author=author, source_duration=duration)
    LOGGER.info(
        "Saved annotation '%s' (v%d): category=%s, rating=%d, approved=%s",
        saved_record.sample_id,
        saved_record.annotation_version,
        saved_record.event_category,
        saved_record.editorial_rating,
        saved_record.approved,
    )
    return saved_record
