"""Evaluation framework comparing candidate highlights against ground-truth annotations."""

from __future__ import annotations

import csv
import json
import logging
import math
from pathlib import Path
from typing import Any, Sequence

from app.vod.selection import canonical_ranking_key
from .dataset import EvaluationDataset
from .models import (
    AnnotationRecord,
    CandidateMatch,
    EvaluationMetricsReport,
    utc_iso_now,
)

LOGGER = logging.getLogger(__name__)


def compute_temporal_iou(
    start_a: float, end_a: float, start_b: float, end_b: float
) -> float:
    """Compute temporal Intersection over Union (IoU) between two intervals."""
    intersection = max(0.0, min(end_a, end_b) - max(start_a, start_b))
    union = max(end_a, end_b) - min(start_a, start_b)
    return (intersection / union) if union > 0 else 0.0


def evaluate_candidates(
    candidates: Sequence[dict[str, Any]],
    labels: Sequence[AnnotationRecord],
    *,
    top_k: int = 10,
    min_iou: float = 0.20,
) -> EvaluationMetricsReport:
    """Evaluate ranked candidate highlights against ground-truth labels with 1-to-1 matching."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    # Sort candidates canonically by (score desc, unclipped_score desc, peak_seconds asc)
    ranked = sorted(
        candidates,
        key=lambda c: canonical_ranking_key(
            c.get("score", 0.0),
            c.get("unclipped_score"),
            c.get("peak_seconds", c.get("start_seconds", 0.0)),
        ),
    )[:top_k]

    claimed_labels: set[str] = set()
    matches: list[CandidateMatch] = []
    rejection_reasons: dict[str, int] = {}
    ious: list[float] = []

    # Map labels by sample_id
    label_map = {r.sample_id: r for r in labels}
    approved_labels = [r for r in labels if r.approved == "y"]
    total_approved_weight = sum(max(1, r.editorial_rating) for r in approved_labels)

    # 1. Deterministic rank-ordered matching
    for rank, cand in enumerate(ranked, 1):
        c_start = float(cand["start_seconds"])
        c_end = float(cand["end_seconds"])
        w_id = str(cand.get("window_id", f"candidate_{rank}"))
        score = float(cand.get("score", 0.0))

        # Find best overlapping label
        best_label: AnnotationRecord | None = None
        best_iou = 0.0

        for label in labels:
            iou = compute_temporal_iou(c_start, c_end, label.start_seconds, label.end_seconds)
            if iou >= min_iou and iou > best_iou:
                best_iou = iou
                best_label = label

        if best_label is not None:
            ious.append(best_iou)
            if best_label.sample_id in claimed_labels:
                # Duplicate prediction claiming already matched label
                matches.append(
                    CandidateMatch(
                        candidate_rank=rank,
                        candidate_window_id=w_id,
                        candidate_start=c_start,
                        candidate_end=c_end,
                        candidate_score=score,
                        matched_sample_id=best_label.sample_id,
                        label_start=best_label.start_seconds,
                        label_end=best_label.end_seconds,
                        label_category=best_label.event_category,
                        label_rating=best_label.editorial_rating,
                        label_approved=best_label.approved,
                        iou=round(best_iou, 4),
                        match_status="duplicate_match",
                    )
                )
            else:
                claimed_labels.add(best_label.sample_id)
                status = "true_positive" if best_label.approved == "y" else "false_positive"
                if best_label.approved != "y" and best_label.rejection_reason:
                    reason = best_label.rejection_reason.strip()
                    rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

                matches.append(
                    CandidateMatch(
                        candidate_rank=rank,
                        candidate_window_id=w_id,
                        candidate_start=c_start,
                        candidate_end=c_end,
                        candidate_score=score,
                        matched_sample_id=best_label.sample_id,
                        label_start=best_label.start_seconds,
                        label_end=best_label.end_seconds,
                        label_category=best_label.event_category,
                        label_rating=best_label.editorial_rating,
                        label_approved=best_label.approved,
                        iou=round(best_iou, 4),
                        match_status=status,
                    )
                )
        else:
            # No overlap with any ground-truth label
            matches.append(
                CandidateMatch(
                    candidate_rank=rank,
                    candidate_window_id=w_id,
                    candidate_start=c_start,
                    candidate_end=c_end,
                    candidate_score=score,
                    matched_sample_id=None,
                    label_start=None,
                    label_end=None,
                    label_category=None,
                    label_rating=None,
                    label_approved=None,
                    iou=0.0,
                    match_status="false_positive",
                )
            )

    # 2. Metric computations
    pred_count = max(1, len(ranked))
    true_positives = [m for m in matches if m.match_status == "true_positive"]
    tp_count = len(true_positives)

    precision = tp_count / pred_count
    recall = (tp_count / len(approved_labels)) if approved_labels else 0.0

    matched_approved_weight = sum(
        max(1, m.label_rating or 1) for m in true_positives
    )
    weighted_recall = (
        (matched_approved_weight / total_approved_weight)
        if total_approved_weight > 0
        else 0.0
    )

    duplicate_count = sum(1 for m in matches if m.match_status == "duplicate_match")
    duplicate_rate = duplicate_count / pred_count
    approval_rate = tp_count / pred_count

    mean_iou = (sum(ious) / len(ious)) if ious else 0.0
    sorted_ious = sorted(ious)
    median_iou = (
        sorted_ious[len(sorted_ious) // 2]
        if sorted_ious
        else 0.0
    )

    # Category breakdowns
    category_metrics: dict[str, dict[str, float]] = {}
    all_categories = {l.event_category for l in labels} | {
        m.label_category for m in matches if m.label_category
    }

    for cat in sorted(all_categories):
        cat_labels = [l for l in approved_labels if l.event_category == cat]
        cat_preds = [m for m in matches if m.label_category == cat]
        cat_tp = [m for m in true_positives if m.label_category == cat]

        cat_prec = (len(cat_tp) / len(cat_preds)) if cat_preds else 0.0
        cat_rec = (len(cat_tp) / len(cat_labels)) if cat_labels else 0.0
        category_metrics[cat] = {
            "ground_truth_count": len(cat_labels),
            "matched_count": len(cat_tp),
            "precision": round(cat_prec, 4),
            "recall": round(cat_rec, 4),
        }

    unmatched_labels = [
        l.sample_id for l in approved_labels if l.sample_id not in claimed_labels
    ]
    is_synthetic = any(l.is_synthetic for l in labels)

    return EvaluationMetricsReport(
        total_candidates=len(candidates),
        total_ground_truth_labels=len(labels),
        approved_ground_truth_labels=len(approved_labels),
        top_k=min(top_k, len(candidates)),
        precision_at_k=round(precision, 4),
        recall_at_k=round(recall, 4),
        weighted_recall_at_k=round(weighted_recall, 4),
        approval_rate_at_k=round(approval_rate, 4),
        mean_iou=round(mean_iou, 4),
        median_iou=round(median_iou, 4),
        duplicate_rate=round(duplicate_rate, 4),
        category_metrics=category_metrics,
        rejection_reasons=rejection_reasons,
        matches=matches,
        unmatched_labels=unmatched_labels,
        is_synthetic=is_synthetic,
    )


def export_evaluation_artifacts(
    report: EvaluationMetricsReport, output_dir: Path
) -> tuple[Path, Path, Path]:
    """Persist JSON report, CSV comparison, and Markdown summary."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. JSON report
    json_path = output_dir / "evaluation_report.json"
    json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    # 2. CSV comparison
    csv_path = output_dir / "evaluation_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = [
            "candidate_rank",
            "candidate_window_id",
            "candidate_start",
            "candidate_end",
            "candidate_score",
            "matched_sample_id",
            "label_start",
            "label_end",
            "label_category",
            "label_rating",
            "label_approved",
            "iou",
            "match_status",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for m in report.matches:
            writer.writerow({
                "candidate_rank": m.candidate_rank,
                "candidate_window_id": m.candidate_window_id,
                "candidate_start": m.candidate_start,
                "candidate_end": m.candidate_end,
                "candidate_score": m.candidate_score,
                "matched_sample_id": m.matched_sample_id or "",
                "label_start": "" if m.label_start is None else m.label_start,
                "label_end": "" if m.label_end is None else m.label_end,
                "label_category": m.label_category or "",
                "label_rating": "" if m.label_rating is None else m.label_rating,
                "label_approved": m.label_approved or "",
                "iou": m.iou,
                "match_status": m.match_status,
            })

    # 3. Markdown summary
    summary_path = output_dir / "evaluation_summary.md"
    summary_text = generate_summary_markdown(report)
    summary_path.write_text(summary_text, encoding="utf-8")

    return json_path, csv_path, summary_path


def generate_summary_markdown(report: EvaluationMetricsReport) -> str:
    """Format human-readable evaluation summary report."""
    synth_badge = " *(SYNTHETIC GROUND TRUTH)*" if report.is_synthetic else " *(AUTHENTIC GROUND TRUTH)*"
    lines = [
        f"# StreamClip AI — Highlight Evaluation Report{synth_badge}",
        f"",
        f"- **Evaluated At**: {report.evaluated_at}",
        f"- **Top-K Window**: K = {report.top_k} (out of {report.total_candidates} candidates)",
        f"- **Ground Truth Pool**: {report.total_ground_truth_labels} total ({report.approved_ground_truth_labels} approved)",
        f"",
        f"## Primary Metrics",
        f"| Metric | Value | Meaning |",
        f"| :--- | :---: | :--- |",
        f"| **Precision@K** | `{report.precision_at_k:.2%}` | Fraction of top-K predictions that match approved moments |",
        f"| **Recall@K** | `{report.recall_at_k:.2%}` | Fraction of all approved ground truth moments captured |",
        f"| **Weighted Recall@K** | `{report.weighted_recall_at_k:.2%}` | Recall weighted by editorial rating (0-5 stars) |",
        f"| **Approval Rate@K** | `{report.approval_rate_at_k:.2%}` | Editorial acceptance rate among top candidates |",
        f"| **Duplicate Rate** | `{report.duplicate_rate:.2%}` | Rate of redundant duplicate predictions in top-K |",
        f"| **Mean IoU** | `{report.mean_iou:.3f}` | Average temporal overlap accuracy on matched windows |",
        f"| **Median IoU** | `{report.median_iou:.3f}` | Median temporal overlap accuracy |",
        f"",
        f"## Category-Level Breakdown",
        f"| Category | GT Count | Matches | Precision | Recall |",
        f"| :--- | :---: | :---: | :---: | :---: |",
    ]
    for cat, data in sorted(report.category_metrics.items()):
        lines.append(
            f"| `{cat}` | {data['ground_truth_count']} | {data['matched_count']} | {data['precision']:.1%} | {data['recall']:.1%} |"
        )

    if report.rejection_reasons:
        lines.extend([
            f"",
            f"## Rejection Reasons Encountered",
            f"| Reason | Count |",
            f"| :--- | :---: |",
        ])
        for reason, count in sorted(report.rejection_reasons.items(), key=lambda i: -i[1]):
            lines.append(f"| `{reason}` | {count} |")

    lines.append("")
    return "\n".join(lines)
