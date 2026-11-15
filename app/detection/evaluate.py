"""Evaluate an offline Stage 2 timeline against timestamped human labels."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TimelineEvent:
    time_seconds: float
    score: float


@dataclass(frozen=True, slots=True)
class Label:
    time_seconds: float
    note: str


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    precision: float
    recall: float
    f1: float
    top_k_hit_rate: float
    misses: list[Label]
    false_positives: list[TimelineEvent]


def load_events(path: Path) -> list[TimelineEvent]:
    """Read event JSON emitted by ``app.detection.offline``."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    events = raw["events"] if isinstance(raw, dict) else raw
    if not isinstance(events, list):
        raise ValueError("timeline JSON must contain an events list")
    return sorted(
        [
            TimelineEvent(float(event["peak_time_seconds"]), float(event.get("score", 0)))
            for event in events
        ],
        key=lambda event: event.time_seconds,
    )


def load_labels(path: Path) -> list[Label]:
    """Read the required ``time_seconds,note`` CSV label format."""
    with path.open("r", newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))
    if not rows or not {"time_seconds", "note"} <= set(rows[0]):
        raise ValueError("labels CSV must contain time_seconds,note columns")
    return sorted(
        [Label(float(row["time_seconds"]), str(row["note"])) for row in rows],
        key=lambda label: label.time_seconds,
    )


def match_events(
    events: Sequence[TimelineEvent], labels: Sequence[Label], tolerance_seconds: float
) -> tuple[list[tuple[TimelineEvent, Label]], list[TimelineEvent], list[Label]]:
    """Greedily pair each event with its nearest unmatched label in tolerance."""
    unmatched_labels = set(range(len(labels)))
    matches: list[tuple[TimelineEvent, Label]] = []
    false_positives: list[TimelineEvent] = []
    for event in events:
        eligible = [
            index
            for index in unmatched_labels
            if abs(event.time_seconds - labels[index].time_seconds) <= tolerance_seconds
        ]
        if not eligible:
            false_positives.append(event)
            continue
        best = min(eligible, key=lambda index: abs(event.time_seconds - labels[index].time_seconds))
        unmatched_labels.remove(best)
        matches.append((event, labels[best]))
    return matches, false_positives, [labels[index] for index in sorted(unmatched_labels)]


def evaluate(
    events: Sequence[TimelineEvent],
    labels: Sequence[Label],
    tolerance_seconds: float,
    top_k: int,
) -> EvaluationResult:
    """Calculate precision/recall/F1 and top-K label coverage."""
    matches, false_positives, misses = match_events(events, labels, tolerance_seconds)
    precision = len(matches) / len(events) if events else 0.0
    recall = len(matches) / len(labels) if labels else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    top_events = sorted(events, key=lambda event: (-event.score, event.time_seconds))[:top_k]
    top_matches, _, _ = match_events(top_events, labels, tolerance_seconds)
    top_k_hit_rate = len(top_matches) / len(labels) if labels else 0.0
    return EvaluationResult(precision, recall, f1, top_k_hit_rate, misses, false_positives)


def tune_cutoff(
    events: Sequence[TimelineEvent], labels: Sequence[Label], tolerance_seconds: float
) -> float:
    """Choose the score floor with the best F1 on the supplied tuning half."""
    candidates = sorted({event.score for event in events} | {float("-inf")})
    best_cutoff = float("-inf")
    best_f1 = -1.0
    for cutoff in candidates:
        result = evaluate([event for event in events if event.score >= cutoff], labels, tolerance_seconds, 10)
        if result.f1 > best_f1:
            best_cutoff = cutoff
            best_f1 = result.f1
    return best_cutoff


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a detection timeline against human labels.")
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=8.0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--split", action="store_true", help="Tune a score cutoff on first-half labels")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _log_result(result: EvaluationResult) -> None:
    LOGGER.info("precision=%.3f recall=%.3f F1=%.3f top-K hit rate=%.3f", result.precision, result.recall, result.f1, result.top_k_hit_rate)
    for label in result.misses:
        LOGGER.info("missed label: %.3fs (%s)", label.time_seconds, label.note)
    for event in result.false_positives:
        LOGGER.info("false positive: %.3fs score=%.3f", event.time_seconds, event.score)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        if args.tolerance < 0 or args.top_k <= 0:
            raise ValueError("tolerance must be non-negative and top-k must be positive")
        events = load_events(args.events)
        labels = load_labels(args.labels)
        if args.split:
            if len(labels) < 2:
                raise ValueError("--split requires at least two labels")
            split_index = len(labels) // 2
            boundary = labels[split_index].time_seconds
            cutoff = tune_cutoff(
                [event for event in events if event.time_seconds < boundary],
                labels[:split_index],
                args.tolerance,
            )
            LOGGER.info("tuned score cutoff %.3f; reporting on labels at/after %.3fs", cutoff, boundary)
            result = evaluate(
                [event for event in events if event.time_seconds >= boundary and event.score >= cutoff],
                labels[split_index:],
                args.tolerance,
                args.top_k,
            )
        else:
            result = evaluate(events, labels, args.tolerance, args.top_k)
        _log_result(result)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        LOGGER.error("Evaluation failed: %s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
