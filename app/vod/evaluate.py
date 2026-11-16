"""Evaluate ranked event peaks against weak temporal interval labels."""

import csv
from dataclasses import dataclass
import json
import logging
import math
from pathlib import Path
import random
from typing import Sequence

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class WeakLabel:
    start: float
    duration: float
    weight: float


@dataclass(frozen=True, slots=True)
class Metrics:
    recall_at_k: float
    precision_at_k: float
    weighted_recall: float
    missed: tuple[int, ...]


def load_labels(path: Path) -> list[WeakLabel]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"start_seconds", "duration_seconds", "weight"}.issubset(reader.fieldnames or []):
            raise ValueError("weak labels need start_seconds,duration_seconds,weight columns")
        labels = [WeakLabel(float(row["start_seconds"]), float(row["duration_seconds"]), float(row["weight"])) for row in reader]
    if any(not math.isfinite(value) or value < 0 for label in labels for value in (label.start, label.duration, label.weight)):
        raise ValueError("weak label values must be finite and non-negative")
    return labels


def metrics(peaks: Sequence[float], labels: Sequence[WeakLabel], tolerance: float) -> Metrics:
    """One-to-one interval matching, with maximum match count via augmenting paths."""
    owner: dict[int, int] = {}
    def match(prediction: int, visited: set[int]) -> bool:
        eligible = [i for i, label in enumerate(labels) if label.start - tolerance <= peaks[prediction] <= label.start + label.duration + tolerance]
        eligible.sort(key=lambda i: abs(peaks[prediction] - (labels[i].start + labels[i].duration / 2)))
        for index in eligible:
            if index in visited:
                continue
            visited.add(index)
            if index not in owner or match(owner[index], visited):
                owner[index] = prediction
                return True
        return False
    for prediction in range(len(peaks)):
        match(prediction, set())
    matched = len(owner)
    total_weight = sum(label.weight for label in labels)
    return Metrics(matched / len(labels) if labels else 0.0, matched / len(peaks) if peaks else 0.0,
                   sum(labels[index].weight for index in owner) / total_weight if total_weight else 0.0,
                   tuple(index for index in range(len(labels)) if index not in owner))


def random_baseline(duration: float, labels: Sequence[WeakLabel], count: int, tolerance: float,
                    *, trials: int = 200, seed: int = 42) -> tuple[float, float, float]:
    rng = random.Random(seed)
    results = [metrics([rng.uniform(0, duration) for _ in range(count)], labels, tolerance) for _ in range(trials)]
    return tuple(sum(getattr(result, field) for result in results) / trials
                 for field in ("recall_at_k", "precision_at_k", "weighted_recall"))


def evaluate(output: Path, labels_path: Path, tolerance: float, top_k: int) -> Metrics:
    if not math.isfinite(tolerance) or tolerance < 0 or top_k <= 0:
        raise ValueError("tolerance must be finite/non-negative and top-k positive")
    data = json.loads((output / "events.json").read_text(encoding="utf-8"))
    duration = float(data["duration_seconds"])
    labels = load_labels(labels_path)
    if any(label.start + label.duration > duration + 0.15 for label in labels):
        raise ValueError("weak labels extend beyond this VOD; check offset alignment")
    ranked = sorted(data["events"], key=lambda event: (-event["score"], event["peak_time_seconds"]))[:top_k]
    result = metrics([event["peak_time_seconds"] for event in ranked], labels, tolerance)
    LOGGER.info("Event-peak evaluation: K=%d available=%d tolerance=%.3fs; one prediction per label", top_k, len(ranked), tolerance)
    LOGGER.info("Detector recall@K=%.4f precision@K=%.4f view-weighted recall=%.4f", result.recall_at_k, result.precision_at_k, result.weighted_recall)
    for index in result.missed:
        label = labels[index]
        LOGGER.info("Missed label %d: %.3fs .. %.3fs weight=%.3f", index + 1, label.start, label.start + label.duration, label.weight)
    baseline = random_baseline(duration, labels, len(ranked), tolerance)
    LOGGER.info("Random baseline (200 trials, seed=42, %d predictions): mean recall@K=%.4f precision@K=%.4f view-weighted recall=%.4f", len(ranked), *baseline)
    return result
