"""Explain labeled windows using the same scoring and peak selection as detection."""

from __future__ import annotations

import argparse
import csv
import logging
import random
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Sequence

from .config import DetectionConfig, load_detection_config
from .evaluate import Label, load_labels
from .features import FEATURE_NAMES
from .models import DetectedEvent, FeatureBin, ScoredBin
from .scoring import pick_events, score_feature_bins

LOGGER = logging.getLogger(__name__)
ORIGIN = datetime(1970, 1, 1, tzinfo=UTC)


def load_features(path: Path) -> list[FeatureBin]:
    """Read the CSV produced by the offline detector."""
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        required = {"time_seconds", "bin_seconds", *FEATURE_NAMES}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"features CSV must contain: {', '.join(sorted(required))}")
        rows = list(reader)
    features = [
        FeatureBin(
            source_id="offline", session_id="offline",
            wall_utc=ORIGIN + timedelta(seconds=float(row["time_seconds"])),
            bin_seconds=float(row["bin_seconds"]),
            **{name: float(row[name]) for name in FEATURE_NAMES},
        )
        for row in rows
    ]
    if not features:
        raise ValueError("features CSV is empty")
    return sorted(features, key=lambda feature: feature.wall_utc)


def seconds(item: ScoredBin) -> float:
    return (item.feature.wall_utc - ORIGIN).total_seconds()


def rules(item: ScoredBin, config: DetectionConfig) -> tuple[bool, bool]:
    """Return multi-feature and single-feature signal rule results for one bin."""
    return (
        sum(z >= config.z_threshold for z in item.z_scores.values()) >= 2,
        max(item.z_scores.values(), default=0.0) >= config.single_feature_threshold,
    )


@dataclass(frozen=True, slots=True)
class WindowExplanation:
    maxima: dict[str, tuple[float, float]]
    best: ScoredBin
    multi: bool
    single: bool
    reasons: tuple[str, ...]


def explain_window(
    window: Sequence[ScoredBin],
    unfiltered_window: Sequence[ScoredBin],
    selected: Sequence[DetectedEvent],
    config: DetectionConfig,
    session_start: datetime,
) -> WindowExplanation:
    """Explain rule and peak outcomes, comparing a no-floor counterfactual."""
    if not window:
        raise ValueError("label window contains no feature bins")
    best = max(window, key=lambda item: (item.score, -seconds(item)))
    maxima = {
        name: (item.z_scores[name], seconds(item))
        for name in FEATURE_NAMES
        for item in [max(window, key=lambda candidate: candidate.z_scores[name])]
    }
    multi, single = rules(best, config)
    reasons: list[str] = []
    has_signal = any(any(rules(item, config)) for item in window)
    qualified = [item for item in window if item.qualifies]
    if not has_signal:
        max_count = max(sum(z >= config.z_threshold for z in item.z_scores.values()) for item in window)
        reasons.append("only one feature above" if max_count == 1 else "below z_threshold")
    if (all((item.feature.wall_utc - session_start).total_seconds() < config.warmup_seconds for item in window)
        or (has_signal and not qualified and any(
            any(rules(item, config)) and (item.feature.wall_utc - session_start).total_seconds() < config.warmup_seconds
            for item in window
        ))):
        reasons.append("warmup")
    floor_suppressed = any(
        any(rules(raw, config)) and not any(rules(current, config))
        and (current.feature.wall_utc - session_start).total_seconds() >= config.warmup_seconds
        for current, raw in zip(window, unfiltered_window)
    )
    if floor_suppressed:
        reasons.append("suppressed by a scale floor (per-bin counterfactual)")
    if qualified:
        # A qualified bin need not be a region peak. Only call it separated
        # when a stronger selected event actually falls inside its exclusion radius.
        blocked = any(
            event.session_id == item.feature.session_id
            and event.score > item.score
            and 0 < abs((event.peak_wall_utc - item.feature.wall_utc).total_seconds()) < config.min_separation_seconds
            for item in qualified for event in selected
        )
        in_selected_region = any(
            event.start_wall_utc <= item.feature.wall_utc < event.end_wall_utc
            for item in qualified for event in selected
        )
        if blocked and not in_selected_region:
            reasons.append("within min_separation of a stronger event")
        elif not in_selected_region:
            reasons.append("qualified region peak not selected (tie or separation)")
    if not reasons:
        reasons.append("none: selected event overlaps window")
    return WindowExplanation(maxima, best, multi, single, tuple(reasons))


def window_at(scored: Sequence[ScoredBin], center: float, tolerance: float) -> list[ScoredBin]:
    return [item for item in scored if abs(seconds(item) - center) <= tolerance]


def background_centers(
    scored: Sequence[ScoredBin], labels: Sequence[Label], tolerance: float,
    *, count: int = 200, seed: int = 42,
) -> list[float]:
    """Sample full-width windows that do not overlap any labeled window."""
    first, last = seconds(scored[0]), seconds(scored[-1]) + scored[-1].feature.bin_seconds
    eligible = [seconds(item) for item in scored
                if first <= seconds(item) - tolerance and seconds(item) + tolerance <= last
                and all(abs(seconds(item) - label.time_seconds) > 2 * tolerance for label in labels)]
    if not eligible:
        return []
    rng = random.Random(seed)
    # Sampling with replacement preserves the requested 200 comparisons even
    # when only a small amount of unlabeled footage is available.
    return [rng.choice(eligible) for _ in range(count)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Explain detection scores around labels")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=8.0)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        if args.tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        config = load_detection_config(args.config)
        features = load_features(args.features)
        labels = load_labels(args.labels)
        scored = score_feature_bins(features, config)
        no_floor = replace(config, scale_floors={name: 0.0 for name in FEATURE_NAMES}, relative_scale_floor=0.0)
        unfloored = score_feature_bins(features, no_floor)
        selected = pick_events(scored, config)
        LOGGER.info("Effective detection config: z_threshold=%s single_feature_threshold=%s weights=%s baseline_seconds=%s warmup_seconds=%s min_separation_seconds=%s scale_floors=%s relative_scale_floor=%s", config.z_threshold, config.single_feature_threshold, config.weights, config.baseline_seconds, config.warmup_seconds, config.min_separation_seconds, config.scale_floors, config.relative_scale_floor)
        label_windows: list[list[ScoredBin]] = []
        for label in labels:
            window = window_at(scored, label.time_seconds, args.tolerance)
            if not window:
                LOGGER.warning("Label %.3fs %s: no feature bins in window", label.time_seconds, label.note)
                continue
            raw_window = window_at(unfloored, label.time_seconds, args.tolerance)
            explanation = explain_window(window, raw_window, selected, config, features[0].wall_utc)
            label_windows.append(window)
            LOGGER.info("Label %.3fs %s", label.time_seconds, label.note)
            for name, (value, at) in explanation.maxima.items():
                LOGGER.info("  %s max z=%.3f at %.3fs", name, value, at)
            multi_times = [seconds(item) for item in window if rules(item, config)[0]]
            single_times = [seconds(item) for item in window if rules(item, config)[1]]
            LOGGER.info("  max combined=%.3f at %.3fs; at max: multi=%s single=%s", explanation.best.score, seconds(explanation.best), explanation.multi, explanation.single)
            LOGGER.info("  window rule passes: multi=%s single=%s; blocking: %s", multi_times or "none", single_times or "none", "; ".join(explanation.reasons))
        centers = background_centers(scored, labels, args.tolerance)
        backgrounds = [window_at(scored, center, args.tolerance) for center in centers]
        LOGGER.info("Summary: %d/%d labeled windows have bins; %d random unlabeled windows, seed=42, sampled with replacement", len(label_windows), len(labels), len(backgrounds))
        for name in FEATURE_NAMES:
            def fraction(windows: Sequence[Sequence[ScoredBin]]) -> float:
                return sum(max(item.z_scores[name] for item in window) >= config.z_threshold for window in windows) / len(windows) if windows else 0.0
            LOGGER.info("  %s labels=%.1f%% background=%.1f%%", name, 100 * fraction(label_windows), 100 * fraction(backgrounds))
        if not backgrounds:
            LOGGER.warning("No full-width unlabeled background windows are available")
        return 0
    except (OSError, ValueError, KeyError) as exc:
        LOGGER.error("Explanation failed: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
