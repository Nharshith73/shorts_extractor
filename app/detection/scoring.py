"""Robust per-stream normalization, event qualification, and peak selection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Mapping, Sequence

import numpy as np

from .config import DetectionConfig
from .features import FEATURE_NAMES
from .models import DetectedEvent, FeatureBin, ScoredBin
from .ranking import ranking_signals


@dataclass(frozen=True, slots=True)
class AuxiliaryFeatures:
    """Optional timestamp-aligned raw signals supplied by an external adapter."""

    values: Mapping[datetime, Mapping[str, float]]
    weights: Mapping[str, float]
    scale_floors: Mapping[str, float]


def robust_zscore(
    value: float,
    baseline: Sequence[float],
    *,
    scale_floor: float = 0.0,
    relative_scale_floor: float = 0.0,
    clip: bool = True,
) -> float:
    """Return a MAD-scaled z-score, normally clipped to 0–10.

    ``clip=False`` exposes the signed value with the same baseline and floors.
    """
    if not baseline:
        return 0.0
    values = np.asarray(baseline, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = max(1.4826 * mad, scale_floor, abs(median) * relative_scale_floor, 1e-8)
    z_score = (value - median) / scale
    return float(np.clip(z_score, 0.0, 10.0) if clip else z_score)


def _feature_value(feature: FeatureBin, name: str) -> float:
    return float(getattr(feature, name))


def score_feature_bins(features: Sequence[FeatureBin], config: DetectionConfig,
                       auxiliary: AuxiliaryFeatures | None = None, *,
                       include_unclipped: bool = False) -> list[ScoredBin]:
    """Score ordered, single-session bins against trailing median/MAD baselines."""
    if not features:
        return []
    ordered = sorted(features, key=lambda feature: feature.wall_utc)
    session_start = ordered[0].wall_utc
    scored: list[ScoredBin] = []
    for index, feature in enumerate(ordered):
        baseline_start = feature.wall_utc - timedelta(seconds=config.baseline_seconds)
        prior = [
            candidate
            for candidate in ordered[:index]
            if candidate.wall_utc >= baseline_start
        ]
        z_scores = {
            name: robust_zscore(
                _feature_value(feature, name),
                [_feature_value(item, name) for item in prior],
                scale_floor=config.scale_floors[name],
                relative_scale_floor=config.relative_scale_floor,
                clip=not include_unclipped,
            )
            for name in FEATURE_NAMES
        }
        unclipped = dict(z_scores) if include_unclipped else None
        if include_unclipped:
            z_scores = {name: float(np.clip(value, 0.0, 10.0)) for name, value in z_scores.items()}
        score = sum(config.weights.get(name, 0.0) * z_scores[name] for name in FEATURE_NAMES)
        unclipped_score = sum(config.weights.get(name, 0.0) * value for name, value in unclipped.items()) if unclipped is not None else None
        if auxiliary is not None:
            for name, weight in auxiliary.weights.items():
                if weight <= 0:
                    continue
                component = robust_zscore(
                    auxiliary.values.get(feature.wall_utc, {}).get(name, 0.0),
                    [auxiliary.values.get(item.wall_utc, {}).get(name, 0.0) for item in prior],
                    scale_floor=auxiliary.scale_floors[name],
                    relative_scale_floor=config.relative_scale_floor,
                    clip=not include_unclipped,
                )
                if unclipped is not None:
                    unclipped[name] = component
                    unclipped_score += weight * component
                    component = float(np.clip(component, 0.0, 10.0))
                z_scores[name] = component
                score += weight * z_scores[name]
        elapsed = (feature.wall_utc - session_start).total_seconds()
        ranking_score, ranking_components, reasons, eligible = ranking_signals(feature, prior, z_scores, config,
            auxiliary.values.get(feature.wall_utc) if auxiliary is not None else None,
            auxiliary.weights if auxiliary is not None else None)
        scored.append(
            ScoredBin(
                feature=feature,
                z_scores=z_scores,
                score=float(score),
                qualifies=elapsed >= config.warmup_seconds and eligible,
                unclipped_z_scores=unclipped,
                unclipped_score=unclipped_score,
                ranking_score=ranking_score, reason_codes=reasons, ranking_components=ranking_components, eligible=eligible,
            )
        )
    return scored


def _regions(scored: Iterable[ScoredBin]) -> list[list[ScoredBin]]:
    regions: list[list[ScoredBin]] = []
    active: list[ScoredBin] = []
    for item in scored:
        if item.qualifies:
            if active:
                previous = active[-1].feature
                expected_next = previous.wall_utc + timedelta(seconds=previous.bin_seconds)
                if item.feature.wall_utc > expected_next + timedelta(milliseconds=50):
                    regions.append(active)
                    active = []
            active.append(item)
        elif active:
            regions.append(active)
            active = []
    if active:
        regions.append(active)
    return regions


def pick_events(scored: Sequence[ScoredBin], config: DetectionConfig) -> list[DetectedEvent]:
    """Convert qualifying regions into separated, highest-score events."""
    candidates: list[DetectedEvent] = []
    for region in _regions(scored):
        peak = max(region, key=lambda item: (item.ranking_score if item.ranking_score is not None else item.score,
                                            item.unclipped_score or 0, -item.feature.wall_utc.timestamp()))
        candidates.append(
            DetectedEvent(
                source_id=peak.feature.source_id,
                session_id=peak.feature.session_id,
                peak_wall_utc=peak.feature.wall_utc,
                start_wall_utc=region[0].feature.wall_utc,
                end_wall_utc=region[-1].feature.wall_utc
                + timedelta(seconds=region[-1].feature.bin_seconds),
                score=peak.ranking_score if peak.ranking_score is not None else peak.score,
                components=peak.z_scores,
                unclipped_z_scores=peak.unclipped_z_scores,
                unclipped_score=peak.unclipped_score,
                reason_codes=peak.reason_codes, ranking_components=peak.ranking_components,
            )
        )
    selected: list[DetectedEvent] = []
    for candidate in sorted(candidates, key=lambda event: (-event.score, event.peak_wall_utc)):
        if all(
            abs((candidate.peak_wall_utc - chosen.peak_wall_utc).total_seconds())
            >= config.min_separation_seconds
            for chosen in selected
        ):
            selected.append(candidate)
    return sorted(selected, key=lambda event: event.peak_wall_utc)


def detect_events(features: Sequence[FeatureBin], config: DetectionConfig,
                  auxiliary: AuxiliaryFeatures | None = None) -> tuple[list[ScoredBin], list[DetectedEvent]]:
    """Score one session and return bins plus qualified, separated events."""
    scored = score_feature_bins(features, config, auxiliary)
    return scored, pick_events(scored, config)
