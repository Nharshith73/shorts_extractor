"""VOD peak ranking independent of the threshold event gate."""

from dataclasses import dataclass
import math

from app.clips.config import CandidateConfig
from app.clips.duration import peak_bounds
from app.detection.config import DetectionConfig
from app.detection.models import DetectedEvent, ScoredBin
from .settings import SelectionConfig
from .timeline import media_seconds, media_time


def rule_name(item: ScoredBin, config: DetectionConfig) -> str:
    multi = sum(value >= config.z_threshold for value in item.z_scores.values()) >= 2
    single = max(item.z_scores.values(), default=0) >= config.single_feature_threshold
    return "multi+single" if multi and single else "multi" if multi else "single" if single else "ranked_only"


def peak_window(event: DetectedEvent, config: CandidateConfig, start: float, end: float) -> tuple[float, float] | None:
    """Preserve lead-in/reaction; shift at media edges instead of losing context."""
    peak = media_seconds(event.peak_wall_utc)
    left = max(start, peak - config.pre_roll_seconds)
    right = max(peak + config.post_roll_seconds,
                         media_seconds(event.end_wall_utc) + config.post_roll_seconds,
                         left + config.min_seconds)
    left, right = peak_bounds(left, right, peak, maximum=config.duration_limit, lower=start, upper=end)
    if not left <= peak < right or right - left < config.min_seconds - 1e-6:
        return None
    return left, right


def rank_peaks(scored: list[ScoredBin], selection: SelectionConfig, detection: DetectionConfig,
               windows: CandidateConfig, start: float, end: float) -> list[tuple[DetectedEvent, str]]:
    budget = selection.top_k or max(3, min(40, math.ceil(selection.clips_per_hour * (end - start) / 3600)))
    selected: list[tuple[DetectedEvent, str]] = []
    accepted_windows: list[tuple[float, float]] = []
    for item in sorted(scored, key=lambda item: (
        -(item.ranking_score if item.ranking_score is not None else item.score),
        -(item.unclipped_score if item.unclipped_score is not None else item.score), item.feature.wall_utc)):
        at = media_seconds(item.feature.wall_utc)
        if at - start < detection.warmup_seconds:
            continue
        if item.eligible is False:
            continue
        if item.eligible is None and max((value for name, value in item.z_scores.items()
                                         if name.startswith("chat_") or name in ("audio_rms_db", "voice_energy_db", "onset_strength")), default=0) < detection.evidence_z_threshold:
            continue
        if selection.min_score is not None and (item.ranking_score if item.ranking_score is not None else item.score) < selection.min_score:
            continue
        event = DetectedEvent(item.feature.source_id, item.feature.session_id, item.feature.wall_utc,
                              item.feature.wall_utc, media_time(at + item.feature.bin_seconds),
                              item.ranking_score if item.ranking_score is not None else item.score, item.z_scores,
                              item.unclipped_z_scores, item.unclipped_score, item.reason_codes, item.ranking_components)
        bounds = peak_window(event, windows, start, end)
        if bounds is None:
            continue
        if any(abs(at - media_seconds(other.peak_wall_utc)) < selection.min_separation_seconds for other, _ in selected):
            continue
        left, right = bounds
        if any(min(right, old_right) - max(left, old_left) > 2.0 for old_left, old_right in accepted_windows):
            continue
        selected.append((event, rule_name(item, detection)))
        accepted_windows.append(bounds)
        if len(selected) >= budget:
            break
    return selected
