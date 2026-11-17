"""Explainable ranking signals derived from saved features, without media decoding."""

from datetime import timedelta
from typing import Mapping, Sequence

import numpy as np

from .config import DetectionConfig
from .models import FeatureBin


def ranking_signals(feature: FeatureBin, prior: Sequence[FeatureBin], z_scores: dict[str, float],
                    config: DetectionConfig, chat_values: Mapping[str, float] | None = None,
                    chat_weights: Mapping[str, float] | None = None) -> tuple[float, dict[str, float], tuple[str, ...], bool]:
    recent = [item for item in prior if item.wall_utc >= feature.wall_utc-timedelta(seconds=config.contrast_seconds)]
    contrast_db = max((getattr(feature, name)-float(np.median([getattr(item, name) for item in recent]))
                       for name in ("audio_rms_db", "voice_energy_db")), default=0) if recent else 0
    contrast = float(np.clip(contrast_db/config.contrast_scale_db, 0, 10))
    # Band-limited RMS / broadband RMS suppresses low-frequency engine ambience.
    voice_ratio = min(1.0, 10 ** (min(0, feature.voice_energy_db-feature.audio_rms_db)/20))
    threshold = config.evidence_z_threshold
    # Wide-dynamic-range VODs can have small 300s z-scores even on real shouts.
    # Local contrast is independent evidence, not another long-baseline gate.
    vocal = max(z_scores.get("voice_energy_db", 0), contrast)*voice_ratio*min(1.0, contrast/threshold)
    chat_scores = {name: min(10, value*(chat_weights or {}).get(name, 1)) for name, value in z_scores.items() if name.startswith("chat_")}
    volume = max((value for name, value in chat_scores.items() if "rate" in name or "chatters" in name), default=0)
    absolute_volume = max((value for name, value in (chat_values or {}).items() if "rate" in name or "chatters" in name), default=0)
    # A lone uppercase/emote message is not a chat surge. Ratios need volume.
    ratio = max((value for name, value in chat_scores.items() if "emote" in name or "caps" in name), default=0)
    chat = max(volume, ratio if volume >= threshold or absolute_volume >= 3 else 0)
    visual = max(z_scores.get("motion", 0), z_scores.get("cut_change", 0))
    audio = z_scores.get("audio_rms_db", 0)
    vocal_evidence = vocal >= threshold and contrast >= threshold
    audio_motion = contrast >= 2*threshold and audio >= threshold and visual >= threshold
    eligible = vocal_evidence or audio_motion or chat >= threshold
    components = dict(vocal_spike=float(vocal), audio_contrast=contrast, audio_spike=audio,
                      onset_spike=z_scores.get("onset_strength", 0), chat_surge=chat, visual_activity=visual)
    codes: list[str] = []
    if vocal_evidence:
        codes.append("vocal_spike")
    if contrast >= threshold:
        codes.append("audio_contrast")
    if chat >= threshold:
        codes.append("chat_surge")
    if audio_motion:
        codes.append("audio_motion_peak")
    if visual >= threshold and not eligible:
        codes.append("visual_only_penalized")
    score = sum(config.ranking_weights[name]*value for name, value in components.items())
    if not eligible:
        score *= config.visual_only_penalty
    return float(score), components, tuple(codes), eligible
