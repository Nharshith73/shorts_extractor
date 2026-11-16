"""VOD-only controls; live candidate defaults are independent."""

from dataclasses import dataclass, replace
import math
from pathlib import Path

import yaml

from app.clips.config import CandidateConfig, load_config as load_candidates


@dataclass(frozen=True, slots=True)
class SelectionConfig:
    clips_per_hour: float = 12
    min_score: float | None = None
    min_separation_seconds: float = 30


@dataclass(frozen=True, slots=True)
class VodConfig:
    chunk_seconds: float = 300
    progress_seconds: float = 15
    pre_roll_seconds: float = 8
    post_roll_seconds: float = 6
    min_seconds: float = 15
    max_seconds: float = 45


def load_settings(path: Path) -> tuple[VodConfig, SelectionConfig, CandidateConfig]:
    with path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    vod = VodConfig(**raw.get("vod", {}))
    selection = SelectionConfig(**raw.get("selection", {}))
    numeric = (*vars_for(vod), selection.clips_per_hour, selection.min_separation_seconds)
    if any(not math.isfinite(value) or value < 0 for value in numeric):
        raise ValueError("VOD and selection settings must be finite and non-negative")
    if vod.chunk_seconds <= 0 or vod.progress_seconds <= 0 or selection.clips_per_hour <= 0:
        raise ValueError("chunk/progress seconds and clips_per_hour must be positive")
    if selection.min_score is not None and not math.isfinite(selection.min_score):
        raise ValueError("selection.min_score must be finite or null")
    base, _, _ = load_candidates(path)
    windows = replace(base, pre_roll_seconds=vod.pre_roll_seconds, post_roll_seconds=vod.post_roll_seconds,
                      min_seconds=vod.min_seconds, max_seconds=vod.max_seconds)
    windows.validate()
    return vod, selection, windows


def vars_for(config: VodConfig) -> tuple[float, ...]:
    return (config.chunk_seconds, config.progress_seconds, config.pre_roll_seconds,
            config.post_roll_seconds, config.min_seconds, config.max_seconds)


def parse_timestamp(text: str) -> float:
    parts = text.split(":")
    if len(parts) != 3:
        raise ValueError("time must be HH:MM:SS (fractional seconds allowed)")
    hours, minutes, seconds = map(float, parts)
    if not all(math.isfinite(value) for value in (hours, minutes, seconds)) or hours < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
        raise ValueError("invalid HH:MM:SS timestamp")
    return hours * 3600 + minutes * 60 + seconds
