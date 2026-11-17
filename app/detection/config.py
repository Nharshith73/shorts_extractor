"""Detection configuration loaded from the Stage 1 YAML file."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import math

import yaml


DEFAULT_WEIGHTS = {
    "audio_rms_db": 1.0,
    "voice_energy_db": 1.0,
    "onset_strength": 1.0,
    "motion": 1.0,
    "cut_change": 1.0,
}
DEFAULT_SCALE_FLOORS = {
    "audio_rms_db": 0.5,
    "voice_energy_db": 0.5,
    "onset_strength": 0.001,
    "motion": 1.0,
    "cut_change": 0.01,
}
DEFAULT_RANKING_WEIGHTS = dict(vocal_spike=2.5, audio_contrast=1.5, audio_spike=0.5,
                               onset_spike=0.5, chat_surge=2.0, visual_activity=0.25)


@dataclass(frozen=True, slots=True)
class DetectionConfig:
    """Tunable Stage 2 signal-processing and anomaly-scoring settings."""

    baseline_seconds: float = 300.0
    warmup_seconds: float = 60.0
    z_threshold: float = 3.0
    single_feature_threshold: float = 6.0
    min_separation_seconds: float = 20.0
    sample_rate: int = 16_000
    video_fps: float = 4.0
    reprocess_poll_seconds: float = 2.0
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    scale_floors: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SCALE_FLOORS))
    relative_scale_floor: float = 0.01
    ranking_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_RANKING_WEIGHTS))
    evidence_z_threshold: float = 2.0
    contrast_seconds: float = 5.0
    contrast_scale_db: float = 2.0
    visual_only_penalty: float = 0.1

    def validate(self) -> None:
        if self.baseline_seconds <= 0:
            raise ValueError("detection.baseline_seconds must be positive")
        if self.warmup_seconds < 0:
            raise ValueError("detection.warmup_seconds cannot be negative")
        if self.z_threshold < 0 or self.single_feature_threshold < 0:
            raise ValueError("detection thresholds cannot be negative")
        if self.min_separation_seconds < 0:
            raise ValueError("detection.min_separation_seconds cannot be negative")
        if self.sample_rate <= 0 or self.video_fps <= 0:
            raise ValueError("detection sample rate and video fps must be positive")
        if self.reprocess_poll_seconds <= 0:
            raise ValueError("detection.reprocess_poll_seconds must be positive")
        unknown = set(self.weights) - set(DEFAULT_WEIGHTS)
        if unknown:
            raise ValueError(f"unsupported detection weights: {', '.join(sorted(unknown))}")
        if any(weight < 0 for weight in self.weights.values()):
            raise ValueError("detection weights cannot be negative")
        if set(self.scale_floors) != set(DEFAULT_SCALE_FLOORS):
            raise ValueError("detection.scale_floors must contain the five feature names")
        if any(value <= 0 for value in self.scale_floors.values()):
            raise ValueError("detection scale floors must be positive")
        if self.relative_scale_floor < 0:
            raise ValueError("detection.relative_scale_floor cannot be negative")
        if set(self.ranking_weights) != set(DEFAULT_RANKING_WEIGHTS) or any(not math.isfinite(value) or value < 0 for value in self.ranking_weights.values()):
            raise ValueError("invalid detection.ranking_weights")
        if any(not math.isfinite(value) or value <= 0 for value in (self.evidence_z_threshold, self.contrast_seconds, self.contrast_scale_db)):
            raise ValueError("evidence and contrast settings must be finite and positive")
        if not 0 <= self.visual_only_penalty <= 1:
            raise ValueError("visual_only_penalty must be 0..1")


def load_detection_config(config_path: Path) -> DetectionConfig:
    """Load only the optional ``detection`` mapping from the project YAML file."""
    with config_path.resolve().open("r", encoding="utf-8") as config_file:
        raw = yaml.safe_load(config_file) or {}
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    section = raw.get("detection", {})
    if not isinstance(section, dict):
        raise ValueError("detection must be a mapping")
    allowed = {
        "baseline_seconds",
        "warmup_seconds",
        "z_threshold",
        "single_feature_threshold",
        "min_separation_seconds",
        "sample_rate",
        "video_fps",
        "reprocess_poll_seconds",
        "weights",
        "scale_floors",
        "relative_scale_floor",
        "chat",
        "ranking_weights", "evidence_z_threshold", "contrast_seconds", "contrast_scale_db", "visual_only_penalty",
    }
    unknown = set(section) - allowed
    if unknown:
        raise ValueError(f"unsupported detection keys: {', '.join(sorted(unknown))}")
    configured_weights = section.get("weights", {})
    if not isinstance(configured_weights, dict):
        raise ValueError("detection.weights must be a mapping")
    weights = dict(DEFAULT_WEIGHTS)
    for name, value in configured_weights.items():
        weights[str(name)] = float(value)
    configured_floors = section.get("scale_floors", {})
    if not isinstance(configured_floors, dict):
        raise ValueError("detection.scale_floors must be a mapping")
    floors = dict(DEFAULT_SCALE_FLOORS)
    for name, value in configured_floors.items():
        floors[str(name)] = float(value)
    ranking_weights = section.get("ranking_weights", {})
    if not isinstance(ranking_weights, dict):
        raise ValueError("detection.ranking_weights must be a mapping")
    config = DetectionConfig(
        baseline_seconds=float(section.get("baseline_seconds", 300)),
        warmup_seconds=float(section.get("warmup_seconds", 60)),
        z_threshold=float(section.get("z_threshold", 3)),
        single_feature_threshold=float(section.get("single_feature_threshold", 6)),
        min_separation_seconds=float(section.get("min_separation_seconds", 20)),
        sample_rate=int(section.get("sample_rate", 16_000)),
        video_fps=float(section.get("video_fps", 4)),
        reprocess_poll_seconds=float(section.get("reprocess_poll_seconds", 2)),
        weights=weights,
        scale_floors=floors,
        relative_scale_floor=float(section.get("relative_scale_floor", 0.01)),
        ranking_weights={**DEFAULT_RANKING_WEIGHTS, **{str(name): float(value) for name, value in ranking_weights.items()}},
        evidence_z_threshold=float(section.get("evidence_z_threshold", 2)),
        contrast_seconds=float(section.get("contrast_seconds", 5)),
        contrast_scale_db=float(section.get("contrast_scale_db", 2)),
        visual_only_penalty=float(section.get("visual_only_penalty", 0.1)),
    )
    config.validate()
    return config


def load_segments_dir(config_path: Path) -> Path:
    """Read the shared Stage 1 segment directory without requiring a source URL."""
    path = config_path.resolve()
    with path.open("r", encoding="utf-8") as config_file:
        raw = yaml.safe_load(config_file) or {}
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    value = raw.get("segments_dir", "data/segments")
    if not isinstance(value, str):
        raise ValueError("segments_dir must be a string path")
    segments_dir = Path(value)
    return segments_dir if segments_dir.is_absolute() else path.parent / segments_dir
