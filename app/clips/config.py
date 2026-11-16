"""Configuration for candidate extraction."""

from dataclasses import dataclass
from pathlib import Path
import math

import yaml


@dataclass(frozen=True, slots=True)
class CandidateConfig:
    pre_roll_seconds: float = 15
    post_roll_seconds: float = 10
    min_seconds: float = 20
    max_seconds: float = 60
    merge_gap_seconds: float = 3
    grace_seconds: float = 0.5
    wait_timeout_seconds: float = 180
    poll_seconds: float = 2
    output_dir: Path = Path("data/candidates")
    video_encoder: str = "libx264"
    encoder: str | None = None

    @property
    def requested_encoder(self) -> str:
        """Canonical selection, retaining the older video_encoder setting."""
        return self.encoder if self.encoder is not None else self.video_encoder

    def validate(self) -> None:
        values = (self.pre_roll_seconds, self.post_roll_seconds, self.min_seconds,
                  self.max_seconds, self.merge_gap_seconds, self.grace_seconds,
                  self.wait_timeout_seconds, self.poll_seconds)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("candidate durations must be finite and non-negative")
        if not 0 < self.min_seconds <= self.max_seconds or self.pre_roll_seconds >= self.max_seconds:
            raise ValueError("require 0 < min_seconds <= max_seconds and pre_roll_seconds < max_seconds")
        if self.wait_timeout_seconds <= 0 or self.poll_seconds <= 0:
            raise ValueError("candidate timeout and polling interval must be positive")
        if self.requested_encoder not in ("auto", "libx264", "h264_nvenc"):
            raise ValueError("encoder must be auto, libx264, or h264_nvenc")


def load_config(path: Path) -> tuple[CandidateConfig, Path, float]:
    path = path.resolve()
    with path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    section = raw.get("candidates", {})
    if not isinstance(section, dict):
        raise ValueError("candidates must be a mapping")
    unknown = set(section) - set(CandidateConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown candidates keys: {sorted(unknown)}")
    if "encoder" in section and "video_encoder" in section and section["encoder"] != section["video_encoder"]:
        raise ValueError("set candidates.encoder or video_encoder, not conflicting values for both")
    values = {key: (value if key in ("output_dir", "video_encoder", "encoder") else float(value))
              for key, value in section.items()}
    output = Path(values.get("output_dir", "data/candidates"))
    values["output_dir"] = output if output.is_absolute() else path.parent / output
    config = CandidateConfig(**values)
    config.validate()
    segments = Path(raw.get("segments_dir", "data/segments"))
    return config, segments if segments.is_absolute() else path.parent / segments, float(raw.get("retention_seconds", 900))
