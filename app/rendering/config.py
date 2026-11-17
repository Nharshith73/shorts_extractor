"""Independent final-render settings; capture configuration stays untouched."""

from dataclasses import dataclass, field
import math
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class FacecamRegion:
    x: int
    y: int
    w: int
    h: int


@dataclass(frozen=True, slots=True)
class RenderConfig:
    whisper_model: str = "small"
    compute_type: str = "int8"
    device: str = "cpu"
    language: str = "auto"
    min_avg_logprob: float = -1.0
    max_no_speech_prob: float = 0.6
    captions_enabled: bool = True
    caption_style: str = "bold_yellow"
    uppercase: bool = True
    max_words: int = 3
    font_size: int = 80
    margin_horizontal: int = 80
    margin_bottom: int = 400
    outline: float = 4
    shadow: float = 2
    fonts_dir: Path = field(default_factory=lambda: ROOT / "assets" / "fonts")
    layout: str = "blurred_background"
    facecam_region: FacecamRegion | None = None
    crop_center_x: float = 0.5
    target_lufs: float = -14
    encoder: str = "auto"
    crf: int = 18
    preset: str = "medium"

    def validate(self) -> None:
        if not self.whisper_model or not self.compute_type or not self.language:
            raise ValueError("render model, compute_type and language cannot be empty")
        if self.device not in ("cpu", "auto", "cuda"):
            raise ValueError("render.device must be cpu, auto, or cuda")
        if self.caption_style not in ("bold_yellow", "clean_white", "boxed"):
            raise ValueError("unknown render.caption_style")
        if self.layout not in ("blurred_background", "stacked", "center_crop"):
            raise ValueError("unknown render.layout")
        if self.encoder not in ("auto", "libx264", "h264_nvenc"):
            raise ValueError("render.encoder must be auto, libx264, or h264_nvenc")
        if not isinstance(self.captions_enabled, bool) or not isinstance(self.uppercase, bool):
            raise ValueError("caption toggles must be YAML booleans")
        for value in (self.max_words, self.font_size, self.margin_horizontal, self.margin_bottom, self.crf):
            if type(value) is not int:
                raise ValueError("word count, font/margin sizes and crf must be integers")
        numbers = (self.min_avg_logprob, self.max_no_speech_prob, self.crop_center_x,
                   self.target_lufs, self.outline, self.shadow)
        if any(not math.isfinite(value) for value in numbers):
            raise ValueError("render numeric settings must be finite")
        if not 0 <= self.max_no_speech_prob <= 1 or self.min_avg_logprob > 0:
            raise ValueError("invalid hallucination thresholds")
        if not 1 <= self.max_words <= 3 or not 20 <= self.font_size <= 160:
            raise ValueError("require 1..3 words and font_size 20..160")
        if not 20 <= self.margin_horizontal <= 300 or not 0 <= self.outline <= 12 or not 0 <= self.shadow <= 12:
            raise ValueError("invalid caption margins, outline or shadow")
        if not math.ceil(1920 * 0.18 + self.outline + self.shadow + 8) <= self.margin_bottom <= 900:
            raise ValueError("margin_bottom must keep captions above the bottom 18% safe area")
        if not 0 <= self.crop_center_x <= 1 or not -70 <= self.target_lufs <= -5 or not 0 <= self.crf <= 51:
            raise ValueError("invalid crop center, LUFS target or crf")
        presets = ("ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow")
        if self.preset not in presets:
            raise ValueError("invalid x264 preset")
        if self.facecam_region is not None:
            region = self.facecam_region
            if any(type(value) is not int for value in (region.x, region.y, region.w, region.h)):
                raise ValueError("facecam_region uses integer source pixels")
            if min(region.x, region.y) < 0 or min(region.w, region.h) <= 0:
                raise ValueError("invalid facecam_region")


def load_render_config(path: Path) -> RenderConfig:
    with path.resolve().open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    if not isinstance(raw, dict) or not isinstance(raw.get("render", {}), dict):
        raise ValueError("configuration and render must be mappings")
    section = dict(raw.get("render", {}))
    if set(section) - set(RenderConfig.__dataclass_fields__):
        raise ValueError(f"unknown render keys: {sorted(set(section) - set(RenderConfig.__dataclass_fields__))}")
    if "encoder" not in section:
        candidates = raw.get("candidates", {})
        section["encoder"] = candidates.get("encoder", candidates.get("video_encoder", "auto"))
    if section.get("facecam_region") is not None:
        region = section["facecam_region"]
        if not isinstance(region, dict) or set(region) != {"x", "y", "w", "h"}:
            raise ValueError("facecam_region requires x,y,w,h")
        section["facecam_region"] = FacecamRegion(**region)
    if "fonts_dir" in section:
        fonts = Path(section["fonts_dir"])
        section["fonts_dir"] = fonts if fonts.is_absolute() else path.resolve().parent / fonts
    config = RenderConfig(**section)
    config.validate()
    return config
