"""FFmpeg original-source export with relative ASS/font filter references."""

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess

from app.clips.render import EncoderSelector, _seek_input
from .captions import font_family, generate_ass
from .config import RenderConfig
from .layouts import LayoutGraph, build_layout
from .transcription import Word


@dataclass(frozen=True, slots=True)
class MediaInfo:
    width: int
    height: int
    duration: float
    has_audio: bool
    fps: float


def probe_media(path: Path, ffprobe_binary: str = "ffprobe") -> MediaInfo:
    result = subprocess.run([ffprobe_binary, "-v", "error", "-show_entries",
        "stream=codec_type,width,height,avg_frame_rate:format=duration", "-of", "json", str(path)],
        check=True, capture_output=True, text=True, timeout=30)
    data = json.loads(result.stdout)
    video = next((stream for stream in data["streams"] if stream["codec_type"] == "video"), None)
    if video is None:
        raise ValueError(f"source has no video: {path}")
    numerator, denominator = map(float, video.get("avg_frame_rate", "0/1").split("/"))
    return MediaInfo(int(video["width"]), int(video["height"]), float(data["format"]["duration"]),
                     any(stream["codec_type"] == "audio" for stream in data["streams"]),
                     numerator / denominator if denominator else 0)


def extract_audio(source: Path, output: Path, start: float, duration: float) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        *_seek_input(source, start), "-t", f"{duration:.6f}", "-map", "0:a:0", "-vn", "-ac", "1",
        "-ar", "16000", "-c:a", "pcm_s16le", str(output)], check=True, capture_output=True, timeout=120)


def stage_fonts(directory: Path, destination: Path) -> str:
    destination.mkdir()
    if directory.is_dir():
        for path in directory.iterdir():
            if path.is_file() and path.suffix.lower() in (".ttf", ".otf"):
                shutil.copyfile(path, destination / path.name)
    return font_family(destination)


def render_original(source: Path, output: Path, start: float, duration: float,
                    info: MediaInfo, words: list[Word], config: RenderConfig,
                    work: Path, selector: EncoderSelector) -> tuple[LayoutGraph, str, str]:
    """Render into a temporary MP4; callers validate and atomically replace it."""
    source, output, work = source.resolve(), output.resolve(), work.resolve()
    family = stage_fonts(config.fonts_dir, work / "fonts")
    captions = bool(words) and config.captions_enabled
    if captions:
        (work / "captions.ass").write_text(generate_ass(words, config, family), encoding="utf-8")
    layout = build_layout(config, info.width, info.height, captions)
    encoder = selector.select(source, start)
    codec = ["-c:v", encoder]
    codec += ["-preset", config.preset, "-crf", str(config.crf)] if encoder == "libx264" else ["-preset", "p4", "-cq", str(config.crf)]
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *_seek_input(source, start)]
    if not info.has_audio:
        command += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
    command += ["-t", f"{duration:.6f}", "-filter_complex", layout.graph,
                "-map", "[video]", "-map", "0:a:0" if info.has_audio else "1:a:0",
                *codec, "-pix_fmt", "yuv420p", "-af", f"loudnorm=I={config.target_lufs}:TP=-1.5:LRA=11",
                "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart", str(output)]
    subprocess.run(command, cwd=work, check=True, capture_output=True, timeout=max(300, duration * 15))
    return layout, family, encoder


def validate_export(path: Path, expected_duration: float) -> MediaInfo:
    info = probe_media(path)
    if (info.width, info.height) != (1080, 1920) or abs(info.fps - 30) > 0.01 or not info.has_audio:
        raise ValueError("export must be 1080x1920 at 30fps with audio")
    if abs(info.duration - expected_duration) > 0.15:
        raise ValueError(f"export duration {info.duration:.3f}s differs from requested {expected_duration:.3f}s")
    return info
