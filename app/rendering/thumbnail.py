"""Automated 9:16 Vertical Thumbnail Generator for YouTube Shorts & Reels."""

from __future__ import annotations

import logging
from pathlib import Path
import re
import subprocess

LOGGER = logging.getLogger(__name__)


def generate_thumbnail(
    video_path: Path | str,
    output_path: Path | str,
    hook_text: str = "WAIT TILL THE END 💀",
    badge_text: str = "GTA VI",
    timestamp_seconds: float = 2.0,
    ffmpeg_binary: str = "ffmpeg",
) -> Path | None:
    """Generate an eye-catching, high-contrast 1080x1920 vertical thumbnail for YouTube Shorts.

    Applies contrast and vibrance grading, a high-CTR corner badge, and a bold yellow hook header.
    """
    vpath = Path(video_path).resolve()
    out_path = Path(output_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not vpath.is_file():
        LOGGER.warning("Video file not found for thumbnail: %s", vpath)
        return None

    clean_hook = re.sub(r"[^\w\s.,!?\'\"-]", "", hook_text.strip()).strip().upper()
    if not clean_hook:
        clean_hook = "INSANE MOMENT"
    if len(clean_hook) > 28:
        clean_hook = clean_hook[:28].rsplit(" ", 1)[0].strip()

    clean_badge = re.sub(r"[^\w\s]", "", badge_text.strip()).strip().upper() or "GTA VI"

    ass_temp = out_path.with_suffix(".thumb_temp.ass")
    ass_content = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Title,Arial Black,70,&H0000FFFF,&H00000000,&H00000000,&HB0000000,-1,0,0,0,100,100,0,0,1,6,4,8,40,40,160,1
Style: Badge,Arial Black,36,&H00FFFFFF,&H00000000,&H000000E0,&H00000000,-1,0,0,0,100,100,0,0,3,4,0,7,50,50,70,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 1,0:00:00.00,0:00:10.00,Badge,,0,0,0,,{{\\b1}}{clean_badge}
Dialogue: 2,0:00:00.00,0:00:10.00,Title,,0,0,0,,{{\\b1}}{clean_hook}
"""
    try:
        ass_temp.write_text(ass_content, encoding="utf-8")
        escaped_ass = str(ass_temp).replace("\\", "/").replace(":", "\\:")

        filter_chain = (
            "scale=1080:1920:force_original_aspect_ratio=increase,"
            "crop=1080:1920:(iw-1080)/2:(ih-1920)/2,"
            "eq=contrast=1.18:saturation=1.25,"
            f"subtitles='{escaped_ass}'"
        )

        cmd = [
            ffmpeg_binary,
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-ss", f"{timestamp_seconds:.2f}",
            "-i", str(vpath),
            "-vf", filter_chain,
            "-frames:v", "1",
            "-q:v", "2",
            str(out_path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0 and out_path.is_file():
            LOGGER.info("Generated Short thumbnail: %s", out_path.name)
            return out_path
        LOGGER.warning("Thumbnail generation failed (code %d): %s", res.returncode, res.stderr[-300:] if res.stderr else "")
    except Exception as exc:
        LOGGER.warning("Exception during thumbnail generation: %s", exc)
    finally:
        ass_temp.unlink(missing_ok=True)

    return None
