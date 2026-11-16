"""Shared frame-accurate cut command for buffered and direct-file sources."""

from pathlib import Path
import logging
import subprocess

LOGGER = logging.getLogger(__name__)


def _text(value: str | bytes | None) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value or ""


def _seek_input(source: Path, start_seconds: float) -> list[str]:
    # Unindexed transport streams can seek forward to the next keyframe and
    # omit video altogether on a short cut. Buffered TS joins are small, so
    # retain decode/discard seeking for those inputs.
    seek = ["-ss", f"{start_seconds:.6f}"]
    if source.suffix.lower() in (".ts", ".mts", ".m2ts", ".mpegts"):
        return ["-i", str(source), *seek]
    return [*seek, "-accurate_seek", "-i", str(source)]


class EncoderSelector:
    """Cache a real encode probe for one worker/run, never an encoder-list lookup."""

    def __init__(self, requested: str = "libx264", ffmpeg_binary: str = "ffmpeg") -> None:
        if requested not in ("auto", "libx264", "h264_nvenc"):
            raise ValueError(f"unsupported encoder: {requested}")
        self.requested = requested
        self.ffmpeg_binary = ffmpeg_binary
        self.selected: str | None = None

    def select(self, source: Path, start_seconds: float) -> str:
        if self.selected is not None:
            return self.selected
        if self.requested != "auto":
            self.selected = self.requested
            return self.selected
        try:
            # Match the actual source's resolution/pixel conversion. The null
            # muxer discards encoded packets; NVENC still encodes real frames.
            result = subprocess.run([
                self.ffmpeg_binary, "-hide_banner", "-loglevel", "error", "-nostdin",
                "-progress", "pipe:1", "-nostats", *_seek_input(source, start_seconds),
                "-t", "1", "-map", "0:v:0", "-an",
                "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "18",
                "-pix_fmt", "yuv420p", "-f", "null", "-",
            ], check=True, capture_output=True, timeout=30)
            progress = dict(line.split("=", 1) for line in _text(result.stdout).splitlines() if "=" in line)
            if int(progress.get("frame", "0")) <= 0 or int(progress.get("out_time_us", "0")) < 900_000:
                raise ValueError("NVENC probe did not encode a full second of video")
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            details = _text(getattr(exc, "stderr", None)).strip() or str(exc)
            LOGGER.warning("NVENC one-second encode probe failed; falling back to libx264. Cause: %s", details)
            self.selected = "libx264"
        else:
            LOGGER.info("NVENC one-second encode probe succeeded; using h264_nvenc")
            self.selected = "h264_nvenc"
        return self.selected


def render_cut(source: Path, output: Path, start_seconds: float, duration_seconds: float,
               *, video_encoder: str = "libx264", ffmpeg_binary: str = "ffmpeg",
               encoder_selector: EncoderSelector | None = None) -> None:
    video_encoder = (encoder_selector or EncoderSelector(video_encoder, ffmpeg_binary)).select(source, start_seconds)
    codec = ["-c:v", video_encoder]
    codec += ["-preset", "veryfast", "-crf", "18"] if video_encoder == "libx264" else ["-preset", "p4", "-cq", "18"]
    # Input-side accurate seeking avoids decoding the entire lead-in. This
    # remains a re-encode; FFmpeg discards preroll before the requested time.
    subprocess.run([ffmpeg_binary, "-hide_banner", "-loglevel", "error", "-y",
                    *_seek_input(source, start_seconds),
                    "-t", f"{duration_seconds:.6f}",
                    "-map", "0:v:0", "-map", "0:a:0?", *codec, "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-movflags", "+faststart", str(output)],
                   check=True, capture_output=True, timeout=300)
