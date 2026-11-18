"""Live stream video recorder using streamlink and ffmpeg."""

from __future__ import annotations

import logging
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any

LOGGER = logging.getLogger(__name__)


class StreamlinkRecorder:
    """Records live streams to an append-safe MPEG-TS video file using streamlink."""

    def __init__(
        self,
        stream_url: str,
        output_dir: Path,
        quality: str = "best",
        filename: str = "live_stream.ts",
        platform: str = "twitch",
    ) -> None:
        self.raw_input = stream_url.strip()
        self.platform = platform.lower()
        if "kick.com" in self.raw_input.lower():
            self.platform = "kick"

        if self.raw_input.startswith("http://") or self.raw_input.startswith("https://"):
            self.stream_url = self.raw_input
        else:
            if self.platform == "kick":
                self.stream_url = f"https://kick.com/{self.raw_input}"
            else:
                self.stream_url = f"https://twitch.tv/{self.raw_input}"

        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.output_dir / filename
        self.quality = quality

        self._process: subprocess.Popen[str] | None = None
        self._start_time: float | None = None
        self._lock = threading.Lock()
        self._error_message: str | None = None
        self._log_fh = None

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return bool(self._process and self._process.poll() is None)

    @property
    def elapsed_seconds(self) -> float:
        if self._start_time is None:
            return 0.0
        return round(time.time() - self._start_time, 2)

    @property
    def error_message(self) -> str | None:
        return self._error_message

    def start(self) -> None:
        """Launch streamlink in background recording process."""
        with self._lock:
            if self._process and self._process.poll() is None:
                LOGGER.info("Streamlink is already actively recording")
                return

            # Clean any old recording file
            if self.output_file.is_file():
                try:
                    self.output_file.unlink()
                except Exception:
                    pass

            cmd = [
                sys.executable,
                "-m",
                "streamlink",
            ]
            if self.platform == "twitch":
                cmd.append("--twitch-disable-ads")

            cmd.extend([
                self.stream_url,
                self.quality,
                "-o",
                str(self.output_file),
                "--loglevel",
                "info",
            ])

            log_file = self.output_dir / "streamlink.log"
            self._log_fh = open(log_file, "w", encoding="utf-8")

            LOGGER.info("Starting streamlink: %s", " ".join(cmd))
            try:
                self._process = subprocess.Popen(
                    cmd,
                    stdout=self._log_fh,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                self._start_time = time.time()
                self._error_message = None
            except Exception as exc:
                self._error_message = str(exc)
                if self._log_fh:
                    try: self._log_fh.close()
                    except Exception: pass
                    self._log_fh = None
                LOGGER.error("Failed to start streamlink: %s", exc)
                raise

    def stop(self) -> None:
        """Stop streamlink process gracefully."""
        with self._lock:
            if self._process:
                LOGGER.info("Stopping streamlink recording...")
                try:
                    self._process.terminate()
                    self._process.wait(timeout=5.0)
                except Exception:
                    try:
                        self._process.kill()
                    except Exception:
                        pass
                self._process = None

            if self._log_fh:
                try:
                    self._log_fh.close()
                except Exception:
                    pass
                self._log_fh = None

    def poll_status(self) -> dict[str, Any]:
        """Check process health and recorded file stats."""
        alive = self.is_recording
        file_size_mb = 0.0
        if self.output_file.is_file():
            try:
                file_size_mb = round(self.output_file.stat().st_size / (1024 * 1024), 2)
            except Exception:
                pass

        if not alive and self._process and self._process.poll() is not None:
            code = self._process.poll()
            if code != 0 and not self._error_message:
                log_file = self.output_dir / "streamlink.log"
                if log_file.is_file():
                    try:
                        lines = log_file.read_text(encoding="utf-8", errors="ignore").splitlines()
                        err_tail = [l for l in lines if "error" in l.lower() or "warning" in l.lower()]
                        self._error_message = " ".join(err_tail[-3:]) if err_tail else f"Streamlink exited with code {code}"
                    except Exception:
                        self._error_message = f"Streamlink exited with code {code}"
                else:
                    self._error_message = f"Streamlink exited with code {code}"

        return {
            "alive": alive,
            "elapsed_seconds": self.elapsed_seconds,
            "file_size_mb": file_size_mb,
            "output_file": str(self.output_file),
            "error": self._error_message,
        }

    def get_duration(self) -> float:
        """Probe recorded video duration in seconds using ffprobe."""
        if not self.output_file.is_file() or self.output_file.stat().st_size == 0:
            return 0.0
        try:
            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration:stream=duration",
                    "-of",
                    "csv=p=0",
                    str(self.output_file),
                ],
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            for val in probe.stdout.splitlines():
                try:
                    dur = float(val.strip(",").strip())
                    if dur > 0.1:
                        return dur
                except ValueError:
                    continue
        except Exception:
            pass
        return self.elapsed_seconds

    def extract_clip(
        self,
        start_seconds: float,
        end_seconds: float,
        out_path: Path,
        safe_pad: float = 0.0,
    ) -> Path:
        """Extract a subclip from the recording using FFmpeg fast seek."""
        if not self.output_file.is_file():
            raise FileNotFoundError(f"Recording file not found: {self.output_file}")

        out_path = Path(out_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        clip_start = max(0.0, start_seconds - safe_pad)
        duration = max(1.0, end_seconds - clip_start + safe_pad)

        # Fast seek with PTS normalization: resets both audio and video presentation
        # timestamps to 0.000s, ensuring precision sync while extracting in < 2 seconds
        # even on multi-hour transport stream recordings (.ts).
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{clip_start:.2f}",
            "-i",
            str(self.output_file),
            "-t",
            f"{duration:.2f}",
            "-vf",
            "setpts=PTS-STARTPTS",
            "-af",
            "asetpts=PTS-STARTPTS",
            "-avoid_negative_ts",
            "make_zero",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "20",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            str(out_path),
        ]

        res = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        if res.returncode != 0:
            raise RuntimeError(f"FFmpeg clip extraction failed: {res.stderr[:300]}")

        return out_path
