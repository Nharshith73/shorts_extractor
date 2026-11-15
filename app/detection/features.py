"""FFmpeg-pipe decoding and inexpensive per-second audio/video features."""

from __future__ import annotations

import logging
import math
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import signal

from .config import DetectionConfig
from .models import FeatureBin

LOGGER = logging.getLogger(__name__)

FEATURE_NAMES = (
    "audio_rms_db",
    "voice_energy_db",
    "onset_strength",
    "motion",
    "cut_change",
)
FRAME_WIDTH = 160
FRAME_HEIGHT = 90


def bin_layout(duration_seconds: float) -> list[tuple[float, float]]:
    """Return ``(offset, duration)`` bins, retaining a final partial bin."""
    if duration_seconds <= 0:
        return []
    full_bins = int(math.floor(duration_seconds))
    bins = [(float(index), 1.0) for index in range(full_bins)]
    partial = duration_seconds - full_bins
    if partial > 1e-6:
        bins.append((float(full_bins), partial))
    return bins


def _run_ffmpeg(command: Sequence[str]) -> bytes:
    result = subprocess.run(
        list(command),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
    )
    return result.stdout


def decode_audio(path: Path, sample_rate: int, ffmpeg_binary: str = "ffmpeg") -> np.ndarray:
    """Decode audio to mono float32 samples through FFmpeg stdout."""
    raw = _run_ffmpeg(
        [
            ffmpeg_binary,
            "-v",
            "error",
            "-i",
            str(path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-f",
            "f32le",
            "pipe:1",
        ]
    )
    return np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=False)


def decode_video_gray(
    path: Path, video_fps: float, ffmpeg_binary: str = "ffmpeg"
) -> np.ndarray:
    """Decode 160×90 gray frames at a low fixed FPS through FFmpeg stdout."""
    raw = _run_ffmpeg(
        [
            ffmpeg_binary,
            "-v",
            "error",
            "-i",
            str(path),
            "-an",
            "-vf",
            f"fps={video_fps},scale={FRAME_WIDTH}:{FRAME_HEIGHT}:flags=bilinear,format=gray",
            "-pix_fmt",
            "gray",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
    )
    frame_size = FRAME_WIDTH * FRAME_HEIGHT
    frame_count = len(raw) // frame_size
    if frame_count == 0:
        return np.empty((0, FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)
    usable = memoryview(raw)[: frame_count * frame_size]
    return np.frombuffer(usable, dtype=np.uint8).reshape(frame_count, FRAME_HEIGHT, FRAME_WIDTH)


def _db(value: float) -> float:
    return float(20.0 * np.log10(max(value, 1e-8)))


def _spectral_flux(samples: np.ndarray) -> float:
    """Mean positive frame-to-frame FFT magnitude change for one bin."""
    frame_size = 512
    hop = 256
    if samples.size == 0:
        return 0.0
    if samples.size < frame_size:
        samples = np.pad(samples, (0, frame_size - samples.size))
    frame_count = 1 + (samples.size - frame_size) // hop
    starts = np.arange(frame_count) * hop
    frames = np.stack([samples[start : start + frame_size] for start in starts])
    magnitudes = np.abs(np.fft.rfft(frames * np.hanning(frame_size), axis=1))
    if magnitudes.shape[0] < 2:
        return 0.0
    flux = np.maximum(np.diff(magnitudes, axis=0), 0.0)
    return float(np.mean(flux))


def audio_features_by_bin(
    samples: np.ndarray, layout: Sequence[tuple[float, float]], sample_rate: int
) -> list[dict[str, float]]:
    """Measure RMS, voice-band RMS, and onset strength for every bin."""
    if samples.size:
        sos = signal.butter(4, (300, 3400), btype="bandpass", fs=sample_rate, output="sos")
        voice_samples = signal.sosfilt(sos, samples).astype(np.float32, copy=False)
    else:
        voice_samples = samples
    result: list[dict[str, float]] = []
    for offset, bin_seconds in layout:
        start = int(round(offset * sample_rate))
        end = min(samples.size, int(round((offset + bin_seconds) * sample_rate)))
        audio_bin = samples[start:end]
        voice_bin = voice_samples[start:end]
        rms = float(np.sqrt(np.mean(np.square(audio_bin)))) if audio_bin.size else 0.0
        voice_rms = float(np.sqrt(np.mean(np.square(voice_bin)))) if voice_bin.size else 0.0
        result.append(
            {
                "audio_rms_db": _db(rms),
                "voice_energy_db": _db(voice_rms),
                "onset_strength": _spectral_flux(audio_bin),
            }
        )
    return result


def video_features_by_bin(
    frames: np.ndarray, layout: Sequence[tuple[float, float]], video_fps: float
) -> list[dict[str, float]]:
    """Measure frame difference and 16-bin histogram change for every bin."""
    motion_values: list[list[float]] = [[] for _ in layout]
    cut_values: list[list[float]] = [[] for _ in layout]
    previous: np.ndarray | None = None
    for index, frame in enumerate(frames):
        bin_index = min(len(layout) - 1, int((index / video_fps) // 1)) if layout else 0
        if previous is not None and layout:
            motion_values[bin_index].append(
                float(np.mean(np.abs(frame.astype(np.float32) - previous.astype(np.float32))))
            )
            histogram = np.bincount((frame.ravel() // 16), minlength=16) / frame.size
            previous_histogram = np.bincount((previous.ravel() // 16), minlength=16) / previous.size
            cut_values[bin_index].append(float(0.5 * np.abs(histogram - previous_histogram).sum()))
        previous = frame
    return [
        {
            "motion": float(np.mean(motion_values[index])) if motion_values[index] else 0.0,
            "cut_change": float(np.mean(cut_values[index])) if cut_values[index] else 0.0,
        }
        for index in range(len(layout))
    ]


def extract_segment_features(
    path: Path,
    *,
    segment_id: int | None,
    source_id: str,
    session_id: str,
    wall_start_utc: datetime,
    duration_seconds: float,
    config: DetectionConfig,
    ffmpeg_binary: str = "ffmpeg",
) -> list[FeatureBin]:
    """Decode a complete segment and return persisted-ready raw feature bins."""
    layout = bin_layout(duration_seconds)
    if not layout:
        return []
    samples = decode_audio(path, config.sample_rate, ffmpeg_binary)
    frames = decode_video_gray(path, config.video_fps, ffmpeg_binary)
    audio = audio_features_by_bin(samples, layout, config.sample_rate)
    video = video_features_by_bin(frames, layout, config.video_fps)
    start = wall_start_utc.astimezone(UTC)
    return [
        FeatureBin(
            segment_id=segment_id,
            source_id=source_id,
            session_id=session_id,
            wall_utc=start + timedelta(seconds=offset),
            bin_seconds=bin_seconds,
            **audio[index],
            **video[index],
        )
        for index, (offset, bin_seconds) in enumerate(layout)
    ]
