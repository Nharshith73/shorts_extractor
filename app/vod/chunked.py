"""Bounded media decoding with continuous bins and feature state."""

from contextlib import contextmanager
import logging
import math
from pathlib import Path
import subprocess
import tempfile
import time
from typing import BinaryIO, Iterator

import numpy as np
from scipy import signal

from app.detection.config import DetectionConfig
from app.detection.features import FRAME_HEIGHT, FRAME_WIDTH, _db, _spectral_flux, bin_layout
from app.detection.models import FeatureBin
from .timeline import media_time

LOGGER = logging.getLogger(__name__)


@contextmanager
def decoder(command: list[str]) -> Iterator[BinaryIO]:
    # stderr goes to disk, avoiding a filled stderr pipe blocking a long decode.
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
        assert process.stdout is not None
        try:
            yield process.stdout
            code = process.wait(timeout=30)
            if code:
                errors.seek(0)
                raise subprocess.CalledProcessError(code, command, stderr=errors.read())
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)


def read_block(stream: BinaryIO, size: int) -> bytes:
    blocks: list[bytes] = []
    count = 0
    while count < size:
        block = stream.read(size - count)
        if not block:
            break
        blocks.append(block)
        count += len(block)
    return b"".join(blocks)


class AudioBins:
    """Carry SOS filter state and unfinished one-second/FFT input bins."""

    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self.sos = signal.butter(4, (300, 3400), btype="bandpass", fs=sample_rate, output="sos")
        self.zi = np.zeros((self.sos.shape[0], 2))
        self.pending = np.empty(0, dtype=np.float32)
        self.voice = np.empty(0, dtype=np.float32)
        self.result: list[dict[str, float]] = []

    def feed(self, samples: np.ndarray) -> None:
        filtered, self.zi = signal.sosfilt(self.sos, samples, zi=self.zi)
        self.pending = np.concatenate((self.pending, samples))
        self.voice = np.concatenate((self.voice, filtered.astype(np.float32)))
        full = self.pending.size // self.sample_rate
        for index in range(full):
            start, end = index * self.sample_rate, (index + 1) * self.sample_rate
            self._emit(self.pending[start:end], self.voice[start:end])
        consumed = full * self.sample_rate
        self.pending = self.pending[consumed:].copy()
        self.voice = self.voice[consumed:].copy()

    def _emit(self, samples: np.ndarray, voice: np.ndarray) -> None:
        rms = float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0
        band = float(np.sqrt(np.mean(voice * voice))) if voice.size else 0.0
        self.result.append(dict(audio_rms_db=_db(rms), voice_energy_db=_db(band), onset_strength=_spectral_flux(samples)))

    def finish(self) -> list[dict[str, float]]:
        if self.pending.size:
            self._emit(self.pending, self.voice)
            self.pending = self.pending[:0]
            self.voice = self.voice[:0]
        return self.result


class VideoBins:
    def __init__(self, count: int, fps: float) -> None:
        self.fps = fps
        self.previous: np.ndarray | None = None
        self.index = 0
        self.motion = np.zeros(count)
        self.cuts = np.zeros(count)
        self.counts = np.zeros(count, dtype=int)

    def feed(self, frames: np.ndarray) -> None:
        for frame in frames:
            bin_index = min(len(self.motion) - 1, int(self.index / self.fps))
            if self.previous is not None and bin_index >= 0:
                self.motion[bin_index] += float(np.mean(np.abs(frame.astype(np.float32) - self.previous.astype(np.float32))))
                histogram = np.bincount(frame.ravel() // 16, minlength=16) / frame.size
                old_histogram = np.bincount(self.previous.ravel() // 16, minlength=16) / self.previous.size
                self.cuts[bin_index] += float(0.5 * np.abs(histogram - old_histogram).sum())
                self.counts[bin_index] += 1
            self.previous = frame.copy()
            self.index += 1

    def finish(self) -> list[dict[str, float]]:
        return [dict(motion=float(motion / count) if count else 0.0, cut_change=float(cut / count) if count else 0.0)
                for motion, cut, count in zip(self.motion, self.cuts, self.counts)]


def extract_chunked(video: Path, config: DetectionConfig, source_id: str, start: float, end: float,
                    chunk_seconds: float = 300, progress_seconds: float = 15,
                    ffmpeg_binary: str = "ffmpeg") -> tuple[list[FeatureBin], dict[str, float]]:
    duration = end - start
    if duration <= 0 or not math.isfinite(chunk_seconds) or chunk_seconds <= 0:
        raise ValueError("positive duration and chunk_seconds required")
    layout = bin_layout(duration)
    # An explicit input seek to zero changes AAC priming on some FFmpeg builds.
    # Omit it for an untrimmed file so the existing decoder's sample grid holds.
    seek = ["-ss", str(start)] if start > 0 else []
    base = [ffmpeg_binary, "-hide_banner", "-loglevel", "error", "-nostdin", *seek, "-i", str(video), "-t", str(duration)]
    timings = dict(audio_decode_seconds=0.0, audio_features_seconds=0.0, video_decode_seconds=0.0, video_features_seconds=0.0)
    begun, last_log = time.perf_counter(), 0.0
    LOGGER.info("Decoding VOD %.1fs .. %.1fs in bounded %.1fs chunks", start, end, chunk_seconds)
    def progress(stage: str, fraction: float) -> None:
        nonlocal last_log
        elapsed = time.perf_counter() - begun
        if elapsed - last_log >= progress_seconds or fraction >= 1:
            completed = duration * ((0.5 if stage == "video" else 0) + 0.5 * min(fraction, 1))
            LOGGER.info("Decode %s %.1f%%; elapsed %.1fs; effective %.2fx", stage, 100 * completed / duration, elapsed, completed / max(elapsed, 1e-9))
            last_log = elapsed
    audio = AudioBins(config.sample_rate)
    samples_read = 0
    with decoder([*base, "-vn", "-ac", "1", "-ar", str(config.sample_rate), "-f", "f32le", "pipe:1"]) as stream:
        while True:
            at = time.perf_counter()
            raw = read_block(stream, max(1, int(chunk_seconds * config.sample_rate)) * 4)
            timings["audio_decode_seconds"] += time.perf_counter() - at
            if not raw:
                break
            at = time.perf_counter()
            samples = np.frombuffer(raw, dtype="<f4")
            # Ignore codec padding beyond the requested analysis end.
            remaining = max(0, int(round(duration * config.sample_rate)) - samples_read)
            samples = samples[:remaining]
            if samples.size:
                audio.feed(samples)
            samples_read += samples.size
            timings["audio_features_seconds"] += time.perf_counter() - at
            progress("audio", samples_read / (duration * config.sample_rate))
    audio_rows = audio.finish()
    video_bins = VideoBins(len(layout), config.video_fps)
    frame_size = FRAME_WIDTH * FRAME_HEIGHT
    with decoder([*base, "-an", "-vf", f"fps={config.video_fps},scale={FRAME_WIDTH}:{FRAME_HEIGHT}:flags=bilinear,format=gray",
                  "-pix_fmt", "gray", "-f", "rawvideo", "pipe:1"]) as stream:
        while True:
            at = time.perf_counter()
            raw = read_block(stream, max(1, int(chunk_seconds * config.video_fps)) * frame_size)
            timings["video_decode_seconds"] += time.perf_counter() - at
            if not raw:
                break
            at = time.perf_counter()
            if len(raw) % frame_size:
                raise ValueError("decoder returned an incomplete video frame")
            video_bins.feed(np.frombuffer(raw, dtype=np.uint8).reshape(-1, FRAME_HEIGHT, FRAME_WIDTH))
            timings["video_features_seconds"] += time.perf_counter() - at
            progress("video", video_bins.index / (duration * config.video_fps))
    video_rows = video_bins.finish()
    silent = dict(audio_rms_db=-160.0, voice_energy_db=-160.0, onset_strength=0.0)
    features = [FeatureBin(source_id, "vod", media_time(start + offset), seconds,
                           **(audio_rows[index] if index < len(audio_rows) else silent), **video_rows[index])
                for index, (offset, seconds) in enumerate(layout)]
    return features, timings
