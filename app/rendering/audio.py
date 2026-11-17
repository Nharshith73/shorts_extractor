"""Clip-local audio preparation and conservative, inspectable reaction fallbacks."""

from dataclasses import dataclass
from pathlib import Path
import wave

import numpy as np
from scipy import signal

ASR_AUDIO_FILTER = ("highpass=f=80,acompressor=threshold=0.25:ratio=4:attack=5:release=80,"
                    "alimiter=limit=0.95:latency=1,loudnorm=I=-18:TP=-2:LRA=7")
AUDIO_PIPELINE_VERSION = 3


@dataclass(frozen=True, slots=True)
class AudioFrame:
    start: float
    end: float
    rms_db: float
    voice_ratio: float


def reaction_frames(path: Path) -> tuple[AudioFrame, ...]:
    with wave.open(str(path)) as audio:
        if audio.getnchannels() != 1 or audio.getsampwidth() != 2:
            raise ValueError("reaction analysis requires mono PCM16 audio")
        rate = audio.getframerate()
        samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").astype(np.float32)/32768
    if not samples.size:
        return ()
    band = signal.sosfilt(signal.butter(4, (300, 3000), btype="bandpass", fs=rate, output="sos"), samples)
    hop = max(1, round(rate/10))
    frames = []
    for offset in range(0, len(samples), hop):
        raw = samples[offset:offset+hop]
        voice = band[offset:offset+hop]
        rms = float(np.sqrt(np.mean(raw*raw)))
        ratio = min(1.0, float(np.sqrt(np.mean(voice*voice)))/max(rms, 1e-8))
        frames.append(AudioFrame(offset/rate, min(len(samples), offset+hop)/rate,
                                 float(20*np.log10(max(rms, 1e-8))), ratio))
    return tuple(frames)
