"""Lazy clip-only speech recognition and style-independent transcript caching."""

from dataclasses import asdict, dataclass
from importlib.metadata import version
import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any, Iterable, Protocol

from app.vod.artifacts import write_json
from .config import RenderConfig
from .audio import AudioFrame, AUDIO_PIPELINE_VERSION
import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Word:
    start: float
    end: float
    text: str
    probability: float = 1.0
    kind: str = "speech"


@dataclass(frozen=True, slots=True)
class SpeechSegment:
    start: float
    end: float
    text: str
    avg_logprob: float
    no_speech_prob: float
    words: tuple[Word, ...]


@dataclass(frozen=True, slots=True)
class Transcript:
    segments: tuple[SpeechSegment, ...]
    language: str | None = None
    engine: str = "faster-whisper"
    audio_frames: tuple[AudioFrame, ...] = ()


class Transcriber(Protocol):
    def transcribe(self, audio: Path) -> Transcript: ...


def convert_segments(segments: Iterable[Any], language: str | None) -> Transcript:
    converted = tuple(SpeechSegment(float(segment.start), float(segment.end), str(segment.text),
        float(segment.avg_logprob), float(segment.no_speech_prob),
        tuple(Word(float(word.start), float(word.end), str(word.word).strip(), float(word.probability))
              for word in (segment.words or ()))) for segment in segments)
    return Transcript(converted, language)


class WhisperTranscriber:
    def __init__(self, config: RenderConfig) -> None:
        self.config = config
        self.model: Any = None

    def transcribe(self, audio: Path) -> Transcript:
        if self.model is None:
            from faster_whisper import WhisperModel
            LOGGER.info("Loading Whisper %s on %s (%s)", self.config.whisper_model, self.config.device, self.config.compute_type)
            self.model = WhisperModel(self.config.whisper_model, device=self.config.device, compute_type=self.config.compute_type)
        segments, info = self.model.transcribe(str(audio),
            language=None if self.config.language == "auto" else self.config.language,
            word_timestamps=True, vad_filter=self.config.vad_filter, condition_on_previous_text=False,
            no_speech_threshold=None, log_prob_threshold=None)
        return convert_segments(segments, info.language)


def usable_words(transcript: Transcript, config: RenderConfig, duration: float) -> tuple[list[Word], int]:
    """Drop suspect segments and invalid words; clamp clip-relative word times."""
    words: list[Word] = []
    rejected = 0
    for segment in sorted(transcript.segments, key=lambda item: item.start):
        if (not all(math.isfinite(value) for value in (segment.avg_logprob, segment.no_speech_prob))
            or segment.avg_logprob < config.min_avg_logprob or segment.no_speech_prob > config.max_no_speech_prob):
            rejected += 1
            continue
        for word in sorted(segment.words, key=lambda item: item.start):
            if not word.text.strip() or not all(math.isfinite(value) for value in (word.start, word.end, word.probability)):
                continue
            start = max(0.0, word.start, words[-1].end if words else 0.0)
            end = min(duration, word.end)
            if end > start and round(end * 100) > round(start * 100):
                words.append(Word(start, end, word.text.strip(), word.probability, word.kind))
    return words, rejected


def add_reaction_tags(transcript: Transcript, words: list[Word], config: RenderConfig) -> tuple[list[Word], list[dict]]:
    """Tag uncovered vocal-band bursts, without pretending to recognize screamed words."""
    frames = transcript.audio_frames
    if not config.reaction_fallback or not frames:
        return words, []
    floor = float(np.percentile([frame.rms_db for frame in frames], 20))
    groups: list[list[AudioFrame]] = []
    for index, frame in enumerate(frames):
        local = float(np.median([item.rms_db for item in frames[max(0, index-50):index]])) if index else floor
        contrast = frame.rms_db-min(local, floor)
        covered = any(word.start < frame.end and word.end > frame.start for word in words)
        high = (frame.rms_db >= config.reaction_min_db and contrast >= config.reaction_contrast_db
                and frame.voice_ratio >= config.reaction_voice_ratio and not covered)
        if high:
            if groups and abs(groups[-1][-1].end-frame.start) < 0.001:
                groups[-1].append(frame)
            else:
                groups.append([frame])
    tags, details = [], []
    for group in groups:
        start, end = group[0].start, group[-1].end
        if end-start < 0.1-1e-6:
            continue
        while start < end-0.01:
            stop = min(end, start+3)
            tags.append(Word(start, stop, config.reaction_tag, 0.0, "reaction"))
            details.append(dict(start=start, end=stop, tag=config.reaction_tag, reason="untranscribed_vocal_band_burst",
                                heuristic=True, peak_db=max(frame.rms_db for frame in group)))
            start = stop
    return sorted([*words, *tags], key=lambda word: word.start), details


def transcript_key(source: Path, start: float, end: float, config: RenderConfig) -> str:
    stat = source.stat()
    settings = dict(schema=1, source=str(source.resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                    start=start, end=end, model=config.whisper_model, language=config.language,
                    device=config.device, compute_type=config.compute_type,
                    faster_whisper=version("faster-whisper"), ctranslate2=version("ctranslate2"),
                    word_timestamps=True, vad_filter=config.vad_filter, condition_on_previous_text=False,
                    audio_pipeline=AUDIO_PIPELINE_VERSION,
                    no_speech_threshold=None, log_prob_threshold=None)
    # Keep cache filenames short under Windows' conventional path-length limit.
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:24]


def read_transcript(path: Path, key: str) -> Transcript | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["key"] != key or data["schema"] != 1:
            return None
        return Transcript(tuple(SpeechSegment(float(segment["start"]), float(segment["end"]), str(segment["text"]),
                                float(segment["avg_logprob"]), float(segment["no_speech_prob"]),
                                tuple(Word(float(word["start"]), float(word["end"]), str(word["text"]),
                                           float(word["probability"]), word.get("kind", "speech")) for word in segment["words"]))
                                for segment in data["transcript"]["segments"]),
                          data["transcript"].get("language"), data["transcript"].get("engine", "faster-whisper"),
                          tuple(AudioFrame(**frame) for frame in data["transcript"].get("audio_frames", ())))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        LOGGER.warning("Ignoring invalid transcript cache %s: %s", path, exc)
        return None


def save_transcript(path: Path, key: str, transcript: Transcript) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, dict(schema=1, key=key, transcript=asdict(transcript)))
