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

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Word:
    start: float
    end: float
    text: str
    probability: float = 1.0


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
            word_timestamps=True, vad_filter=True, condition_on_previous_text=False,
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
                words.append(Word(start, end, word.text.strip(), word.probability))
    return words, rejected


def transcript_key(source: Path, start: float, end: float, config: RenderConfig) -> str:
    stat = source.stat()
    settings = dict(schema=1, source=str(source.resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                    start=start, end=end, model=config.whisper_model, language=config.language,
                    device=config.device, compute_type=config.compute_type,
                    faster_whisper=version("faster-whisper"), ctranslate2=version("ctranslate2"),
                    word_timestamps=True, vad_filter=True, condition_on_previous_text=False,
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
                                           float(word["probability"])) for word in segment["words"]))
                                for segment in data["transcript"]["segments"]),
                          data["transcript"].get("language"), data["transcript"].get("engine", "faster-whisper"))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        LOGGER.warning("Ignoring invalid transcript cache %s: %s", path, exc)
        return None


def save_transcript(path: Path, key: str, transcript: Transcript) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, dict(schema=1, key=key, transcript=asdict(transcript)))
