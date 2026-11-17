"""Data models and type definitions for Stage 4A gameplay intelligence & evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import math
from typing import Any

STANDARD_CATEGORIES: tuple[str, ...] = (
    "funny",
    "unexpected",
    "impressive",
    "dramatic",
    "chaotic",
    "skillful",
    "interaction",
    "fail",
    "story_moment",
    "ordinary",
    "unknown",
)


def utc_iso_now() -> str:
    """Return current timestamp in ISO 8601 UTC format."""
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class AnnotationRecord:
    """A ground-truth human (or verified synthetic) gameplay annotation."""

    sample_id: str
    source_fingerprint: dict[str, Any]
    source_id: str
    start_seconds: float
    end_seconds: float
    event_category: str
    editorial_rating: int  # 0 to 5
    context_completeness: bool
    setup_presence: bool
    payoff_presence: bool
    approved: str  # "y", "n", ""
    rejection_reason: str = ""
    notes: str = ""
    annotation_version: int = 1
    is_synthetic: bool = False
    created_at: str = field(default_factory=utc_iso_now)
    updated_at: str = field(default_factory=utc_iso_now)
    history: list[dict[str, Any]] = field(default_factory=list)

    def validate(self, video_duration: float | None = None) -> None:
        """Validate annotation consistency and interval bounds."""
        if not self.sample_id.strip():
            raise ValueError("sample_id must not be empty")
        if not math.isfinite(self.start_seconds) or not math.isfinite(self.end_seconds):
            raise ValueError("timestamps must be finite numbers")
        if self.start_seconds < 0:
            raise ValueError(f"start_seconds ({self.start_seconds}) cannot be negative")
        if self.end_seconds <= self.start_seconds:
            raise ValueError(
                f"end_seconds ({self.end_seconds}) must be strictly greater than start_seconds ({self.start_seconds})"
            )
        if video_duration is not None and self.end_seconds > video_duration + 0.25:
            raise ValueError(
                f"end_seconds ({self.end_seconds}) exceeds source duration ({video_duration})"
            )
        if not isinstance(self.editorial_rating, int) or not 0 <= self.editorial_rating <= 5:
            raise ValueError(
                f"editorial_rating ({self.editorial_rating}) must be an integer between 0 and 5"
            )
        if self.approved not in ("y", "n", ""):
            raise ValueError(f"approved must be 'y', 'n', or '' (got '{self.approved}')")
        if not self.event_category.strip():
            raise ValueError("event_category cannot be empty")

    @property
    def duration_seconds(self) -> float:
        return round(self.end_seconds - self.start_seconds, 6)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnnotationRecord:
        payload = dict(data)
        payload["start_seconds"] = float(payload["start_seconds"])
        payload["end_seconds"] = float(payload["end_seconds"])
        payload["editorial_rating"] = int(payload["editorial_rating"])
        payload["context_completeness"] = bool(payload.get("context_completeness", False))
        payload["setup_presence"] = bool(payload.get("setup_presence", False))
        payload["payoff_presence"] = bool(payload.get("payoff_presence", False))
        payload["annotation_version"] = int(payload.get("annotation_version", 1))
        payload["is_synthetic"] = bool(payload.get("is_synthetic", False))
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class VisualFrame:
    """Sampled visual frame metadata."""

    timestamp: float
    frame_path: str
    frame_type: str  # uniform | hook | peak | cut | boundary
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class TranscriptWordContext:
    word: str
    start: float
    end: float
    probability: float = 1.0


@dataclass(frozen=True, slots=True)
class TranscriptSegmentContext:
    text: str
    start: float
    end: float
    avg_logprob: float
    no_speech_prob: float
    words: tuple[TranscriptWordContext, ...] = ()


@dataclass(slots=True)
class SpeechContext:
    language: str | None = None
    segments: list[TranscriptSegmentContext] = field(default_factory=list)
    words: list[TranscriptWordContext] = field(default_factory=list)
    has_speech: bool = False
    provenance: str = "none"  # "cache" | "transcribed" | "none"


@dataclass(slots=True)
class ChatContext:
    present: bool = False
    message_count: int = 0
    unique_chatters: int = 0
    emote_density: float = 0.0
    caps_ratio: float = 0.0
    messages: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class CandidateContext:
    """Rich multimodal context package for a candidate highlight."""

    schema_version: int
    candidate_id: str
    source_video: str
    source_fingerprint: dict[str, Any]
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    peak_seconds: float
    candidate_score: float
    components: dict[str, float]
    unclipped_score: float | None = None
    unclipped_z_scores: dict[str, float] = field(default_factory=dict)
    rule: str = "ranked_only"
    frames: list[VisualFrame] = field(default_factory=list)
    speech: SpeechContext = field(default_factory=lambda: SpeechContext(has_speech=False, provenance="none"))
    chat: ChatContext = field(default_factory=lambda: ChatContext(present=False))
    extracted_at: str = field(default_factory=utc_iso_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CandidateContext:
        payload = dict(data)
        if "frames" in payload and isinstance(payload["frames"], list):
            payload["frames"] = [
                VisualFrame(**f) if isinstance(f, dict) else f
                for f in payload["frames"]
            ]
        if "speech" in payload and isinstance(payload["speech"], dict):
            sp = payload["speech"]
            words = [
                TranscriptWordContext(**w) if isinstance(w, dict) else w
                for w in sp.get("words", [])
            ]
            segments = [
                TranscriptSegmentContext(
                    text=s["text"],
                    start=s["start"],
                    end=s["end"],
                    avg_logprob=s.get("avg_logprob", 0.0),
                    no_speech_prob=s.get("no_speech_prob", 0.0),
                    words=tuple(
                        TranscriptWordContext(**w) if isinstance(w, dict) else w
                        for w in s.get("words", [])
                    ),
                )
                if isinstance(s, dict)
                else s
                for s in sp.get("segments", [])
            ]
            payload["speech"] = SpeechContext(
                language=sp.get("language"),
                segments=segments,
                words=words,
                has_speech=sp.get("has_speech", False),
                provenance=sp.get("provenance", "none"),
            )
        if "chat" in payload and isinstance(payload["chat"], dict):
            ch = payload["chat"]
            payload["chat"] = ChatContext(
                present=ch.get("present", False),
                message_count=ch.get("message_count", 0),
                unique_chatters=ch.get("unique_chatters", 0),
                emote_density=ch.get("emote_density", 0.0),
                caps_ratio=ch.get("caps_ratio", 0.0),
                messages=ch.get("messages", []),
            )
        return cls(**payload)


@dataclass(slots=True)
class CandidateMatch:
    """Record of a predicted candidate matched against ground-truth label."""

    candidate_rank: int
    candidate_window_id: str
    candidate_start: float
    candidate_end: float
    candidate_score: float
    matched_sample_id: str | None
    label_start: float | None
    label_end: float | None
    label_category: str | None
    label_rating: int | None
    label_approved: str | None
    iou: float
    match_status: str  # true_positive | false_positive | duplicate_match


@dataclass(slots=True)
class EvaluationMetricsReport:
    """Comprehensive evaluation metrics for candidate selection against ground truth."""

    total_candidates: int
    total_ground_truth_labels: int
    approved_ground_truth_labels: int
    top_k: int
    precision_at_k: float
    recall_at_k: float
    weighted_recall_at_k: float
    approval_rate_at_k: float
    mean_iou: float
    median_iou: float
    duplicate_rate: float
    category_metrics: dict[str, dict[str, float]]
    rejection_reasons: dict[str, int]
    matches: list[CandidateMatch]
    unmatched_labels: list[str]
    is_synthetic: bool = False
    evaluated_at: str = field(default_factory=utc_iso_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
