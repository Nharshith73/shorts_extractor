"""Provider-independent local model inference interface for future Stage 4B AI Highlight Judge."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import logging
import time
from typing import Any, Protocol, runtime_checkable

from .models import CandidateContext

LOGGER = logging.getLogger(__name__)

VALID_DECISIONS: tuple[str, ...] = ("APPROVE", "REJECT", "UNCERTAIN_NEEDS_HUMAN")
VALID_LAYOUTS: tuple[str, ...] = ("blurred_background", "stacked", "center_crop")


@dataclass(slots=True)
class HighlightJudgment:
    """Structured editorial evaluation emitted by a local AI Judge."""

    window_id: str
    decision: str  # APPROVE | REJECT | UNCERTAIN_NEEDS_HUMAN
    confidence: float  # 0.0 to 1.0
    editorial_scores: dict[str, float]  # hook_strength, narrative_coherence, etc.
    primary_category: str
    reasoning: str
    suggested_title: str
    hook_text: str
    recommended_layout: str  # blurred_background | stacked | center_crop
    suggested_bounds: dict[str, Any]
    model_metadata: dict[str, Any] = field(default_factory=dict)
    is_mock: bool = False

    @property
    def editorial_rating(self) -> int:
        """Derived 0-5 integer editorial rating based on entertainment score or metadata."""
        if "editorial_rating" in self.model_metadata:
            return int(self.model_metadata["editorial_rating"])
        ev = self.editorial_scores.get("entertainment_value", 6.0)
        return max(0, min(5, round(ev / 2.0)))

    @property
    def event_category(self) -> str:
        return self.primary_category

    def validate(self) -> None:
        """Validate judgment payload against Stage 4 JSON specification."""
        if not self.window_id.strip():
            raise ValueError("window_id cannot be empty")
        if self.decision not in VALID_DECISIONS:
            raise ValueError(f"decision must be one of {VALID_DECISIONS}, got '{self.decision}'")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence ({self.confidence}) must be in [0.0, 1.0]")
        if self.recommended_layout not in VALID_LAYOUTS:
            raise ValueError(f"recommended_layout must be one of {VALID_LAYOUTS}")

        required_scores = {
            "hook_strength",
            "narrative_coherence",
            "entertainment_value",
            "visual_clarity",
            "pacing",
        }
        missing_scores = required_scores - set(self.editorial_scores.keys())
        if missing_scores:
            raise ValueError(f"editorial_scores missing required fields: {sorted(missing_scores)}")

        for score_name, score_val in self.editorial_scores.items():
            if not 0.0 <= score_val <= 10.0:
                raise ValueError(f"score '{score_name}' ({score_val}) must be in [0.0, 10.0]")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["editorial_rating"] = self.editorial_rating
        data["event_category"] = self.event_category
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HighlightJudgment:
        judgment = cls(**data)
        judgment.validate()
        return judgment


@runtime_checkable
class HighlightJudgeProvider(Protocol):
    """Protocol for provider-independent local multimodal model inference."""

    @property
    def model_name(self) -> str:
        """Model identifier string."""
        ...

    def evaluate_candidate(
        self, context: CandidateContext, timeout_seconds: float = 60.0
    ) -> HighlightJudgment:
        """Process candidate multimodal context and emit structured judgment."""
        ...

    def judge_candidate(
        self, context: CandidateContext, timeout_seconds: float = 60.0
    ) -> HighlightJudgment:
        """Alias for evaluate_candidate."""
        ...


class MockHighlightJudgeProvider:
    """Deterministic reference mock provider used for testing inference contracts."""

    def __init__(
        self,
        model_name: str = "mock-highlight-judge",
        default_decision: str | None = None,
        default_rating: int | None = None,
        default_category: str | None = None,
    ) -> None:
        self._model_name = model_name
        if default_decision:
            up = default_decision.upper()
            self._default_decision = "APPROVE" if up == "APPROVE" else ("REJECT" if up == "REJECT" else "UNCERTAIN_NEEDS_HUMAN")
        else:
            self._default_decision = None
        self._default_rating = default_rating
        self._default_category = default_category

    @property
    def model_name(self) -> str:
        return self._model_name

    def judge_candidate(
        self, context: CandidateContext, timeout_seconds: float = 60.0
    ) -> HighlightJudgment:
        return self.evaluate_candidate(context, timeout_seconds=timeout_seconds)

    def evaluate_candidate(
        self, context: CandidateContext, timeout_seconds: float = 60.0
    ) -> HighlightJudgment:
        start_time = time.perf_counter()

        # Heuristic mock decision based on signal score and motion
        score = context.candidate_score
        motion = context.components.get("motion", 0.0)
        has_speech = context.speech.has_speech

        if self._default_decision is not None:
            decision = self._default_decision
            confidence = 0.90
            category = self._default_category or ("clutch_kill" if decision == "APPROVE" else "ordinary")
            reasoning = f"Mock preset decision: {decision}"
            hook_text = "MOCK HOOK"
            title = "MOCK TITLE"
        elif score >= 4.0:
            decision = "APPROVE"
            confidence = min(0.95, 0.60 + score * 0.05)
            category = "clutch_kill" if motion > 2.0 else "funny_dialogue"
            reasoning = f"Strong signal peak (score={score:.2f}) with dynamic action and clear payoff."
            hook_text = "WATCH THIS PLAY"
            title = "UNREAL CLUTCH MOMENT! 🔥"
        elif score >= 2.0:
            decision = "UNCERTAIN_NEEDS_HUMAN"
            confidence = 0.50
            category = "ordinary"
            reasoning = f"Moderate excitement (score={score:.2f}), requires human check."
            hook_text = "CLOSE CALL"
            title = "ALMOST DID IT"
        else:
            decision = "REJECT"
            confidence = 0.85
            category = "dead_air_silence" if not has_speech else "repetitive_driving"
            reasoning = f"Low energy activity (score={score:.2f}) lacking dramatic hook."
            hook_text = ""
            title = ""

        if self._default_category is not None:
            category = self._default_category

        ev = float(self._default_rating * 2.0) if self._default_rating is not None else min(10.0, max(1.0, score * 1.2))

        metadata: dict[str, Any] = {
            "provider": "mock",
            "model": self.model_name,
            "latency_seconds": round(time.perf_counter() - start_time, 4),
        }
        if self._default_rating is not None:
            metadata["editorial_rating"] = self._default_rating

        judgment = HighlightJudgment(
            window_id=context.candidate_id,
            decision=decision,
            confidence=round(confidence, 4),
            editorial_scores={
                "hook_strength": min(10.0, max(1.0, score * 1.5)),
                "narrative_coherence": 8.0 if has_speech else 4.0,
                "entertainment_value": ev,
                "visual_clarity": 8.5,
                "pacing": 7.5,
            },
            primary_category=category,
            reasoning=reasoning,
            suggested_title=title,
            hook_text=hook_text,
            recommended_layout="blurred_background",
            suggested_bounds={
                "refined_start_seconds": context.start_seconds,
                "refined_end_seconds": context.end_seconds,
                "facecam_detected": False,
                "facecam_region": None,
            },
            model_metadata=metadata,
            is_mock=True,
        )
        judgment.validate()
        return judgment
