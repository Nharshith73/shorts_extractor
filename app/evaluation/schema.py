"""Strict, versioned schema definitions and validation for Stage 4B AI Highlight Judgments."""

from __future__ import annotations

import math
from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator, model_validator

VALID_EVIDENCE_SOURCES = ("visual", "transcript", "chat", "audio", "inference")
VALID_LAYOUTS = ("blurred_background", "stacked", "center_crop")
VALID_CATEGORIES = (
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


class EvidenceItem(BaseModel):
    """An individual piece of grounded multimodal evidence supporting an editorial judgment."""

    timestamp: float = Field(..., description="Source timestamp in seconds where observation occurred")
    source: Literal["visual", "transcript", "chat", "audio", "inference"] = Field(
        ..., description="Data source tier: visual, transcript, chat, audio, or inference"
    )
    observation: str = Field(..., min_length=1, description="Factual description of what was observed")

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: float) -> float:
        if not math.isfinite(v) or v < 0:
            raise ValueError(f"timestamp must be a non-negative finite number (got {v})")
        return round(float(v), 3)


class AIJudgmentRecord(BaseModel):
    """Stage 4B AI Highlight Judgment Schema representing deep editorial reasoning."""

    schema_version: str = Field(default="1.0", description="Schema version identifier")
    candidate_id: str = Field(..., min_length=1, description="Candidate window ID")
    event_category: str = Field(..., description="Categorical classification")
    summary: str = Field(..., min_length=5, description="Concise editorial summary of the clip")
    editorial_rating: int = Field(..., ge=0, le=5, description="Editorial rating (0-5 stars)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model certainty in judgment (0.0 to 1.0)")
    context_complete: bool = Field(..., description="Whether clip contains necessary context")
    setup_present: bool = Field(..., description="Whether setup / anticipation is present")
    payoff_present: bool = Field(..., description="Whether a clear climax / punchline occurred")
    entertainment_factors: list[str] = Field(
        default_factory=list, description="Specific entertainment elements identified"
    )
    evidence: list[EvidenceItem] = Field(
        default_factory=list, min_length=1, description="Observed multimodal evidence list"
    )
    hook_suggestion: str = Field(default="", description="Suggested on-screen text or caption hook")
    recommended_start_seconds: float = Field(..., description="Refined start timestamp in seconds")
    recommended_end_seconds: float = Field(..., description="Refined end timestamp in seconds")
    recommended_layout: str = Field(
        default="blurred_background", description="Optimal vertical layout profile"
    )
    rejection_reason: str | None = Field(
        default=None, description="Reason if editorial_rating <= 2; null if approved"
    )
    model_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Inference runtime provenance"
    )

    @field_validator("hook_suggestion", mode="before")
    @classmethod
    def normalize_hook(cls, v: Any) -> str:
        if v is None:
            return ""
        return str(v)

    @field_validator("recommended_layout", mode="before")
    @classmethod
    def normalize_layout(cls, v: Any) -> str:
        val = str(v).lower().strip() if v else "blurred_background"
        if val in VALID_LAYOUTS:
            return val
        if "stack" in val:
            return "stacked"
        if "crop" in val:
            return "center_crop"
        return "blurred_background"

    @field_validator("event_category", mode="before")
    @classmethod
    def validate_category(cls, v: Any) -> str:
        cat = str(v).strip().lower() if v else "unknown"
        if cat in VALID_CATEGORIES:
            return cat
        return "unknown"

    @model_validator(mode="after")
    def validate_interval_and_approval(self) -> AIJudgmentRecord:
        if self.recommended_end_seconds <= self.recommended_start_seconds:
            raise ValueError(
                f"recommended_end_seconds ({self.recommended_end_seconds}) must be strictly greater than "
                f"recommended_start_seconds ({self.recommended_start_seconds})"
            )
        # Low rating should ideally have a rejection reason
        if self.editorial_rating <= 2 and not self.rejection_reason:
            self.rejection_reason = "Low entertainment value or missing payoff"
        return self

    @property
    def is_approved(self) -> bool:
        """Threshold: clips with rating >= 3 are approved for vertical publishing."""
        return self.editorial_rating >= 3

    @property
    def recommended_duration(self) -> float:
        return round(self.recommended_end_seconds - self.recommended_start_seconds, 3)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
