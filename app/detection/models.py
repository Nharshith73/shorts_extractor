"""Typed records shared by feature extraction, scoring, and persistence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class FeatureBin:
    """Raw lightweight features measured over one full or partial second."""

    source_id: str
    session_id: str
    wall_utc: datetime
    bin_seconds: float
    audio_rms_db: float
    voice_energy_db: float
    onset_strength: float
    motion: float
    cut_change: float
    segment_id: int | None = None


@dataclass(frozen=True, slots=True)
class ScoredBin:
    """A feature bin with robust-normalized components and qualification state."""

    feature: FeatureBin
    z_scores: dict[str, float]
    score: float
    qualifies: bool


@dataclass(frozen=True, slots=True)
class DetectedEvent:
    """One peak selected from a contiguous exciting region."""

    source_id: str
    session_id: str
    peak_wall_utc: datetime
    start_wall_utc: datetime
    end_wall_utc: datetime
    score: float
    components: dict[str, float]
