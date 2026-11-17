"""Rank each available live backlog, with per-source rolling-hour quotas."""

from datetime import datetime, timedelta
import math
from typing import Sequence

from .config import CandidateConfig
from .store import Candidate
from .windows import Event


def visual_only(event: Event) -> bool:
    if event.reason_codes:
        return "visual_only_penalized" in event.reason_codes
    support = max((value for name, value in event.components.items()
                   if name.startswith("chat_") or name in ("audio_rms_db", "voice_energy_db", "onset_strength")), default=0)
    return bool(event.components) and support < 2


def rejection_reason(event: Event, selected: Sequence[Event], candidates: Sequence[Candidate],
                     config: CandidateConfig) -> str:
    if visual_only(event):
        return "visual_only_penalized"
    neighbors = [*selected, *(candidate.window.primary for candidate in candidates
                              if candidate.status in ("waiting", "extracting", "extracted"))]
    if any(other.source_id == event.source_id and abs((other.peak-event.peak).total_seconds()) < config.min_separation_seconds for other in neighbors):
        return "suppressed_nms"
    return "quota_reached"


def ranked_live_events(events: Sequence[Event], candidates: Sequence[Candidate], config: CandidateConfig,
                       now: datetime) -> list[Event]:
    selected: list[Event] = []
    new_slots: list[Event] = []
    active = [item for item in candidates if item.status in ("waiting", "extracting", "extracted")]
    quota = config.top_k or max(1, math.ceil(config.clips_per_hour))
    for event in sorted(events, key=lambda item: (-item.score, item.peak)):
        if visual_only(event):
            continue
        # Later nearby events stay eligible for a waiting candidate's merge.
        waiting = [item for item in active if item.status == "waiting"
                   and item.window.primary.source_id == event.source_id
                   and item.window.primary.session_id == event.session_id
                   and abs((item.window.primary.peak-event.peak).total_seconds()) < config.min_separation_seconds]
        if waiting:
            selected.append(event)
            continue
        prior = [item.window.primary for item in active if item.window.primary.source_id == event.source_id
                 and item.window.primary.peak >= now-timedelta(hours=1)]
        peers = [item for item in new_slots if item.source_id == event.source_id]
        if len(prior)+len(peers) >= quota:
            continue
        neighbors = [item for item in selected if item.source_id == event.source_id]
        if any(abs((event.peak-other.peak).total_seconds()) < config.min_separation_seconds for other in (*prior, *neighbors)):
            continue
        selected.append(event)
        new_slots.append(event)
    return selected
