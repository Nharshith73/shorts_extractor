"""Pure candidate window planning, merging, and boundary trimming."""

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from typing import Sequence

from app.ingestion.rolling_buffer import SegmentRecord, SessionBoundary
from .config import CandidateConfig
from .duration import peak_bounds


@dataclass(frozen=True, slots=True)
class Event:
    id: int
    source_id: str
    session_id: str
    peak: datetime
    start: datetime
    end: datetime
    score: float
    reason_codes: tuple[str, ...] = ()
    components: dict[str, float] = field(default_factory=dict)
    ranking_components: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Window:
    events: tuple[Event, ...]
    start: datetime
    end: datetime
    has_gap: bool = False

    @property
    def primary(self) -> Event:
        return max(self.events, key=lambda event: (event.score, -event.peak.timestamp()))


def event_window(event: Event, config: CandidateConfig) -> Window:
    start = min(event.peak, event.start) - timedelta(seconds=config.pre_roll_seconds)
    end = max(event.peak, event.end) + timedelta(seconds=config.post_roll_seconds)
    if (end - start).total_seconds() < config.min_seconds:
        end = start + timedelta(seconds=config.min_seconds)
    return cap_window(Window((event,), start, end), config)


def cap_window(window: Window, config: CandidateConfig) -> Window:
    origin = window.primary.peak
    start, end = peak_bounds((window.start-origin).total_seconds(), (window.end-origin).total_seconds(),
                             0, maximum=config.duration_limit)
    return replace(window, start=origin + timedelta(seconds=start), end=origin + timedelta(seconds=end))


def merge_windows(windows: Sequence[Window], config: CandidateConfig) -> list[Window]:
    result: list[Window] = []
    for window in sorted(windows, key=lambda item: (item.primary.source_id, item.primary.session_id, item.start)):
        if result:
            previous = result[-1]
            same_session = (previous.primary.source_id, previous.primary.session_id) == (window.primary.source_id, window.primary.session_id)
            end = max(previous.end, window.end)
            # A nearby cluster becomes one peak-centred clip even if its union is long.
            close_peaks = abs((window.primary.peak-previous.primary.peak).total_seconds()) < config.min_separation_seconds
            fits = (end-previous.start).total_seconds() <= config.duration_limit
            if same_session and window.start <= previous.end + timedelta(seconds=config.merge_gap_seconds) and (fits or close_peaks):
                result[-1] = cap_window(Window(previous.events + window.events, previous.start, end), config)
                continue
        result.append(cap_window(window, config))
    return result


def trim_gaps(window: Window, boundaries: Sequence[SessionBoundary], segments: Sequence[SegmentRecord]) -> Window:
    """Keep the contiguous side containing the highest-scoring event's peak."""
    start, end, peak = window.start, window.end, window.primary.peak
    for boundary in boundaries:
        if boundary.source_id != window.primary.source_id:
            continue
        at = boundary.boundary_utc
        if start < at < end:
            if at <= peak:
                start = at
            else:
                # The previous session can end before the new one starts.
                previous = next((record for record in segments if record.id == boundary.previous_segment_id), None)
                end = min(end, previous.wall_end_utc if previous else at)
    # Re-anchors or missing middle segments can create gaps inside a session.
    ordered = sorted((record for record in segments if record.source_id == window.primary.source_id
                      and record.session_id == window.primary.session_id and record.status == "complete"), key=lambda record: record.wall_start_utc)
    for left, right in zip(ordered, ordered[1:]):
        if (right.wall_start_utc - left.wall_end_utc).total_seconds() > 0.15:
            if peak >= right.wall_start_utc:
                start = max(start, right.wall_start_utc)
            elif peak < left.wall_end_utc:
                end = min(end, left.wall_end_utc)
    return replace(window, start=start, end=end, has_gap=window.has_gap or start != window.start or end != window.end)


def availability(window: Window, segments: Sequence[SegmentRecord], config: CandidateConfig, created: datetime, now: datetime,
                 retention_seconds: float = 900) -> str:
    records = sorted((record for record in segments if record.source_id == window.primary.source_id and record.session_id == window.primary.session_id and record.status == "complete"), key=lambda record: record.wall_start_utc)
    end = window.end + timedelta(seconds=config.grace_seconds)
    covering = [record for record in records if record.wall_start_utc < end and record.wall_end_utc > window.start]
    if records and records[0].wall_start_utc > window.start + timedelta(seconds=0.15):
        return "expired"
    if not covering:
        if (now - window.end).total_seconds() > retention_seconds:
            return "expired"
    else:
        cursor = window.start
        for record in covering:
            if record.path is None or not record.path.is_file():
                return "expired"
            if record.wall_start_utc > cursor + timedelta(seconds=0.15):
                return "expired"
            cursor = max(cursor, record.wall_end_utc)
        if cursor >= end:
            return "ready"
    return "timeout" if (now - created).total_seconds() >= config.wait_timeout_seconds else "waiting"
