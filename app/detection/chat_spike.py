"""Rolling median chat spike detection for live streams and recorded VODs."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import statistics
import time
from typing import Any, Sequence


@dataclass(slots=True)
class ChatSpikeEvent:
    """Detected chat excitement spike."""

    timestamp: float          # Peak timestamp of the spike (stream elapsed or epoch)
    message_count: int        # Messages counted in this bucket
    baseline_median: float    # Median messages per bucket over preceding window
    spike_ratio: float        # message_count / baseline_median
    lead_in_seconds: float = 20.0
    lookahead_seconds: float = 10.0
    context_start: float = 0.0
    context_end: float = 0.0

    def __post_init__(self) -> None:
        if self.context_start == 0.0 and self.context_end == 0.0:
            object.__setattr__(self, "context_start", max(0.0, self.timestamp - self.lead_in_seconds))
            object.__setattr__(self, "context_end", self.timestamp + self.lookahead_seconds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": round(self.timestamp, 2),
            "message_count": self.message_count,
            "baseline_median": round(self.baseline_median, 2),
            "spike_ratio": round(self.spike_ratio, 2),
            "context_start": round(self.context_start, 2),
            "context_end": round(self.context_end, 2),
            "duration": round(self.context_end - self.context_start, 2),
        }


class ChatSpikeDetector:
    """Real-time rolling median chat spike detector.

    A spike occurs when the message count in the current 10s bucket is:
    1. >= 2.5x the median of the last 3 minutes (18 buckets).
    2. >= min_messages (e.g. 5 for quiet streams, 15 for huge streams).
    """

    def __init__(
        self,
        bucket_seconds: float = 10.0,
        history_minutes: float = 3.0,
        spike_factor: float = 2.5,
        min_messages: int = 5,
        cooldown_seconds: float = 30.0,
        lead_in_seconds: float = 20.0,
        lookahead_seconds: float = 10.0,
    ) -> None:
        self.bucket_seconds = bucket_seconds
        self.history_buckets_count = max(3, int((history_minutes * 60.0) / bucket_seconds))
        self.spike_factor = spike_factor
        self.min_messages = min_messages
        self.cooldown_seconds = cooldown_seconds
        self.lead_in_seconds = lead_in_seconds
        self.lookahead_seconds = lookahead_seconds

        self._history_buckets: deque[int] = deque(maxlen=self.history_buckets_count)
        self._current_bucket_start: float | None = None
        self._current_bucket_count: int = 0
        self._last_spike_timestamp: float = -9999.0
        self._detected_spikes: list[ChatSpikeEvent] = []

    @property
    def current_median(self) -> float:
        if not self._history_buckets:
            return 1.0
        return float(statistics.median(self._history_buckets))

    @property
    def current_bucket_count(self) -> int:
        return self._current_bucket_count

    @property
    def detected_spikes(self) -> list[ChatSpikeEvent]:
        return list(self._detected_spikes)

    def add_message(self, timestamp: float) -> ChatSpikeEvent | None:
        """Feed a message timestamp and evaluate bucket transitions."""
        bucket_index = int(timestamp // self.bucket_seconds)
        bucket_start = bucket_index * self.bucket_seconds

        if self._current_bucket_start is None:
            self._current_bucket_start = bucket_start
            self._current_bucket_count = 1
            return None

        # Check if message falls in the current bucket
        if bucket_start == self._current_bucket_start:
            self._current_bucket_count += 1
            return None

        # Bucket completed -> evaluate spike on the completed bucket
        spike = self._evaluate_completed_bucket(
            bucket_time=self._current_bucket_start + (self.bucket_seconds / 2.0),
            count=self._current_bucket_count,
        )

        # Push the evaluated bucket into history
        self._history_buckets.append(self._current_bucket_count)

        # Fill any empty intermediate buckets with 0
        current_idx = int(self._current_bucket_start // self.bucket_seconds)
        missing_buckets = bucket_index - current_idx - 1
        for _ in range(max(0, missing_buckets)):
            self._history_buckets.append(0)

        self._current_bucket_start = bucket_start
        self._current_bucket_count = 1
        return spike

    def flush(self, current_time: float | None = None) -> ChatSpikeEvent | None:
        """Force evaluate the current in-progress bucket (e.g. on stream close or step)."""
        if self._current_bucket_start is None or self._current_bucket_count == 0:
            return None

        t = current_time or (self._current_bucket_start + (self.bucket_seconds / 2.0))
        spike = self._evaluate_completed_bucket(bucket_time=t, count=self._current_bucket_count)
        self._history_buckets.append(self._current_bucket_count)
        self._current_bucket_start = None
        self._current_bucket_count = 0
        return spike

    def _evaluate_completed_bucket(self, bucket_time: float, count: int) -> ChatSpikeEvent | None:
        """Check if count meets spike conditions."""
        median = self.current_median
        spike_ratio = count / max(1.0, median)

        threshold = max(float(self.min_messages), self.spike_factor * max(1.0, median))

        if count >= threshold:
            # Check cooldown debounce
            if bucket_time - self._last_spike_timestamp >= self.cooldown_seconds:
                self._last_spike_timestamp = bucket_time
                event = ChatSpikeEvent(
                    timestamp=bucket_time,
                    message_count=count,
                    baseline_median=median,
                    spike_ratio=spike_ratio,
                    lead_in_seconds=self.lead_in_seconds,
                    lookahead_seconds=self.lookahead_seconds,
                    context_start=max(0.0, bucket_time - self.lead_in_seconds),
                    context_end=bucket_time + self.lookahead_seconds,
                )
                self._detected_spikes.append(event)
                return event
        return None


def detect_spikes_from_chat_log(
    messages: Sequence[dict[str, Any] | float],
    bucket_seconds: float = 10.0,
    history_minutes: float = 3.0,
    spike_factor: float = 2.5,
    min_messages: int = 5,
    cooldown_seconds: float = 30.0,
    lead_in_seconds: float = 20.0,
    lookahead_seconds: float = 10.0,
) -> list[ChatSpikeEvent]:
    """Offline detector: scan a list of messages and find all excitement spikes."""
    detector = ChatSpikeDetector(
        bucket_seconds=bucket_seconds,
        history_minutes=history_minutes,
        spike_factor=spike_factor,
        min_messages=min_messages,
        cooldown_seconds=cooldown_seconds,
        lead_in_seconds=lead_in_seconds,
        lookahead_seconds=lookahead_seconds,
    )

    timestamps: list[float] = []
    for m in messages:
        if isinstance(m, (int, float)):
            timestamps.append(float(m))
        elif isinstance(m, dict):
            ts = m.get("timestamp") or m.get("time_in_seconds") or m.get("offset") or 0.0
            timestamps.append(float(ts))

    timestamps.sort()
    spikes: list[ChatSpikeEvent] = []
    for ts in timestamps:
        ev = detector.add_message(ts)
        if ev:
            spikes.append(ev)

    last_ev = detector.flush()
    if last_ev:
        spikes.append(last_ev)

    return spikes
