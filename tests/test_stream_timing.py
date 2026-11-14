from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.ingestion.stream_reader import SessionTimestampTracker


def test_session_timestamp_tracker_reanchors_slow_source() -> None:
    tracker = SessionTimestampTracker()
    origin = datetime(2026, 1, 1, tzinfo=UTC)

    first_start, first_reanchored, first_drift = tracker.resolve(origin, 6, 2.0)
    assert first_start == origin
    assert first_reanchored is False
    assert first_drift == 0.0

    slow_filesystem_time = origin + timedelta(seconds=9)
    second_start, second_reanchored, second_drift = tracker.resolve(
        slow_filesystem_time, 6, 2.0
    )
    assert second_start == slow_filesystem_time
    assert second_reanchored is True
    assert second_drift == 3.0

    third_start, third_reanchored, third_drift = tracker.resolve(
        slow_filesystem_time + timedelta(seconds=6, milliseconds=100), 6, 2.0
    )
    assert third_start == slow_filesystem_time + timedelta(seconds=6)
    assert third_reanchored is False
    assert third_drift == pytest.approx(0.1)
