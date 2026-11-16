"""Fixed synthetic wall clock: one media second equals one timeline second."""

from datetime import UTC, datetime, timedelta
import math

EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def media_time(seconds: float) -> datetime:
    if not math.isfinite(seconds):
        raise ValueError("media seconds must be finite")
    return EPOCH + timedelta(seconds=seconds)


def media_seconds(timestamp: datetime) -> float:
    if timestamp.tzinfo is None:
        raise ValueError("timeline timestamp must be timezone-aware")
    return (timestamp - EPOCH).total_seconds()
