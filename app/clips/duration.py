"""One hard duration limit shared by live clips, VOD previews and finals."""

import math

MAX_CLIP_SECONDS = 60.0
# Leave room for video-frame/AAC-packet rounding at the hard container limit.
ENCODE_MAX_SECONDS = MAX_CLIP_SECONDS - 0.05


def peak_bounds(start: float, end: float, peak: float, *, maximum: float = MAX_CLIP_SECONDS,
                lower: float | None = None, upper: float | None = None) -> tuple[float, float]:
    """Recenter oversized windows with two-thirds setup, then shift at edges."""
    limit = min(maximum, MAX_CLIP_SECONDS)
    if not all(math.isfinite(value) for value in (start, end, peak, limit)) or limit <= 0 or end <= start:
        raise ValueError("invalid clip window")
    if end - start > limit:
        start = peak - limit * (2 / 3)
        end = start + limit
    if lower is not None and start < lower:
        end += lower - start
        start = lower
    if upper is not None and end > upper:
        start -= end - upper
        end = upper
    if lower is not None:
        start = max(lower, start)
    if upper is not None:
        end = min(upper, end)
    if end <= start or not start <= peak <= end:
        raise ValueError("peak is outside available footage")
    return start, end


def encode_duration(duration: float) -> float:
    if not math.isfinite(duration) or not 0 < duration <= MAX_CLIP_SECONDS+1e-6:
        raise ValueError(f"clip duration must be in (0, {MAX_CLIP_SECONDS:g}]; replan around its peak")
    return min(duration, ENCODE_MAX_SECONDS)


def guard_duration(actual: float, expected: float) -> None:
    if not math.isfinite(actual) or actual > MAX_CLIP_SECONDS or actual <= 0:
        raise ValueError(f"export duration {actual:.6f}s violates the strict {MAX_CLIP_SECONDS:.1f}s limit")
    if abs(actual - expected) > 0.15:
        raise ValueError(f"export duration {actual:.3f}s differs from requested {expected:.3f}s")
