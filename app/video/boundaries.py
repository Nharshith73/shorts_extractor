"""Resolve source-relative AI boundary recommendations conservatively."""
import math
from typing import Any, Sequence


def snap_to_speech_boundaries(
    start: float,
    end: float,
    peak: float,
    segments: Sequence[Any],
    min_duration: float = 15.0,
    max_duration: float = 45.0,
    max_lookback: float = 8.0,
    max_lookforward: float = 8.0,
) -> tuple[float, float]:
    """Snap candidate window boundaries to sentence beginnings and punchline ends.
    
    Prevents cutting speakers off mid-word or mid-sentence and ensures the comedic
    or dramatic setup is fully included.
    """
    if not segments:
        return start, end

    # 1. Look back from start to find where this sentence/thought began
    best_start = start
    lookback_floor = max(0.0, start - max_lookback)
    for s in segments:
        s_start = float(getattr(s, "start", 0.0) if hasattr(s, "start") else (s.get("start", 0.0) if isinstance(s, dict) else 0.0))
        s_end = float(getattr(s, "end", 0.0) if hasattr(s, "end") else (s.get("end", 0.0) if isinstance(s, dict) else 0.0))
        if lookback_floor <= s_start <= start + 1.0:
            best_start = s_start
            break
        elif s_start < lookback_floor and s_end > lookback_floor:
            words = getattr(s, "words", []) if hasattr(s, "words") else (s.get("words", []) if isinstance(s, dict) else [])
            for w in words:
                w_start = float(getattr(w, "start", 0.0) if hasattr(w, "start") else (w.get("start", 0.0) if isinstance(w, dict) else 0.0))
                if lookback_floor <= w_start <= start + 0.5:
                    best_start = w_start
                    break

    # 2. Look forward from end to find where the sentence/reaction ends
    best_end = end
    lookforward_ceiling = end + max_lookforward
    for s in reversed(segments):
        s_start = float(getattr(s, "start", 0.0) if hasattr(s, "start") else (s.get("start", 0.0) if isinstance(s, dict) else 0.0))
        s_end = float(getattr(s, "end", 0.0) if hasattr(s, "end") else (s.get("end", 0.0) if isinstance(s, dict) else 0.0))
        if end - 1.0 <= s_end <= lookforward_ceiling:
            best_end = s_end
            break

    # 3. Enforce adaptive bounds [min_duration, max_duration]
    dur = best_end - best_start
    if dur < min_duration:
        best_end = best_start + min_duration
    elif dur > max_duration:
        best_end = best_start + max_duration

    return round(best_start, 2), round(best_end, 2)


def resolve_boundaries(start, end, duration, recommended_start, recommended_end, confidence,
                       context_complete, minimum=1.0, maximum=60.0, min_confidence=0.5,
                       enabled=True, context_start=None, context_end=None):
    original = [float(start), float(end)]
    if not all(math.isfinite(x) for x in (*original, duration, minimum, maximum)) or not (0 <= start < end <= duration) or not (0 < minimum <= maximum):
        raise ValueError('invalid original boundaries or duration constraints')
    result = {'original': original, 'recommended': [recommended_start, recommended_end],
              'applied': original[:], 'accepted': False, 'reason': None,
              'coordinate_system': 'source_seconds', 'provenance': 'candidate'}
    reason = None
    if not enabled: reason = 'refinement disabled'
    elif not context_complete: reason = 'incomplete context'
    elif not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or confidence < min_confidence: reason = 'low or invalid confidence'
    elif not all(isinstance(x, (int, float)) and math.isfinite(x) for x in (recommended_start, recommended_end)): reason = 'missing or non-finite recommendation'
    elif not 0 <= recommended_start < recommended_end <= duration: reason = 'invalid or out-of-source recommendation'
    elif recommended_start < (start if context_start is None else context_start) or recommended_end > (end if context_end is None else context_end): reason = 'outside protected context'
    elif not minimum <= recommended_end - recommended_start <= maximum: reason = 'duration outside limits'
    if reason:
        result['reason'] = reason
    else:
        result.update(applied=[recommended_start, recommended_end], accepted=True, provenance='ai_recommendation')
    return result
