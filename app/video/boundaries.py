"""Resolve source-relative AI boundary recommendations conservatively."""
import math


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
