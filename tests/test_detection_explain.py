from datetime import UTC, datetime, timedelta

from app.detection.config import DetectionConfig
from app.detection.explain import background_centers, explain_window
from app.detection.features import FEATURE_NAMES
from app.detection.models import DetectedEvent, FeatureBin, ScoredBin
from app.detection.evaluate import Label


ORIGIN = datetime(1970, 1, 1, tzinfo=UTC)


def item(at: float, scores: tuple[float, float, float, float, float], qualifies: bool = False) -> ScoredBin:
    feature = FeatureBin("source", "session", ORIGIN + timedelta(seconds=at), 1.0, 0, 0, 0, 0, 0)
    components = dict(zip(FEATURE_NAMES, scores))
    return ScoredBin(feature, components, sum(scores), qualifies)


def reasons(current: ScoredBin, raw: ScoredBin | None = None, events: tuple[DetectedEvent, ...] = ()) -> tuple[str, ...]:
    return explain_window([current], [raw or current], events, DetectionConfig(), ORIGIN).reasons


def test_classifies_below_and_only_one() -> None:
    assert "below z_threshold" in reasons(item(100, (1, 1, 1, 1, 1)))
    assert "only one feature above" in reasons(item(100, (4, 0, 0, 0, 0)))


def test_classifies_warmup_and_actual_floor_counterfactual() -> None:
    assert "warmup" in reasons(item(30, (4, 4, 0, 0, 0)))
    assert "warmup" in reasons(item(30, (1, 1, 0, 0, 0)))
    blocked = reasons(item(100, (1, 1, 0, 0, 0)), item(100, (4, 4, 0, 0, 0), True))
    assert "suppressed by a scale floor (per-bin counterfactual)" in blocked
    assert "warmup" not in blocked


def test_classifies_separation_by_stronger_selected_event() -> None:
    selected = DetectedEvent("source", "session", ORIGIN + timedelta(seconds=110), ORIGIN + timedelta(seconds=110), ORIGIN + timedelta(seconds=111), 12, {})
    assert "within min_separation of a stronger event" in reasons(item(100, (4, 4, 0, 0, 0), True), events=(selected,))


def test_background_windows_do_not_overlap_labels() -> None:
    bins = [item(i, (0, 0, 0, 0, 0)) for i in range(100)]
    centers = background_centers(bins, [Label(50, "event")], 8)
    assert len(centers) == 200
    assert all(abs(center - 50) > 16 for center in centers)
    assert centers == background_centers(bins, [Label(50, "event")], 8)
