from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.detection.config import DetectionConfig
from app.detection.models import FeatureBin, ScoredBin
from app.detection.scoring import detect_events, pick_events, robust_zscore, score_feature_bins


def feature_at(
    seconds: float,
    value: float,
    *,
    session_id: str = "session-a",
    bin_seconds: float = 1.0,
) -> FeatureBin:
    return FeatureBin(
        source_id="source-a",
        session_id=session_id,
        wall_utc=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=seconds),
        bin_seconds=bin_seconds,
        audio_rms_db=value,
        voice_energy_db=value,
        onset_strength=value,
        motion=value,
        cut_change=value,
    )


def test_robust_zscore_uses_median_mad_and_clipping() -> None:
    assert robust_zscore(3.0, [1.0, 2.0, 3.0, 4.0, 5.0]) == pytest.approx(0.0)
    assert robust_zscore(100.0, [0.0, 0.0, 0.0]) == 10.0
    assert robust_zscore(-1.0, [0.0, 0.0, 0.0]) == 0.0


def test_peak_picking_obeys_minimum_separation() -> None:
    config = DetectionConfig(warmup_seconds=0, min_separation_seconds=20)
    candidates = [
        ScoredBin(feature_at(70, 0), {"motion": 7.0}, 8.0, True),
        ScoredBin(feature_at(71, 0), {"motion": 0.0}, 0.0, False),
        ScoredBin(feature_at(75, 0), {"motion": 9.0}, 10.0, True),
        ScoredBin(feature_at(76, 0), {"motion": 0.0}, 0.0, False),
        ScoredBin(feature_at(100, 0), {"motion": 8.0}, 9.0, True),
    ]
    events = pick_events(candidates, config)
    assert [event.peak_wall_utc.second for event in events] == [15, 40]


def test_warmup_and_session_reset_suppress_early_events() -> None:
    config = DetectionConfig(warmup_seconds=2, z_threshold=3, single_feature_threshold=6)
    first_session = [feature_at(0, 0), feature_at(1, 0), feature_at(2, 10)]
    scored, events = detect_events(first_session, config)
    assert scored[-1].qualifies is True
    assert len(events) == 1

    reset_session = [
        feature_at(20, 0, session_id="session-b"),
        feature_at(21, 0, session_id="session-b"),
        feature_at(21.5, 10, session_id="session-b"),
    ]
    reset_scored = score_feature_bins(reset_session, config)
    assert reset_scored[-1].qualifies is False


def test_near_constant_baseline_and_terminal_partial_bin_do_not_trigger() -> None:
    config = DetectionConfig(warmup_seconds=30, baseline_seconds=30)
    features = [
        FeatureBin(
            source_id="source-a",
            session_id="session-a",
            wall_utc=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=index),
            bin_seconds=0.06 if index == 179 else 1.0,
            audio_rms_db=-61.075 + (0.006 if index in (30, 60, 120, 179) else 0.0),
            voice_energy_db=-73.408 + (0.007 if index in (30, 60, 120, 179) else 0.0),
            onset_strength=0.000006,
            motion=0.0,
            cut_change=0.0,
        )
        for index in range(180)
    ]
    scored, events = detect_events(features, config)
    assert not events
    assert all(not scored[index].qualifies for index in (30, 60, 120, 179))
