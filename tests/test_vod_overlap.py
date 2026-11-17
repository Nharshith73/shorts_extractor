import csv
from dataclasses import replace
import json
from pathlib import Path

import pytest

from app.clips.config import CandidateConfig
from app.detection.config import DetectionConfig
from app.detection.models import FeatureBin, ScoredBin
from app.detection.scoring import AuxiliaryFeatures, robust_zscore, score_feature_bins
from app.vod.artifacts import write_features, write_json
from app.vod.pipeline import rescore
from app.vod.selection import peak_window, rank_peaks
from app.vod.settings import SelectionConfig
from app.vod.timeline import media_seconds, media_time


def item(second: float, score: float, raw: float | None = None) -> ScoredBin:
    feature = FeatureBin("s", "vod", media_time(second), 1, -20, -30, 0, 0, 0)
    return ScoredBin(feature, {"motion": min(score, 10), "voice_energy_db": 2}, score, False,
                     {"motion": raw} if raw is not None else None, raw)


def select(items: list[ScoredBin], *, windows: CandidateConfig | None = None, separation: float = 0):
    windows = windows or CandidateConfig(pre_roll_seconds=8, post_roll_seconds=6, min_seconds=15, max_seconds=45)
    return rank_peaks(items, SelectionConfig(min_separation_seconds=separation),
                      DetectionConfig(warmup_seconds=0), windows, 0, 240)


def test_overlapping_lower_score_is_dropped_and_quota_refilled() -> None:
    selected = select([item(40, 10), item(45, 9), item(60, 8), item(100, 7)])
    assert [media_seconds(event.peak_wall_utc) for event, _ in selected] == [40, 60, 100]


def test_exactly_two_seconds_overlap_allowed_but_more_rejected() -> None:
    assert len(select([item(40, 10), item(53, 9)])) == 2
    assert len(select([item(40, 10), item(52.999, 9)])) == 1


def test_final_long_windows_obey_overlap_even_when_peak_separation_passes() -> None:
    windows = CandidateConfig(pre_roll_seconds=8, post_roll_seconds=6, min_seconds=30, max_seconds=30)
    selected = select([item(40, 10), item(60, 9), item(100, 8), item(160, 7)], windows=windows, separation=20)
    assert [media_seconds(event.peak_wall_utc) for event, _ in selected] == [40, 100, 160]
    bounds = [peak_window(event, windows, 0, 240) for event, _ in selected]
    for index, (start, end) in enumerate(bounds):
        assert all(min(end, right) - max(start, left) <= 2 for left, right in bounds[index + 1:])


def test_longer_defaults_drop_overlaps_and_continue_filling_the_shortlist() -> None:
    selected = select([item(40, 10), item(60, 9), item(95, 8), item(150, 7)], windows=CandidateConfig())
    assert [media_seconds(event.peak_wall_utc) for event, _ in selected] == [40, 95, 150]
    bounds = [peak_window(event, CandidateConfig(), 0, 240) for event, _ in selected]
    assert all(end-start == 46 for start, end in bounds)
    for index, (start, end) in enumerate(bounds):
        assert all(min(end, right) - max(start, left) <= 2 for left, right in bounds[index+1:])


def test_unclipped_sum_breaks_only_capped_score_ties() -> None:
    selected = select([item(40, 20, 40), item(45, 20, 80), item(100, 19, 1000), item(160, 21, 21)])
    assert [media_seconds(event.peak_wall_utc) for event, _ in selected] == [160, 45, 100]
    assert selected[1][0].unclipped_score == 80


def test_clipped_scores_and_rules_are_unchanged_by_raw_diagnostics() -> None:
    base = FeatureBin("s", "vod", media_time(0), 1, -20, -30, 0, 1, 0.1)
    features = [replace(base, wall_utc=media_time(second)) for second in range(30)]
    features.append(replace(base, wall_utc=media_time(30), motion=41, cut_change=0.5, audio_rms_db=-25))
    config = DetectionConfig(warmup_seconds=0)
    auxiliary = AuxiliaryFeatures({feature.wall_utc: {"chat_rate": 20 if index == 30 else 0}
                                   for index, feature in enumerate(features)}, {"chat_rate": 2}, {"chat_rate": 1})
    old = score_feature_bins(features, config, auxiliary)
    diagnostic = score_feature_bins(features, config, auxiliary, include_unclipped=True)
    assert [(row.z_scores, row.score, row.qualifies) for row in old] == [(row.z_scores, row.score, row.qualifies) for row in diagnostic]
    last = diagnostic[-1]
    assert last.z_scores["motion"] == last.z_scores["cut_change"] == 10
    assert last.unclipped_z_scores["motion"] == pytest.approx(40)
    assert last.unclipped_z_scores["cut_change"] == pytest.approx(40)
    assert last.unclipped_z_scores["audio_rms_db"] < 0
    expected = sum(config.weights.get(name, 2 if name == "chat_rate" else 0) * value
                   for name, value in last.unclipped_z_scores.items())
    assert last.unclipped_score == pytest.approx(expected)
    assert robust_zscore(100, [0, 0, 0], scale_floor=1) == 10


def test_rescore_exports_raw_components_to_csv_and_sidecars(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("detection:\n  warmup_seconds: 0\nselection:\n  min_separation_seconds: 0\n", encoding="utf-8")
    base = item(0, 0).feature
    features = [replace(base, wall_utc=media_time(second), motion=100 if second in (40, 90, 140) else 0,
                        audio_rms_db=-10 if second in (40, 90, 140) else -20,
                        voice_energy_db=-15 if second in (40, 90, 140) else -30)
                for second in range(180)]
    write_features(tmp_path / "features.csv", features, None)
    write_json(tmp_path / "analysis.json", dict(video="unavailable.mp4", source_id="s", fingerprint={},
        source_duration_seconds=180, start_seconds=0, end_seconds=180, chat=None))
    rescore(tmp_path, config)
    with (tmp_path / "review.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3
    for row in rows:
        raw = json.loads(row["unclipped_z_scores"])
        assert raw["motion"] == 100
        assert float(row["unclipped_score"]) == sum(raw.values())
        sidecar = json.loads(Path(row["clip_path"]).with_suffix(".json").read_text())
        assert sidecar["unclipped_z_scores"] == raw
        assert sidecar["unclipped_score"] == sum(raw.values())
        assert sidecar["components"]["motion"] == 10
