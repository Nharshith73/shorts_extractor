from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from app.clips.config import CandidateConfig
from app.detection.config import DetectionConfig
from app.detection.features import audio_features_by_bin, bin_layout, video_features_by_bin
from app.detection.models import FeatureBin, ScoredBin, DetectedEvent
from app.vod.artifacts import write_features, write_json, write_review
from app.vod.chunked import AudioBins, VideoBins
from app.vod.pipeline import rescore
from app.vod.review import review, summary
from app.vod.selection import peak_window, rank_peaks
from app.vod.settings import SelectionConfig, load_settings, parse_timestamp
from app.vod.timeline import media_time, media_seconds


def bin_at(second: float, score: float) -> ScoredBin:
    feature = FeatureBin("source", "vod", media_time(second), 1, -20, -30, 0, 0, 0)
    return ScoredBin(feature, {"motion": score}, score, False)


def test_chunk_state_preserves_partial_bins_filter_and_motion() -> None:
    sample_rate = 16000
    samples = np.random.default_rng(42).normal(0, 0.1, int(2.75 * sample_rate)).astype(np.float32)
    layout = bin_layout(2.75)
    audio = AudioBins(sample_rate)
    for start in range(0, len(samples), 5911):
        audio.feed(samples[start:start + 5911])
    actual = audio.finish()
    expected = audio_features_by_bin(samples, layout, sample_rate)
    for left, right in zip(actual, expected):
        assert left == pytest.approx(right, abs=1e-6)
    assert len(actual) == 3 and layout[-1][1] == 0.75
    frames = np.random.default_rng(2).integers(0, 256, (11, 90, 160), dtype=np.uint8)
    video = VideoBins(3, 4)
    for start in range(0, 11, 3):
        video.feed(frames[start:start+3])
    for left, right in zip(video.finish(), video_features_by_bin(frames, layout, 4)):
        assert left == pytest.approx(right, abs=1e-5)


def test_ranked_selection_ignores_threshold_gate_and_applies_optional_floor() -> None:
    config = DetectionConfig(warmup_seconds=0)
    windows = CandidateConfig(pre_roll_seconds=8, post_roll_seconds=6, min_seconds=15, max_seconds=45)
    scored = [bin_at(20, 1), bin_at(21, 2), bin_at(80, 0.5), bin_at(140, 0.2), bin_at(170, 0.1)]
    selected = rank_peaks(scored, SelectionConfig(min_separation_seconds=30), config, windows, 0, 180)
    assert [media_seconds(event.peak_wall_utc) for event, _ in selected] == [21, 80, 140]
    assert all(rule == "ranked_only" for _, rule in selected)
    assert rank_peaks(scored, SelectionConfig(min_score=3), config, windows, 0, 180) == []
    many = [bin_at(second, 1) for second in range(60, 7200, 60)]
    assert len(rank_peaks(many, SelectionConfig(clips_per_hour=100), config, windows, 0, 7200)) == 40


def test_vod_defaults_peak_window_and_original_time_trimming(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("{}", encoding="utf-8")
    _, _, windows = load_settings(config)
    assert (windows.pre_roll_seconds, windows.post_roll_seconds, windows.min_seconds, windows.max_seconds) == (8, 6, 15, 45)
    event = DetectedEvent("s", "vod", media_time(120), media_time(80), media_time(121), 5, {})
    assert peak_window(event, windows, 100, 200) == (112, 127)
    assert peak_window(event, windows, 118, 200) == (118, 133)
    assert peak_window(event, windows, 100, 122) is None
    assert parse_timestamp("01:02:03.5") == 3723.5


def test_rescore_never_decodes_and_reuses_unchanged_windows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("detection:\n  warmup_seconds: 0\nselection:\n  min_separation_seconds: 20\n", encoding="utf-8")
    features = [bin_at(second, 0).feature for second in range(180)]
    write_features(tmp_path / "features.csv", features, None)
    write_json(tmp_path / "analysis.json", dict(video=str(tmp_path / "unavailable.mp4"), source_id="s", fingerprint={},
        source_duration_seconds=180, start_seconds=0, end_seconds=180, chat=None))
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("rescore must not decode or render")
    monkeypatch.setattr("app.vod.pipeline.extract_chunked", forbidden)
    monkeypatch.setattr("app.vod.pipeline.render_cut", forbidden)
    monkeypatch.setattr("app.vod.pipeline.probe_duration", lambda p: 15.0)
    rescore(tmp_path, config)
    clips = json.loads((tmp_path / "clips.json").read_text())
    assert len(clips) == 3 and not any(clip["extracted"] for clip in clips)
    from app.vod.pipeline import make_render_cache_key, make_render_provenance
    cache = {}
    for clip in clips:
        path = Path(clip["clip_path"])
        path.write_bytes(b"cached video")
        prov = make_render_provenance({}, clip["start_seconds"], clip["end_seconds"], "libx264", "libx264")
        cache[clip["window_id"]] = dict(
            clip_path=str(path),
            size_bytes=path.stat().st_size,
            video_encoder="libx264",
            render_cache_key=make_render_cache_key(prov),
            provenance=prov,
        )
    write_json(tmp_path / "clip_cache.json", cache)
    rescore(tmp_path, config)
    assert all(clip["reused"] for clip in json.loads((tmp_path / "clips.json").read_text()))
    rescore(tmp_path, config, extract=True)
    assert all(clip["reused"] for clip in json.loads((tmp_path / "clips.json").read_text()))
    assert "<video" in (tmp_path / "index.html").read_text()


def test_rejection_reasons_and_summary_preserve_notes(tmp_path: Path) -> None:
    rows = []
    for index, motion in enumerate((6, 2)):
        path = tmp_path / f"clip{index}.mp4"
        write_json(path.with_suffix(".json"), {"components": {"motion": motion, "onset_strength": 1}})
        rows.append(dict(clip_path=str(path), start_seconds="0", end_seconds="15", score="5", top_features="motion",
                         approved="", notes="note"))
    write_review(tmp_path / "review.csv", rows)
    choices = iter(("y", "n", "e"))
    review(tmp_path, lambda prompt: next(choices))
    result = summary(tmp_path)
    assert result["approval_rate"] == 0.5
    assert result["reasons"] == {"e": 1}
    assert result["feature_averages"]["approved"]["motion"] == 6
    assert result["feature_averages"]["rejected"]["motion"] == 2
