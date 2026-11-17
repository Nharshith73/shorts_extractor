"""Regression tests for P1: Consistent ranking semantics across selection and evaluation."""

from datetime import UTC, datetime
import json
from pathlib import Path

import pytest

from app.clips.config import CandidateConfig
from app.detection.config import DetectionConfig
from app.detection.models import FeatureBin, ScoredBin
from app.vod.evaluate import evaluate
from app.vod.pipeline import event_data
from app.vod.selection import canonical_ranking_key, rank_peaks
from app.vod.settings import SelectionConfig
from app.vod.timeline import EPOCH, media_time


def _dummy_bin(wall_seconds: float, score: float, unclipped_score: float) -> ScoredBin:
    t = media_time(wall_seconds)
    feature = FeatureBin(
        source_id="src1",
        session_id="sess1",
        wall_utc=t,
        bin_seconds=1.0,
        audio_rms_db=-10.0,
        voice_energy_db=-10.0,
        onset_strength=1.0,
        motion=5.0,
        cut_change=0.5,
    )
    return ScoredBin(
        feature=feature,
        z_scores={"audio_rms_db": score},
        score=score,
        qualifies=True,
        unclipped_z_scores={"audio_rms_db": unclipped_score},
        unclipped_score=unclipped_score,
    )


def test_tied_clipped_score_breaks_tie_with_unclipped_score_in_selection_and_evaluation(tmp_path: Path) -> None:
    """Prove that when clipped scores tie, unclipped score breaks the tie identically in selection and evaluation."""
    # Candidate A: earlier in time (200s), but lower unclipped score (11.0)
    # Candidate B: later in time (500s), but higher unclipped score (15.0)
    # Both have identical clipped score: 10.0
    cand_a = _dummy_bin(wall_seconds=200.0, score=10.0, unclipped_score=11.0)
    cand_b = _dummy_bin(wall_seconds=500.0, score=10.0, unclipped_score=15.0)

    # 1. Direct key check
    key_a = canonical_ranking_key(cand_a.score, cand_a.unclipped_score, 200.0)
    key_b = canonical_ranking_key(cand_b.score, cand_b.unclipped_score, 500.0)
    # Candidate B must rank before Candidate A because -15.0 < -11.0
    assert key_b < key_a

    # 2. Run selection
    selection_cfg = SelectionConfig(clips_per_hour=12, min_score=None, min_separation_seconds=30)
    detection_cfg = DetectionConfig(warmup_seconds=60)
    candidate_cfg = CandidateConfig(pre_roll_seconds=8, post_roll_seconds=6, min_seconds=15, max_seconds=45)

    selected = rank_peaks([cand_a, cand_b], selection_cfg, detection_cfg, candidate_cfg, start=0.0, end=1000.0)
    assert len(selected) == 2
    # First chosen candidate must be B (peak at 500s)
    top_event, _ = selected[0]
    second_event, _ = selected[1]
    assert (top_event.peak_wall_utc - EPOCH).total_seconds() == pytest.approx(500.0)
    assert (second_event.peak_wall_utc - EPOCH).total_seconds() == pytest.approx(200.0)

    # 3. Export to events.json
    events_json = {
        "video": "test.mp4",
        "duration_seconds": 1000.0,
        "start_seconds": 0.0,
        "end_seconds": 1000.0,
        "events": [
            dict(**event_data(event, rank), rank=rank, rule=rule)
            for rank, (event, rule) in enumerate(selected, 1)
        ],
    }
    output_dir = tmp_path / "vod_out"
    output_dir.mkdir()
    (output_dir / "events.json").write_text(json.dumps(events_json, indent=2), encoding="utf-8")

    # 4. Create weak labels file:
    # Label 1 matches Candidate B around 500s with weight 10.0
    # Label 2 matches Candidate A around 200s with weight 5.0
    labels_csv = tmp_path / "labels.csv"
    labels_csv.write_text(
        "start_seconds,duration_seconds,weight\n"
        "495.0,10.0,10.0\n"
        "195.0,10.0,5.0\n",
        encoding="utf-8",
    )

    # 5. Evaluate with top_k=1
    # Evaluation must evaluate Candidate B (rank 1), matching the 500s label
    metrics_top1 = evaluate(output_dir, labels_csv, tolerance=10.0, top_k=1)
    assert metrics_top1.precision_at_k == 1.0  # 1/1
    # Recall at 1 is 1 out of 2 labels = 0.5
    assert metrics_top1.recall_at_k == 0.5
    # Weighted recall is 10.0 / 15.0 = 0.6667 (proving it chose Candidate B, not Candidate A)
    assert metrics_top1.weighted_recall == pytest.approx(10.0 / 15.0, rel=1e-3)
    # The missed label is index 1 (the 200s label)
    assert metrics_top1.missed == (1,)


def test_legacy_data_fallback_when_unclipped_scores_missing() -> None:
    """When unclipped scores are missing/None, fallback gracefully to clipped score."""
    key1 = canonical_ranking_key(score=8.5, unclipped_score=None, time_seconds=100.0)
    key2 = canonical_ranking_key(score=8.5, unclipped_score=8.5, time_seconds=100.0)
    assert key1 == key2

    # Tied clipped score without unclipped scores falls back to earliest timestamp
    key_earlier = canonical_ranking_key(score=8.5, unclipped_score=None, time_seconds=50.0)
    key_later = canonical_ranking_key(score=8.5, unclipped_score=None, time_seconds=150.0)
    assert key_earlier < key_later
