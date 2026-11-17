"""Stage 4A: Comprehensive Unit and Regression Tests.

Covers:
1. Dataset creation and loading (EvaluationDataset).
2. Annotation updates with revision history.
3. Invalid timestamp rejection.
4. Source fingerprint mismatch prevention.
5. Synthetic versus human label separation.
6. Frame sampling and original timestamp preservation.
7. Transcript reuse from cache without re-transcription.
8. Missing audio handling.
9. Missing chat handling.
10. Context artifact recovery and schema validation.
11. Candidate-to-label deterministic matching.
12. Precision@K and Recall@K calculations.
13. Duplicate prediction handling.
14. Deterministic evaluation reports (JSON, CSV, MD).
15. Backward compatibility with existing app.vod.evaluate.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import pytest

from app.evaluation.context import (
    FrameSamplerConfig,
    extract_chat_context,
    extract_speech_context,
    plan_frame_timestamps,
)
from app.evaluation.dataset import (
    DEFAULT_MANIFEST_PATH,
    EvaluationDataset,
    compute_source_fingerprint,
    generate_sample_id,
)
from app.evaluation.evaluate import (
    compute_temporal_iou,
    evaluate_candidates,
    export_evaluation_artifacts,
    generate_summary_markdown,
)
from app.evaluation.inference import (
    HighlightJudgment,
    HighlightJudgeProvider,
    MockHighlightJudgeProvider,
)
from app.evaluation.models import (
    AnnotationRecord,
    CandidateContext,
    CandidateMatch,
    ChatContext,
    EvaluationMetricsReport,
    SpeechContext,
    TranscriptSegmentContext,
    TranscriptWordContext,
    VisualFrame,
)


# ---------------------------------------------------------------------------
# Test 1: Dataset creation and loading
# ---------------------------------------------------------------------------
def test_dataset_creation_and_loading(tmp_path: Path) -> None:
    manifest_file = tmp_path / "manifest.json"
    dataset = EvaluationDataset(manifest_file)
    assert len(dataset) == 0

    record = AnnotationRecord(
        sample_id="test_sample_01",
        source_fingerprint={"path": "video.mp4", "size": 12345, "mtime_ns": 9999},
        source_id="vod_100",
        start_seconds=10.0,
        end_seconds=25.0,
        event_category="chaotic",
        editorial_rating=4,
        context_completeness=True,
        setup_presence=True,
        payoff_presence=True,
        approved="y",
        notes="High action sequence",
        is_synthetic=False,
    )
    dataset.add_or_update(record)
    assert len(dataset) == 1
    assert manifest_file.is_file()

    # Companion CSV exists
    csv_file = manifest_file.with_suffix(".csv")
    assert csv_file.is_file()
    csv_text = csv_file.read_text(encoding="utf-8")
    assert "test_sample_01" in csv_text
    assert "chaotic" in csv_text

    # Reload from disk
    reloaded = EvaluationDataset(manifest_file)
    assert len(reloaded) == 1
    loaded_rec = reloaded.get("test_sample_01")
    assert loaded_rec is not None
    assert loaded_rec.sample_id == "test_sample_01"
    assert loaded_rec.editorial_rating == 4
    assert loaded_rec.approved == "y"
    assert loaded_rec.duration_seconds == 15.0


# ---------------------------------------------------------------------------
# Test 2: Annotation updates with revision history
# ---------------------------------------------------------------------------
def test_annotation_updates_with_revision_history(tmp_path: Path) -> None:
    manifest_file = tmp_path / "manifest.json"
    dataset = EvaluationDataset(manifest_file)

    rec_v1 = AnnotationRecord(
        sample_id="sample_rev",
        source_fingerprint={"path": "vid.mp4", "size": 100, "mtime_ns": 1},
        source_id="vid",
        start_seconds=5.0,
        end_seconds=15.0,
        event_category="chaotic",
        editorial_rating=2,
        context_completeness=True,
        setup_presence=False,
        payoff_presence=True,
        approved="n",
        rejection_reason="Weak setup",
        notes="Initial review",
        annotation_version=1,
    )
    dataset.add_or_update(rec_v1, author="editor_1")
    assert dataset.get("sample_rev").annotation_version == 1
    assert len(dataset.get("sample_rev").history) == 0

    # Second pass: update rating and approve
    rec_v2 = AnnotationRecord(
        sample_id="sample_rev",
        source_fingerprint={"path": "vid.mp4", "size": 100, "mtime_ns": 1},
        source_id="vid",
        start_seconds=5.0,
        end_seconds=15.0,
        event_category="impressive",
        editorial_rating=4,
        context_completeness=True,
        setup_presence=True,
        payoff_presence=True,
        approved="y",
        rejection_reason="",
        notes="Re-reviewed with extended context",
    )
    saved = dataset.add_or_update(rec_v2, author="senior_editor")
    assert saved.annotation_version == 2
    assert len(saved.history) == 1
    h0 = saved.history[0]
    assert h0["version"] == 1
    assert h0["author"] == "senior_editor"
    assert h0["editorial_rating"] == 2
    assert h0["approved"] == "n"
    assert h0["rejection_reason"] == "Weak setup"


# ---------------------------------------------------------------------------
# Test 3: Invalid timestamp rejection
# ---------------------------------------------------------------------------
def test_invalid_timestamp_rejection() -> None:
    base = dict(
        sample_id="inv_test",
        source_fingerprint={"path": "v.mp4", "size": 10, "mtime_ns": 1},
        source_id="v",
        event_category="fail",
        editorial_rating=3,
        context_completeness=True,
        setup_presence=True,
        payoff_presence=True,
        approved="y",
    )

    # Negative start
    r1 = AnnotationRecord(**base, start_seconds=-1.0, end_seconds=10.0)
    with pytest.raises(ValueError, match="cannot be negative"):
        r1.validate(video_duration=60.0)

    # end <= start
    r2 = AnnotationRecord(**base, start_seconds=15.0, end_seconds=15.0)
    with pytest.raises(ValueError, match="strictly greater"):
        r2.validate(video_duration=60.0)

    r3 = AnnotationRecord(**base, start_seconds=20.0, end_seconds=10.0)
    with pytest.raises(ValueError, match="strictly greater"):
        r3.validate(video_duration=60.0)

    # Exceeds duration
    r4 = AnnotationRecord(**base, start_seconds=50.0, end_seconds=65.0)
    with pytest.raises(ValueError, match="exceeds source duration"):
        r4.validate(video_duration=60.0)

    # Non-finite numbers
    r5 = AnnotationRecord(**base, start_seconds=float("nan"), end_seconds=10.0)
    with pytest.raises(ValueError, match="finite numbers"):
        r5.validate(video_duration=60.0)


# ---------------------------------------------------------------------------
# Test 4: Source fingerprint mismatch prevention
# ---------------------------------------------------------------------------
def test_source_fingerprint_mismatch_prevention(tmp_path: Path) -> None:
    manifest_file = tmp_path / "manifest.json"
    dataset = EvaluationDataset(manifest_file)

    rec = AnnotationRecord(
        sample_id="sample_fp",
        source_fingerprint={"path": "C:/vods/stream_a.mp4", "size": 5000000, "mtime_ns": 1111},
        source_id="stream_a",
        start_seconds=1.0,
        end_seconds=10.0,
        event_category="chaotic",
        editorial_rating=3,
        context_completeness=True,
        setup_presence=True,
        payoff_presence=True,
        approved="y",
    )
    dataset.add_or_update(rec)

    # Same source_id but different file size -> should be filtered out
    foreign_fp = {"path": "C:/vods/stream_a.mp4", "size": 9999999, "mtime_ns": 2222}
    matches = dataset.filter_by_source(source_id="stream_a", source_fingerprint=foreign_fp)
    assert len(matches) == 0

    # Matching fingerprint -> matched
    matching_fp = {"path": "C:/vods/stream_a.mp4", "size": 5000000, "mtime_ns": 1111}
    matches_ok = dataset.filter_by_source(source_id="stream_a", source_fingerprint=matching_fp)
    assert len(matches_ok) == 1


# ---------------------------------------------------------------------------
# Test 5: Synthetic versus human label separation
# ---------------------------------------------------------------------------
def test_synthetic_vs_human_label_separation(tmp_path: Path) -> None:
    manifest_file = tmp_path / "manifest.json"
    dataset = EvaluationDataset(manifest_file)

    rec_human = AnnotationRecord(
        sample_id="h1",
        source_fingerprint={"path": "v.mp4", "size": 100, "mtime_ns": 1},
        source_id="v",
        start_seconds=0.0,
        end_seconds=10.0,
        event_category="funny",
        editorial_rating=4,
        context_completeness=True,
        setup_presence=True,
        payoff_presence=True,
        approved="y",
        is_synthetic=False,
    )
    rec_synth = AnnotationRecord(
        sample_id="s1",
        source_fingerprint={"path": "v.mp4", "size": 100, "mtime_ns": 1},
        source_id="v",
        start_seconds=15.0,
        end_seconds=25.0,
        event_category="fail",
        editorial_rating=3,
        context_completeness=True,
        setup_presence=True,
        payoff_presence=True,
        approved="y",
        is_synthetic=True,
    )
    dataset.add_or_update(rec_human)
    dataset.add_or_update(rec_synth)

    assert len(dataset.human_records()) == 1
    assert dataset.human_records()[0].sample_id == "h1"

    assert len(dataset.synthetic_records()) == 1
    assert dataset.synthetic_records()[0].sample_id == "s1"

    raw = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert raw["human_samples"] == 1
    assert raw["synthetic_samples"] == 1
    assert raw["total_samples"] == 2


# ---------------------------------------------------------------------------
# Test 6: Frame sampling and original timestamp preservation
# ---------------------------------------------------------------------------
def test_frame_sampling_and_timestamp_preservation() -> None:
    cfg = FrameSamplerConfig(
        max_frames=8,
        include_hook_frame=True,
        include_peak_frame=True,
        include_boundary_frames=True,
    )
    # 20 second window from 10.0 to 30.0, peak at 17.5
    sampled = plan_frame_timestamps(10.0, 30.0, peak_seconds=17.5, cut_timestamps=[14.0], config=cfg)

    # Must preserve exact original timestamps
    types = {f[1]: f[0] for f in sampled}
    assert types["boundary_start"] == 10.0
    assert types["hook"] == 10.5
    assert types["peak"] == 17.5
    assert types["scene_cut"] == 14.0
    assert 29.0 <= types["boundary_end"] <= 30.0

    # All timestamps must be inside the window
    for t, label in sampled:
        assert 10.0 <= t <= 30.0


# ---------------------------------------------------------------------------
# Test 7: Transcript reuse from cache without re-transcription
# ---------------------------------------------------------------------------
def test_transcript_reuse_from_cache(tmp_path: Path) -> None:
    transcripts_dir = tmp_path / ".transcripts"
    transcripts_dir.mkdir(parents=True)
    cache_file = transcripts_dir / "my_vod.json"

    # Mock whisper cache with word-level timestamps
    cached_payload = {
        "text": "Check this insane stunt out",
        "language": "en",
        "segments": [
            {
                "id": 0,
                "start": 5.0,
                "end": 8.0,
                "text": "Check this insane stunt out",
                "words": [
                    {"word": "Check", "start": 5.0, "end": 5.4, "probability": 0.95},
                    {"word": "this", "start": 5.5, "end": 5.8, "probability": 0.98},
                    {"word": "insane", "start": 5.9, "end": 6.5, "probability": 0.99},
                    {"word": "stunt", "start": 6.6, "end": 7.2, "probability": 0.97},
                    {"word": "out", "start": 7.3, "end": 7.8, "probability": 0.96},
                ],
            },
            {
                "id": 1,
                "start": 20.0,
                "end": 22.0,
                "text": "Too late",
                "words": [{"word": "Too", "start": 20.0, "end": 20.5, "probability": 0.9}],
            },
        ],
    }
    cache_file.write_text(json.dumps(cached_payload), encoding="utf-8")

    # Slice window [4.0, 10.0] -> should capture segment 0 and its words, ignoring segment 1
    speech_ctx = extract_speech_context(
        candidate_start=4.0,
        candidate_end=10.0,
        transcripts_dir=transcripts_dir,
    )

    assert speech_ctx.has_speech is True
    assert speech_ctx.provenance == "cache"
    assert speech_ctx.language == "en"
    assert len(speech_ctx.segments) == 1
    assert speech_ctx.segments[0].text == "Check this insane stunt out"
    assert len(speech_ctx.words) == 5
    assert speech_ctx.words[2].word == "insane"


# ---------------------------------------------------------------------------
# Test 8: Missing audio handling
# ---------------------------------------------------------------------------
def test_missing_audio_handling(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty_transcripts"
    empty_dir.mkdir()

    speech_ctx = extract_speech_context(
        candidate_start=0.0,
        candidate_end=15.0,
        transcripts_dir=empty_dir,
    )

    assert speech_ctx.has_speech is False
    assert speech_ctx.provenance == "none"
    assert len(speech_ctx.segments) == 0
    assert len(speech_ctx.words) == 0


# ---------------------------------------------------------------------------
# Test 9: Missing chat handling
# ---------------------------------------------------------------------------
def test_missing_chat_handling() -> None:
    # No path supplied
    chat_none = extract_chat_context(None, 0.0, 15.0)
    assert chat_none.present is False
    assert chat_none.message_count == 0

    # Non-existent file
    chat_missing = extract_chat_context(Path("does_not_exist.json"), 0.0, 15.0)
    assert chat_missing.present is False
    assert chat_missing.message_count == 0

    # Present chat
    mock_chat_data = {
        "comments": [
            {
                "content_offset_seconds": 12.0,
                "commenter": {"display_name": "gamer1"},
                "message": {
                    "body": "OMEGALUL THAT WAS CRAZY",
                    "emotes": [{"_id": "emote123", "begin": 0, "end": 7}],
                },
            },
            {
                "content_offset_seconds": 14.5,
                "commenter": {"display_name": "gamer2"},
                "message": {"body": "poggers lol"},
            },
            {
                "content_offset_seconds": 50.0,
                "commenter": {"display_name": "gamer3"},
                "message": {"body": "outside window"},
            },
        ]
    }
    chat_tmp = Path("mock_chat.json")
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(mock_chat_data, f)
        temp_name = Path(f.name)

    try:
        chat_extracted = extract_chat_context(temp_name, 10.0, 20.0)
        assert chat_extracted.present is True
        assert chat_extracted.message_count == 2
        assert chat_extracted.unique_chatters == 2
        assert chat_extracted.emote_density > 0.0
    finally:
        if temp_name.exists():
            temp_name.unlink()


# ---------------------------------------------------------------------------
# Test 10: Context artifact recovery and schema validation
# ---------------------------------------------------------------------------
def test_context_artifact_recovery_and_schema(tmp_path: Path) -> None:
    candidate_ctx = CandidateContext(
        schema_version=1,
        candidate_id="cand_abc123",
        source_video="C:/vods/test.mp4",
        source_fingerprint={"path": "test.mp4", "size": 1000, "mtime_ns": 1},
        start_seconds=10.0,
        end_seconds=25.0,
        duration_seconds=15.0,
        peak_seconds=18.0,
        candidate_score=4.85,
        components={"audio_rms_db": 1.2, "motion": 3.4},
        frames=[
            VisualFrame(timestamp=10.0, frame_path="frame_0.jpg", frame_type="boundary_start", width=640, height=360),
            VisualFrame(timestamp=18.0, frame_path="frame_1.jpg", frame_type="peak", width=640, height=360),
        ],
        speech=SpeechContext(language="en", segments=[], words=[], has_speech=False, provenance="none"),
        chat=ChatContext(present=False, message_count=0, unique_chatters=0, emote_density=0.0, caps_ratio=0.0, messages=[]),
    )

    ctx_file = tmp_path / "context.json"
    ctx_file.write_text(json.dumps(candidate_ctx.to_dict(), indent=2), encoding="utf-8")

    # Recover from JSON
    loaded_data = json.loads(ctx_file.read_text(encoding="utf-8"))
    recovered = CandidateContext.from_dict(loaded_data)

    assert recovered.candidate_id == "cand_abc123"
    assert recovered.duration_seconds == 15.0
    assert len(recovered.frames) == 2
    assert recovered.frames[1].frame_type == "peak"
    assert recovered.candidate_score == 4.85


# ---------------------------------------------------------------------------
# Test 11: Candidate-to-label deterministic matching
# ---------------------------------------------------------------------------
def test_candidate_to_label_matching() -> None:
    labels = [
        AnnotationRecord(
            sample_id="gt_01",
            source_fingerprint={},
            source_id="v",
            start_seconds=10.0,
            end_seconds=20.0,
            event_category="chaotic",
            editorial_rating=5,
            context_completeness=True,
            setup_presence=True,
            payoff_presence=True,
            approved="y",
        ),
        AnnotationRecord(
            sample_id="gt_02",
            source_fingerprint={},
            source_id="v",
            start_seconds=40.0,
            end_seconds=50.0,
            event_category="fail",
            editorial_rating=3,
            context_completeness=True,
            setup_presence=True,
            payoff_presence=True,
            approved="n",
            rejection_reason="Unexciting stumble",
        ),
    ]

    candidates = [
        # Candidate 1: overlaps gt_01 with high IoU
        {"window_id": "c1", "start_seconds": 11.0, "end_seconds": 19.0, "score": 6.0},
        # Candidate 2: overlaps gt_02 (which was rejected)
        {"window_id": "c2", "start_seconds": 39.0, "end_seconds": 49.0, "score": 5.0},
        # Candidate 3: no overlap with any label
        {"window_id": "c3", "start_seconds": 70.0, "end_seconds": 80.0, "score": 4.0},
    ]

    report = evaluate_candidates(candidates, labels, top_k=3, min_iou=0.20)
    assert report.total_candidates == 3
    assert len(report.matches) == 3

    m1 = report.matches[0]
    assert m1.candidate_window_id == "c1"
    assert m1.matched_sample_id == "gt_01"
    assert m1.match_status == "true_positive"

    m2 = report.matches[1]
    assert m2.candidate_window_id == "c2"
    assert m2.matched_sample_id == "gt_02"
    assert m2.match_status == "false_positive"  # gt_02 was not approved

    m3 = report.matches[2]
    assert m3.candidate_window_id == "c3"
    assert m3.matched_sample_id is None
    assert m3.match_status == "false_positive"


# ---------------------------------------------------------------------------
# Test 12: Precision@K and Recall@K calculations
# ---------------------------------------------------------------------------
def test_precision_recall_at_k() -> None:
    labels = [
        AnnotationRecord(
            sample_id="gt_a",
            source_fingerprint={},
            source_id="v",
            start_seconds=10.0,
            end_seconds=20.0,
            event_category="chaotic",
            editorial_rating=4,
            context_completeness=True,
            setup_presence=True,
            payoff_presence=True,
            approved="y",
        ),
        AnnotationRecord(
            sample_id="gt_b",
            source_fingerprint={},
            source_id="v",
            start_seconds=50.0,
            end_seconds=60.0,
            event_category="impressive",
            editorial_rating=5,
            context_completeness=True,
            setup_presence=True,
            payoff_presence=True,
            approved="y",
        ),
    ]

    # 4 candidates: c1 matches gt_a (TP), c2, c3, c4 match nothing (FP)
    candidates = [
        {"window_id": "c1", "start_seconds": 10.0, "end_seconds": 20.0, "score": 5.0},
        {"window_id": "c2", "start_seconds": 25.0, "end_seconds": 35.0, "score": 4.0},
        {"window_id": "c3", "start_seconds": 70.0, "end_seconds": 80.0, "score": 3.0},
        {"window_id": "c4", "start_seconds": 85.0, "end_seconds": 95.0, "score": 2.0},
    ]

    # Top-2: c1 (TP), c2 (FP). Total approved GT = 2 (gt_a, gt_b).
    # Precision@2 = 1 / 2 = 0.50. Recall@2 = 1 / 2 = 0.50.
    report = evaluate_candidates(candidates, labels, top_k=2, min_iou=0.20)
    assert report.top_k == 2
    assert report.precision_at_k == 0.5
    assert report.recall_at_k == 0.5
    # Weighted recall: gt_a weight=4, gt_b weight=5, total=9. Matched gt_a=4 -> 4/9 ~ 0.4444
    assert round(report.weighted_recall_at_k, 2) == 0.44


# ---------------------------------------------------------------------------
# Test 13: Duplicate prediction handling
# ---------------------------------------------------------------------------
def test_duplicate_prediction_handling() -> None:
    labels = [
        AnnotationRecord(
            sample_id="gt_single",
            source_fingerprint={},
            source_id="v",
            start_seconds=10.0,
            end_seconds=25.0,
            event_category="clutch_kill",
            editorial_rating=5,
            context_completeness=True,
            setup_presence=True,
            payoff_presence=True,
            approved="y",
        )
    ]

    # Two overlapping candidates both overlapping gt_single
    candidates = [
        {"window_id": "c_rank1", "start_seconds": 10.0, "end_seconds": 25.0, "score": 6.0},
        {"window_id": "c_rank2", "start_seconds": 12.0, "end_seconds": 24.0, "score": 5.5},
    ]

    report = evaluate_candidates(candidates, labels, top_k=2, min_iou=0.20)
    assert len(report.matches) == 2

    m1 = report.matches[0]
    assert m1.candidate_window_id == "c_rank1"
    assert m1.match_status == "true_positive"

    m2 = report.matches[1]
    assert m2.candidate_window_id == "c_rank2"
    assert m2.match_status == "duplicate_match"

    assert report.duplicate_rate == 0.5  # 1 out of 2 candidates is a duplicate


# ---------------------------------------------------------------------------
# Test 14: Deterministic evaluation reports (JSON, CSV, Markdown)
# ---------------------------------------------------------------------------
def test_evaluation_reports_generation(tmp_path: Path) -> None:
    labels = [
        AnnotationRecord(
            sample_id="gt_report",
            source_fingerprint={},
            source_id="v",
            start_seconds=0.0,
            end_seconds=10.0,
            event_category="funny",
            editorial_rating=4,
            context_completeness=True,
            setup_presence=True,
            payoff_presence=True,
            approved="y",
        )
    ]
    candidates = [
        {"window_id": "cand_rep", "start_seconds": 0.0, "end_seconds": 10.0, "score": 5.0}
    ]

    report = evaluate_candidates(candidates, labels, top_k=1, min_iou=0.20)
    out_dir = tmp_path / "reports"
    json_path, csv_path, md_path = export_evaluation_artifacts(report, out_dir)

    assert json_path.is_file()
    assert csv_path.is_file()
    assert md_path.is_file()

    # JSON report check
    json_data = json.loads(json_path.read_text(encoding="utf-8"))
    assert json_data["precision_at_k"] == 1.0
    assert json_data["recall_at_k"] == 1.0

    # CSV report check (verify 0.0 is not rendered as empty string)
    csv_text = csv_path.read_text(encoding="utf-8")
    lines = csv_text.strip().splitlines()
    assert len(lines) == 2
    row = lines[1].split(",")
    # candidate_start should be 0.0, label_start should be 0.0
    assert row[2] == "0.0"
    assert row[6] == "0.0"

    # Markdown check
    md_text = md_path.read_text(encoding="utf-8")
    assert "StreamClip AI — Highlight Evaluation Report" in md_text
    assert "Precision@K" in md_text


# ---------------------------------------------------------------------------
# Test 15: Backward compatibility with existing VOD processing
# ---------------------------------------------------------------------------
def test_backward_compatibility_with_existing_vod_processing() -> None:
    # Verify that existing app.vod.evaluate still functions properly
    from app.vod.evaluate import WeakLabel, metrics

    labels = [WeakLabel(start=10.0, duration=5.0, weight=1.0)]
    peaks = [12.0, 25.0]

    res = metrics(peaks, labels, tolerance=1.0)
    assert res.recall_at_k == 1.0
    assert res.precision_at_k == 0.5
    assert res.weighted_recall == 1.0
    assert len(res.missed) == 0


# ---------------------------------------------------------------------------
# Test 16: HighlightJudgeProvider interface and mock provider
# ---------------------------------------------------------------------------
def test_highlight_judge_provider_contract() -> None:
    provider: HighlightJudgeProvider = MockHighlightJudgeProvider(
        default_decision="APPROVE",
        default_rating=4,
        default_category="impressive",
    )
    assert provider.model_name == "mock-highlight-judge"

    dummy_context = CandidateContext(
        schema_version=1,
        candidate_id="c_test",
        source_video="test.mp4",
        source_fingerprint={},
        start_seconds=10.0,
        end_seconds=25.0,
        duration_seconds=15.0,
        peak_seconds=15.0,
        candidate_score=5.0,
        components={},
    )

    judgment = provider.judge_candidate(dummy_context)
    assert judgment.decision == "APPROVE"
    assert judgment.editorial_rating == 4
    assert judgment.event_category == "impressive"
    assert 0.0 <= judgment.confidence <= 1.0

    # Ensure judgment serializes to valid dict
    d = judgment.to_dict()
    assert d["decision"] == "APPROVE"
    assert d["editorial_rating"] == 4
