"""Stage 4B: Comprehensive Unit and Regression Tests for AI Highlight Judge.

Covers:
1. EvidenceItem schema validation, source constraints, and non-negative timestamp enforcement.
2. AIJudgmentRecord schema validation, normalizers, and boundary constraints.
3. QwenProviderConfig defaults and judge.yaml configuration file parsing.
4. Multimodal prompt construction, transcript inclusion, chat metrics, and base64 frame encoding.
5. Missing audio, missing chat, and missing frame graceful degradation.
6. Self-healing retry policy on malformed JSON responses.
7. Provider server lifecycle management (health check, auto-start, cleanup).
8. Candidate judging workflow (atomic JSON persistence, dual-ranked CSV export).
9. Judgment caching, model fingerprint matching, and force re-evaluation.
10. Target candidate filtering and top-k limits.
11. Opt-in real local Qwen2.5-VL model smoke test.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from app.evaluation.judge import judge_candidates, load_candidate_context
from app.evaluation.models import (
    CandidateContext,
    ChatContext,
    SpeechContext,
    TranscriptSegmentContext,
    TranscriptWordContext,
    VisualFrame,
)
from app.evaluation.providers.qwen import QwenProviderConfig, QwenVisionLanguageProvider
from app.evaluation.schema import (
    VALID_CATEGORIES,
    VALID_EVIDENCE_SOURCES,
    VALID_LAYOUTS,
    AIJudgmentRecord,
    EvidenceItem,
)


def make_dummy_context(
    candidate_id: str,
    source_video: str = "source.mp4",
    start_seconds: float = 10.0,
    end_seconds: float = 20.0,
    peak_seconds: float = 15.0,
    candidate_score: float = 4.0,
    components: dict[str, float] | None = None,
    frames: list[VisualFrame] | None = None,
    speech: SpeechContext | None = None,
    chat: ChatContext | None = None,
) -> CandidateContext:
    """Helper to instantiate a valid CandidateContext for tests."""
    return CandidateContext(
        schema_version=1,
        candidate_id=candidate_id,
        source_video=source_video,
        source_fingerprint={"path": source_video, "size": 1000},
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        duration_seconds=end_seconds - start_seconds,
        peak_seconds=peak_seconds,
        candidate_score=candidate_score,
        components=components or {},
        frames=frames or [],
        speech=speech or SpeechContext(has_speech=False, provenance="none"),
        chat=chat or ChatContext(present=False),
    )


# ---------------------------------------------------------------------------
# Test 1: EvidenceItem Schema & Constraints
# ---------------------------------------------------------------------------
def test_evidence_item_valid() -> None:
    for src in VALID_EVIDENCE_SOURCES:
        item = EvidenceItem(timestamp=14.52, source=src, observation=f"Observed via {src}")
        assert item.timestamp == 14.52
        assert item.source == src
        assert item.observation == f"Observed via {src}"


def test_evidence_item_rejects_negative_or_infinite_timestamp() -> None:
    with pytest.raises(ValueError):
        EvidenceItem(timestamp=-1.0, source="visual", observation="Invalid time")

    with pytest.raises(ValueError):
        EvidenceItem(timestamp=float("inf"), source="visual", observation="Infinite time")


def test_evidence_item_rejects_invalid_source() -> None:
    with pytest.raises(ValueError):
        EvidenceItem(timestamp=5.0, source="rumor", observation="Invalid source tier")  # type: ignore[arg-type]


def test_evidence_item_rejects_empty_observation() -> None:
    with pytest.raises(ValueError):
        EvidenceItem(timestamp=5.0, source="visual", observation="")


# ---------------------------------------------------------------------------
# Test 2: AIJudgmentRecord Schema & Normalizers
# ---------------------------------------------------------------------------
def test_ai_judgment_record_valid() -> None:
    record = AIJudgmentRecord(
        schema_version="1.0",
        candidate_id="cand_001",
        event_category="chaotic",
        summary="A high speed chase resulting in multiple vehicle explosions.",
        editorial_rating=4,
        confidence=0.88,
        context_complete=True,
        setup_present=True,
        payoff_present=True,
        entertainment_factors=["police_pursuit", "vehicle_explosion"],
        evidence=[
            EvidenceItem(timestamp=12.0, source="visual", observation="Cruiser catches fire"),
            EvidenceItem(timestamp=14.5, source="transcript", observation="Player yells 'Look out!'"),
        ],
        hook_suggestion="WATCH THIS ESCAPE!",
        recommended_start_seconds=10.0,
        recommended_end_seconds=22.0,
        recommended_layout="blurred_background",
        rejection_reason=None,
    )
    assert record.is_approved is True
    assert record.recommended_duration == 12.0
    data = record.to_dict()
    assert data["candidate_id"] == "cand_001"
    assert len(data["evidence"]) == 2
    assert data["editorial_rating"] == 4


def test_ai_judgment_record_normalizers() -> None:
    # 1. Null hook_suggestion normalizes to empty string
    record = AIJudgmentRecord(
        candidate_id="cand_002",
        event_category="UNKNOWN_CATEGORY",
        summary="Routine driving with no significant action occurring.",
        editorial_rating=1,
        confidence=0.95,
        context_complete=True,
        setup_present=False,
        payoff_present=False,
        entertainment_factors=["routine_transit"],
        evidence=[EvidenceItem(timestamp=5.0, source="visual", observation="Driving straight")],
        hook_suggestion=None,  # type: ignore[arg-type]
        recommended_start_seconds=0.0,
        recommended_end_seconds=10.0,
        recommended_layout="static_layout",
        rejection_reason=None,
    )
    assert record.hook_suggestion == ""
    assert record.event_category == "unknown"
    assert record.recommended_layout == "blurred_background"
    assert record.is_approved is False
    # Auto-populated rejection reason for rating <= 2
    assert record.rejection_reason is not None


def test_ai_judgment_record_layout_variations() -> None:
    # Center crop and stacked variants
    rec_crop = AIJudgmentRecord(
        candidate_id="cand_003",
        event_category="funny",
        summary="Funny encounter with NPC.",
        editorial_rating=3,
        confidence=0.7,
        context_complete=True,
        setup_present=True,
        payoff_present=True,
        evidence=[EvidenceItem(timestamp=2.0, source="inference", observation="Funny dialog")],
        recommended_start_seconds=1.0,
        recommended_end_seconds=6.0,
        recommended_layout="vertical_crop",
    )
    assert rec_crop.recommended_layout == "center_crop"

    rec_stack = AIJudgmentRecord(
        candidate_id="cand_004",
        event_category="skillful",
        summary="Precision sniper shot while moving.",
        editorial_rating=5,
        confidence=0.9,
        context_complete=True,
        setup_present=True,
        payoff_present=True,
        evidence=[EvidenceItem(timestamp=4.0, source="visual", observation="Headshot hit marker")],
        recommended_start_seconds=2.0,
        recommended_end_seconds=7.0,
        recommended_layout="stacked_layout",
    )
    assert rec_stack.recommended_layout == "stacked"


def test_ai_judgment_record_interval_rejection() -> None:
    with pytest.raises(ValueError, match="recommended_end_seconds"):
        AIJudgmentRecord(
            candidate_id="cand_invalid",
            event_category="funny",
            summary="Invalid timestamps provided.",
            editorial_rating=3,
            confidence=0.8,
            context_complete=True,
            setup_present=True,
            payoff_present=True,
            evidence=[EvidenceItem(timestamp=10.0, source="visual", observation="Frame")],
            recommended_start_seconds=15.0,
            recommended_end_seconds=10.0,  # End before start
        )


# ---------------------------------------------------------------------------
# Test 3: Configuration Loading & judge.yaml
# ---------------------------------------------------------------------------
def test_qwen_provider_config_defaults() -> None:
    config = QwenProviderConfig()
    assert config.n_ctx == 6144
    assert config.n_gpu_layers == 99
    assert config.max_frames == 8
    assert config.max_retries == 2
    assert config.auto_start_server is True


def test_judge_yaml_parsing() -> None:
    import yaml

    judge_yaml_path = Path("judge.yaml")
    assert judge_yaml_path.is_file()

    with judge_yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert "judge" in data
    j = data["judge"]
    assert j["provider"] == "local-qwen"
    assert j["n_ctx"] == 6144
    assert j["n_gpu_layers"] == 99
    assert j["quantization"] == "Q4_K_M"
    assert j["license"] == "Apache-2.0"


# ---------------------------------------------------------------------------
# Test 4: Multimodal Prompt Construction & Formatting
# ---------------------------------------------------------------------------
def test_build_prompt_messages(tmp_path: Path) -> None:
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    f1 = frame_dir / "f1.jpg"
    f2 = frame_dir / "f2.jpg"
    f1.write_bytes(b"\xff\xd8\xff\xd9_dummy_frame_1")
    f2.write_bytes(b"\xff\xd8\xff\xd9_dummy_frame_2")

    ctx = make_dummy_context(
        candidate_id="test_cand_prompt",
        start_seconds=10.0,
        end_seconds=20.0,
        peak_seconds=15.0,
        candidate_score=4.5,
        components={"audio_energy": 2.5, "motion": 3.1},
        frames=[
            VisualFrame(timestamp=10.0, frame_path=str(f1), frame_type="boundary_start", width=640, height=360),
            VisualFrame(timestamp=15.0, frame_path=str(f2), frame_type="peak", width=640, height=360),
        ],
        speech=SpeechContext(
            has_speech=True,
            segments=[
                TranscriptSegmentContext(
                    text="Get to the chopper!",
                    start=11.0,
                    end=13.0,
                    avg_logprob=-0.2,
                    no_speech_prob=0.01,
                    words=(
                        TranscriptWordContext(word="Get", start=11.0, end=11.5),
                        TranscriptWordContext(word="to", start=11.5, end=11.8),
                        TranscriptWordContext(word="the", start=11.8, end=12.2),
                        TranscriptWordContext(word="chopper", start=12.2, end=13.0),
                    ),
                )
            ],
        ),
        chat=ChatContext(
            present=True,
            message_count=42,
            unique_chatters=18,
            emote_density=0.65,
        ),
    )

    provider = QwenVisionLanguageProvider(QwenProviderConfig())
    messages = provider._build_prompt_messages(ctx)

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "StreamClip AI" in messages[0]["content"]
    assert "visual" in messages[0]["content"]
    assert "transcript" in messages[0]["content"]
    assert "chat" in messages[0]["content"]

    user_content = messages[1]["content"]
    assert isinstance(user_content, list)
    text_blobs = [item["text"] for item in user_content if item.get("type") == "text"]
    full_text = " ".join(text_blobs)

    assert "Get to the chopper!" in full_text
    assert "42 messages" in full_text
    assert "emote density: 0.65" in full_text
    assert "audio_energy=2.50" in full_text
    assert "motion=3.10" in full_text

    # Verify images were embedded
    images = [item for item in user_content if item.get("type") == "image_url"]
    assert len(images) == 2
    assert images[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_build_prompt_messages_missing_modalities() -> None:
    ctx = make_dummy_context(
        candidate_id="sparse_cand",
        start_seconds=0.0,
        end_seconds=10.0,
        peak_seconds=5.0,
        candidate_score=1.2,
        frames=[
            VisualFrame(timestamp=2.0, frame_path="non_existent_frame.jpg", frame_type="peak", width=640, height=360)
        ],
        speech=SpeechContext(has_speech=False),
        chat=ChatContext(present=False),
    )

    provider = QwenVisionLanguageProvider(QwenProviderConfig())
    messages = provider._build_prompt_messages(ctx)

    user_content = messages[1]["content"]
    text_blobs = [item["text"] for item in user_content if item.get("type") == "text"]
    full_text = " ".join(text_blobs)

    assert "(No dialogue detected)" in full_text
    assert "(No chat data)" in full_text
    # Missing frame does not throw an exception; it logs a warning and skips the image
    images = [item for item in user_content if item.get("type") == "image_url"]
    assert len(images) == 0


# ---------------------------------------------------------------------------
# Test 5: Self-Healing Retry Loop
# ---------------------------------------------------------------------------
def test_judge_candidate_record_retry_recovery() -> None:
    ctx = make_dummy_context(candidate_id="retry_cand")

    provider = QwenVisionLanguageProvider(QwenProviderConfig(max_retries=3))
    provider.ensure_ready = MagicMock()  # Mock server readiness

    # Attempt 1: Malformed JSON syntax
    # Attempt 2: Valid JSON
    valid_record_json = json.dumps({
        "schema_version": "1.0",
        "candidate_id": "retry_cand",
        "event_category": "dramatic",
        "summary": "Close call near the intersection.",
        "editorial_rating": 4,
        "confidence": 0.85,
        "context_complete": True,
        "setup_present": True,
        "payoff_present": True,
        "entertainment_factors": ["near_miss"],
        "evidence": [{"timestamp": 15.0, "source": "visual", "observation": "Swerved to avoid truck"}],
        "hook_suggestion": "ALMOST DIED HERE!",
        "recommended_start_seconds": 10.0,
        "recommended_end_seconds": 20.0,
        "recommended_layout": "blurred_background",
        "rejection_reason": None,
    })

    side_effects = [
        ("{ malformed json ...", {"prompt_tokens": 100, "completion_tokens": 10}),
        (valid_record_json, {"prompt_tokens": 120, "completion_tokens": 80}),
    ]

    with patch.object(provider, "_call_completion", side_effect=side_effects):
        record = provider.judge_candidate_record(ctx)

    assert record.candidate_id == "retry_cand"
    assert record.editorial_rating == 4
    assert record.model_metadata["attempt"] == 2
    assert record.is_approved is True


def test_judge_candidate_record_exceeds_retries() -> None:
    ctx = make_dummy_context(candidate_id="fail_cand")

    provider = QwenVisionLanguageProvider(QwenProviderConfig(max_retries=2))
    provider.ensure_ready = MagicMock()

    side_effects = [
        ("invalid json 1", {"prompt_tokens": 50, "completion_tokens": 5}),
        ("invalid json 2", {"prompt_tokens": 50, "completion_tokens": 5}),
    ]

    with patch.object(provider, "_call_completion", side_effect=side_effects):
        with pytest.raises(RuntimeError, match="Failed to generate valid AIJudgmentRecord"):
            provider.judge_candidate_record(ctx)


# ---------------------------------------------------------------------------
# Test 6: Server Lifecycle & Readiness
# ---------------------------------------------------------------------------
def test_server_is_running_check() -> None:
    provider = QwenVisionLanguageProvider(QwenProviderConfig())

    with patch.object(provider._http_client, "get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        assert provider.is_server_running() is True

    with patch.object(provider._http_client, "get", side_effect=Exception("Connection refused")):
        assert provider.is_server_running() is False


def test_server_start_missing_binary() -> None:
    cfg = QwenProviderConfig(
        server_binary=Path("non_existent_llama_server.exe"),
        model_path=Path("non_existent_model.gguf"),
    )
    provider = QwenVisionLanguageProvider(cfg)
    provider.is_server_running = MagicMock(return_value=False)

    with pytest.raises(FileNotFoundError, match="llama-server binary not found"):
        provider.start_server()


def test_server_close_terminates_process() -> None:
    provider = QwenVisionLanguageProvider(QwenProviderConfig())
    mock_proc = MagicMock()
    provider._server_process = mock_proc

    provider.close()
    mock_proc.terminate.assert_called_once()
    assert provider._server_process is None


# ---------------------------------------------------------------------------
# Test 7: Judge Candidates Workflow (Persistence & Dual-Ranking CSV)
# ---------------------------------------------------------------------------
def test_judge_candidates_workflow(tmp_path: Path) -> None:
    vod_dir = tmp_path / "vod_test"
    vod_dir.mkdir()

    # Create clips.json with 2 candidates
    clips = [
        {
            "window_id": "cand_alpha",
            "start_seconds": 10.0,
            "end_seconds": 25.0,
            "peak_seconds": 18.0,
            "score": 4.5,
            "unclipped_score": 5.2,
            "video": str(vod_dir / "vod.mp4"),
        },
        {
            "window_id": "cand_beta",
            "start_seconds": 30.0,
            "end_seconds": 45.0,
            "peak_seconds": 38.0,
            "score": 3.2,
            "unclipped_score": 3.8,
            "video": str(vod_dir / "vod.mp4"),
        },
    ]
    (vod_dir / "clips.json").write_text(json.dumps(clips), encoding="utf-8")

    # Create dummy context files for both
    for cid in ("cand_alpha", "cand_beta"):
        ctx_dir = vod_dir / "context" / cid
        ctx_dir.mkdir(parents=True)
        ctx = make_dummy_context(
            candidate_id=cid,
            source_video=str(vod_dir / "vod.mp4"),
            start_seconds=10.0,
            end_seconds=25.0,
            peak_seconds=18.0,
        )
        (ctx_dir / "context.json").write_text(json.dumps(ctx.to_dict()), encoding="utf-8")

    # Mock provider that gives cand_beta a higher AI editorial rating than cand_alpha
    # to verify dual-ranking inversion (signal rank vs AI rank)
    mock_provider = MagicMock()
    mock_provider.model_name = "MockJudgeModel"

    def mock_judge(ctx: CandidateContext, timeout_seconds: float = 60.0) -> AIJudgmentRecord:
        if ctx.candidate_id == "cand_alpha":
            # Signal rank #1, but AI rejects it
            return AIJudgmentRecord(
                candidate_id="cand_alpha",
                event_category="ordinary",
                summary="Routine gameplay with minimal interest.",
                editorial_rating=1,
                confidence=0.9,
                context_complete=True,
                setup_present=False,
                payoff_present=False,
                evidence=[EvidenceItem(timestamp=18.0, source="visual", observation="Driving slowly")],
                recommended_start_seconds=10.0,
                recommended_end_seconds=25.0,
                model_metadata={"model": "MockJudgeModel", "latency_seconds": 1.2},
            )
        else:
            # Signal rank #2, but AI rates it 5 stars
            return AIJudgmentRecord(
                candidate_id="cand_beta",
                event_category="chaotic",
                summary="Massive multi-car pileup with huge explosion.",
                editorial_rating=5,
                confidence=0.95,
                context_complete=True,
                setup_present=True,
                payoff_present=True,
                evidence=[EvidenceItem(timestamp=38.0, source="visual", observation="Explosion")],
                recommended_start_seconds=30.0,
                recommended_end_seconds=45.0,
                model_metadata={"model": "MockJudgeModel", "latency_seconds": 1.5},
            )

    mock_provider.judge_candidate_record.side_effect = mock_judge

    records, csv_path = judge_candidates(vod_dir=vod_dir, provider=mock_provider)

    assert len(records) == 2
    assert csv_path.is_file()

    # Verify JSON persistence
    alpha_json = vod_dir / "judgments" / "cand_alpha.json"
    beta_json = vod_dir / "judgments" / "cand_beta.json"
    assert alpha_json.is_file()
    assert beta_json.is_file()

    # Read and verify CSV dual rankings
    with csv_path.open("r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) == 2
    # First row in AI ranking should be cand_beta (rating 5)
    row_1 = reader[0]
    assert row_1["candidate_id"] == "cand_beta"
    assert row_1["ai_rank"] == "1"
    assert row_1["signal_rank"] == "2"
    assert row_1["ai_editorial_rating"] == "5"
    assert row_1["ai_approved"] == "y"

    # Second row should be cand_alpha (rating 1)
    row_2 = reader[1]
    assert row_2["candidate_id"] == "cand_alpha"
    assert row_2["ai_rank"] == "2"
    assert row_2["signal_rank"] == "1"
    assert row_2["ai_editorial_rating"] == "1"
    assert row_2["ai_approved"] == "n"


# ---------------------------------------------------------------------------
# Test 8: Judgment Caching & Force Re-Evaluation
# ---------------------------------------------------------------------------
def test_judge_candidates_caching_and_force(tmp_path: Path) -> None:
    vod_dir = tmp_path / "vod_cache_test"
    vod_dir.mkdir()

    clips = [
        {
            "window_id": "cand_cached",
            "start_seconds": 5.0,
            "end_seconds": 15.0,
            "score": 3.0,
            "video": str(vod_dir / "vod.mp4"),
        }
    ]
    (vod_dir / "clips.json").write_text(json.dumps(clips), encoding="utf-8")

    ctx_dir = vod_dir / "context" / "cand_cached"
    ctx_dir.mkdir(parents=True)
    ctx = make_dummy_context(
        candidate_id="cand_cached",
        source_video=str(vod_dir / "vod.mp4"),
        start_seconds=5.0,
        end_seconds=15.0,
        peak_seconds=10.0,
    )
    (ctx_dir / "context.json").write_text(json.dumps(ctx.to_dict()), encoding="utf-8")

    mock_provider = MagicMock()
    mock_provider.model_name = "CachedModelV1"

    rec = AIJudgmentRecord(
        candidate_id="cand_cached",
        event_category="interaction",
        summary="Initial run summary.",
        editorial_rating=3,
        confidence=0.8,
        context_complete=True,
        setup_present=True,
        payoff_present=True,
        evidence=[EvidenceItem(timestamp=10.0, source="visual", observation="Initial run")],
        recommended_start_seconds=5.0,
        recommended_end_seconds=15.0,
        model_metadata={"model": "CachedModelV1", "latency_seconds": 1.0},
    )
    mock_provider.judge_candidate_record.return_value = rec

    # Run 1: Should call provider
    records1, _ = judge_candidates(vod_dir=vod_dir, provider=mock_provider)
    assert mock_provider.judge_candidate_record.call_count == 1
    assert len(records1) == 1

    # Run 2: Without force, should reuse cache and NOT call provider
    records2, _ = judge_candidates(vod_dir=vod_dir, provider=mock_provider, force=False)
    assert mock_provider.judge_candidate_record.call_count == 1
    assert len(records2) == 1
    assert records2[0].summary == "Initial run summary."

    # Run 3: With force=True, should re-invoke provider
    updated_rec = AIJudgmentRecord(
        candidate_id="cand_cached",
        event_category="interaction",
        summary="Forced re-evaluation summary.",
        editorial_rating=4,
        confidence=0.9,
        context_complete=True,
        setup_present=True,
        payoff_present=True,
        evidence=[EvidenceItem(timestamp=10.0, source="visual", observation="Forced rerun")],
        recommended_start_seconds=5.0,
        recommended_end_seconds=15.0,
        model_metadata={"model": "CachedModelV1", "latency_seconds": 1.1},
    )
    mock_provider.judge_candidate_record.return_value = updated_rec
    records3, _ = judge_candidates(vod_dir=vod_dir, provider=mock_provider, force=True)
    assert mock_provider.judge_candidate_record.call_count == 2
    assert records3[0].summary == "Forced re-evaluation summary."


# ---------------------------------------------------------------------------
# Test 9: Candidate Filtering and Top-K Limits
# ---------------------------------------------------------------------------
def test_judge_candidates_filtering(tmp_path: Path) -> None:
    vod_dir = tmp_path / "vod_filter_test"
    vod_dir.mkdir()

    clips = [
        {"window_id": "c1", "start_seconds": 0.0, "end_seconds": 10.0, "score": 5.0},
        {"window_id": "c2", "start_seconds": 10.0, "end_seconds": 20.0, "score": 4.0},
        {"window_id": "c3", "start_seconds": 20.0, "end_seconds": 30.0, "score": 3.0},
    ]
    (vod_dir / "clips.json").write_text(json.dumps(clips), encoding="utf-8")

    for cid in ("c1", "c2", "c3"):
        ctx_dir = vod_dir / "context" / cid
        ctx_dir.mkdir(parents=True)
        ctx = make_dummy_context(
            candidate_id=cid,
            start_seconds=0.0,
            end_seconds=10.0,
            peak_seconds=5.0,
        )
        (ctx_dir / "context.json").write_text(json.dumps(ctx.to_dict()), encoding="utf-8")

    mock_provider = MagicMock()
    mock_provider.model_name = "FilterTestModel"
    mock_provider.judge_candidate_record.side_effect = lambda ctx, **kw: AIJudgmentRecord(
        candidate_id=ctx.candidate_id,
        event_category="funny",
        summary="Sample clip",
        editorial_rating=3,
        confidence=0.8,
        context_complete=True,
        setup_present=True,
        payoff_present=True,
        evidence=[EvidenceItem(timestamp=5.0, source="visual", observation="Action")],
        recommended_start_seconds=0.0,
        recommended_end_seconds=10.0,
        model_metadata={"model": "FilterTestModel"},
    )

    # Top-K = 1
    recs_top1, _ = judge_candidates(vod_dir=vod_dir, provider=mock_provider, top_k=1)
    assert len(recs_top1) == 1
    assert recs_top1[0].candidate_id == "c1"

    # Specific candidate_id filter
    recs_c2, _ = judge_candidates(vod_dir=vod_dir, provider=mock_provider, candidate_id="c2")
    assert len(recs_c2) == 1
    assert recs_c2[0].candidate_id == "c2"


# ---------------------------------------------------------------------------
# Test 10: Opt-in Real Model Smoke Test
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not (
        Path("D:/tools/llama.cpp/llama-server.exe").is_file()
        and Path("D:/models/qwen2.5-vl-7b-gguf/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf").is_file()
    ),
    reason="Local Qwen2.5-VL weights or llama-server not present",
)
def test_real_qwen_inference_smoke() -> None:
    """Smoke test running real inference against local llama.cpp server if running."""
    provider = QwenVisionLanguageProvider(QwenProviderConfig())
    if not provider.is_server_running():
        pytest.skip("llama-server is not currently running at localhost:8080")

    # Use existing candidate context from synthetic VOD if available
    ctx_path = Path("data/synthetic_vod/output/context/2aae2654e2ecfe1a8dc53ede/context.json")
    if not ctx_path.is_file():
        pytest.skip("Synthetic candidate context not available for smoke test")

    data = json.loads(ctx_path.read_text(encoding="utf-8"))
    ctx = CandidateContext.from_dict(data)

    record = provider.judge_candidate_record(ctx, timeout_seconds=30.0)
    assert record.candidate_id == ctx.candidate_id
    assert record.editorial_rating in range(0, 6)
    assert record.confidence >= 0.0 and record.confidence <= 1.0
    assert len(record.evidence) >= 1
    assert record.model_metadata.get("tokens_per_second", 0) > 0.0
