"""Unit and integration tests for Groq Provider and Minimalist Web Studio."""

from pathlib import Path
import pytest
from starlette.testclient import TestClient

from app.evaluation.providers.groq import (
    GroqHighlightJudgeProvider,
    GroqProviderConfig,
    parse_header_time,
    parse_wait_seconds,
    resolve_groq_api_key,
)
from app.evaluation.schema import AIJudgmentRecord
from app.video.engine import make_provider, select
from app.web.server import app


def test_groq_provider_initialization():
    config = GroqProviderConfig(api_key="gsk_test_fake_api_key_1234567890")
    provider = GroqHighlightJudgeProvider(config)
    assert provider.model_name == "openai/gpt-oss-20b"
    assert provider.has_valid_key() is True

    # Custom model config
    config_custom = GroqProviderConfig(
        api_key="gsk_test_fake_api_key_1234567890",
        model_name="openai/gpt-oss-120b",
        use_vision=False,
    )
    provider_custom = GroqHighlightJudgeProvider(config_custom)
    assert provider_custom.model_name == "openai/gpt-oss-120b"


def test_groq_provider_missing_key():
    config = GroqProviderConfig(api_key="")
    provider = GroqHighlightJudgeProvider(config)
    assert provider.has_valid_key() is False
    with pytest.raises(ValueError, match="GROQ_API_KEY is not set"):
        # Dummy context with mock
        from app.evaluation.models import CandidateContext, SpeechContext, ChatContext
        ctx = CandidateContext(
            schema_version=1,
            candidate_id="c1",
            source_video=Path("dummy.mp4"),
            source_fingerprint={},
            start_seconds=10.0,
            end_seconds=25.0,
            duration_seconds=15.0,
            peak_seconds=15.0,
            candidate_score=12.0,
            components={},
            speech=SpeechContext(has_speech=False, segments=[]),
            chat=ChatContext(present=False),
        )
        provider.judge_candidate_record(ctx)


def test_make_provider_supports_groq():
    provider = make_provider("judge.yaml", mock=True)
    assert provider.model_name == "mock-highlight-judge"

    groq_provider = make_provider("judge.yaml", provider_type="groq")
    assert isinstance(groq_provider, GroqHighlightJudgeProvider)


def test_web_studio_endpoints():
    client = TestClient(app)

    # 1. Index page
    r = client.get("/")
    assert r.status_code == 200
    assert "StreamClip Studio" in r.text
    assert "#ffffff" in r.text

    # 2. Videos list
    r_videos = client.get("/api/videos")
    assert r_videos.status_code == 200
    assert "videos" in r_videos.json()

    # 3. Jobs list
    r_jobs = client.get("/api/jobs")
    assert r_jobs.status_code == 200
    assert "jobs" in r_jobs.json()

    # 4. Settings
    r_settings = client.get("/api/settings")
    assert r_settings.status_code == 200
    assert "groq_configured" in r_settings.json()


def test_groq_rate_limit_parsing():
    # Test parse_header_time
    assert parse_header_time("5.437s") == 5.437
    assert parse_header_time("832.5ms") == 0.8325
    assert parse_header_time("1m12s") == 72.0
    assert parse_header_time("8m38.4s") == 518.4
    assert parse_header_time("2.5") == 2.5
    assert parse_header_time("") == 0.0

    # Test parse_wait_seconds from error message
    err1 = (
        "Rate limit reached for model `openai/gpt-oss-20b` on tokens per minute (TPM): "
        "Limit 8000, Used 6632, Requested 1479. Please try again in 832.5ms. Need more tokens?"
    )
    assert parse_wait_seconds(err1) == 0.8325

    err2 = "Please try again in 4.5s. Need more tokens?"
    assert parse_wait_seconds(err2) == 4.5

    err3 = "Please try again in 1m15s."
    assert parse_wait_seconds(err3) == 75.0


def test_groq_rate_limit_transparent_retry(monkeypatch):
    config = GroqProviderConfig(api_key="gsk_fake_key_1234567890", max_retries=2)
    provider = GroqHighlightJudgeProvider(config)

    calls = 0

    class MockResponse:
        def __init__(self, status_code, json_data, headers=None, text=""):
            self.status_code = status_code
            self._json = json_data
            self.headers = headers or {}
            self.text = text

        def json(self):
            return self._json

    def mock_post(url, headers=None, json=None, timeout=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            # First call: return 429
            return MockResponse(
                429,
                {},
                headers={"x-ratelimit-reset-tokens": "0.01s"},
                text="Rate limit reached. Please try again in 10ms.",
            )
        # Second call: return 200
        return MockResponse(
            200,
            {
                "choices": [{"message": {"content": '{"summary": "funny clip", "editorial_rating": 4}'}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            },
            headers={"x-ratelimit-remaining-tokens": "7000"},
        )

    monkeypatch.setattr(provider._http_client, "post", mock_post)
    content, telemetry = provider._call_completion([{"role": "user", "content": "test"}], timeout_seconds=10.0)
    assert calls == 2
    assert "funny clip" in content
    assert telemetry["prompt_tokens"] == 100


def test_select_honors_min_shorts_guarantee():
    candidates = [
        {"window_id": "c1", "start_seconds": 10.0, "end_seconds": 20.0},
        {"window_id": "c2", "start_seconds": 30.0, "end_seconds": 40.0},
        {"window_id": "c3", "start_seconds": 50.0, "end_seconds": 60.0},
    ]

    # Only c1 has editorial_rating >= 4; c2 and c3 have rating 2
    from app.evaluation.schema import EvidenceItem
    records = [
        AIJudgmentRecord(
            schema_version="1.0",
            candidate_id="c1",
            event_category="funny",
            summary="top moment",
            editorial_rating=5,
            confidence=0.9,
            context_complete=True,
            setup_present=True,
            payoff_present=True,
            entertainment_factors=[],
            evidence=[EvidenceItem(timestamp=15.0, source="visual", observation="action")],
            recommended_start_seconds=10.0,
            recommended_end_seconds=20.0,
        ),
        AIJudgmentRecord(
            schema_version="1.0",
            candidate_id="c2",
            event_category="ordinary",
            summary="secondary moment 1",
            editorial_rating=2,
            confidence=0.7,
            context_complete=True,
            setup_present=True,
            payoff_present=True,
            entertainment_factors=[],
            evidence=[EvidenceItem(timestamp=35.0, source="visual", observation="transit")],
            recommended_start_seconds=30.0,
            recommended_end_seconds=40.0,
        ),
        AIJudgmentRecord(
            schema_version="1.0",
            candidate_id="c3",
            event_category="ordinary",
            summary="secondary moment 2",
            editorial_rating=2,
            confidence=0.6,
            context_complete=True,
            setup_present=True,
            payoff_present=True,
            entertainment_factors=[],
            evidence=[EvidenceItem(timestamp=55.0, source="visual", observation="transit")],
            recommended_start_seconds=50.0,
            recommended_end_seconds=60.0,
        ),
    ]

    # If min_shorts=1, only c1 is selected
    selected_1 = select(
        candidates=candidates,
        records=records,
        max_shorts=3,
        minimum=4,
        overlap=0.3,
        max_duration=60.0,
        require_context=False,
        min_shorts=1,
    )
    assert len(selected_1) == 1
    assert selected_1[0]["window_id"] == "c1"

    # If min_shorts=3, selection relaxes threshold to guarantee 3 shorts
    selected_3 = select(
        candidates=candidates,
        records=records,
        max_shorts=3,
        minimum=4,
        overlap=0.3,
        max_duration=60.0,
        require_context=False,
        min_shorts=3,
    )
    assert len(selected_3) == 3
    assert [c["window_id"] for c in selected_3] == ["c1", "c2", "c3"]


def test_snap_to_speech_boundaries():
    from app.video.boundaries import snap_to_speech_boundaries

    segments = [
        {"start": 10.5, "end": 14.2, "text": "Wait, did you see that car?"},
        {"start": 14.5, "end": 18.0, "text": "It literally flew off the cliff!"},
        {"start": 18.2, "end": 22.0, "text": "Hahaha, no way!"},
    ]

    # Raw candidate was 13.0 to 20.0
    s_start, s_end = snap_to_speech_boundaries(
        start=13.0,
        end=20.0,
        peak=16.0,
        segments=segments,
        min_duration=10.0,
        max_duration=30.0,
        max_lookback=5.0,
        max_lookforward=5.0,
    )
    assert s_start == 10.5
    assert s_end == 22.0


def test_web_studio_dashboard_full_management(tmp_path, monkeypatch):
    from starlette.testclient import TestClient
    from app.web import server

    # Point INBOX_DIR and DATA_DIR to isolated tmp_path
    monkeypatch.setattr(server, "INBOX_DIR", tmp_path / "inbox")
    monkeypatch.setattr(server, "COMPLETED_DIR", tmp_path / "completed")
    monkeypatch.setattr(server, "PROCESSING_DIR", tmp_path / "processing")
    monkeypatch.setattr(server, "DATA_DIR", tmp_path)

    (tmp_path / "inbox").mkdir(parents=True, exist_ok=True)
    (tmp_path / "completed").mkdir(parents=True, exist_ok=True)
    (tmp_path / "processing").mkdir(parents=True, exist_ok=True)

    client = TestClient(server.app)

    # 1. HTML Dashboard Check
    r_page = client.get("/")
    assert r_page.status_code == 200
    assert "Storage & Reset" in r_page.text
    assert "dropzone" in r_page.text
    assert "layout-select" in r_page.text
    assert "encoder-select" in r_page.text
    assert "Copy Hook" in r_page.text

    # 2. Upload file
    dummy_content = b"RIFF....AVI"
    r_upload = client.post(
        "/api/upload",
        files={"file": ("test_clip.mp4", dummy_content, "video/mp4")}
    )
    assert r_upload.status_code == 200
    assert r_upload.json()["status"] == "UPLOADED"
    assert (tmp_path / "inbox" / "test_clip.mp4").is_file()

    # Verify listed in videos
    r_videos = client.get("/api/videos")
    assert any(v["filename"] == "test_clip.mp4" for v in r_videos.json()["videos"])

    # 3. Delete single video
    r_del = client.delete("/api/videos/test_clip.mp4")
    assert r_del.status_code == 200
    assert not (tmp_path / "inbox" / "test_clip.mp4").exists()

    # 4. Upload multiple and test delete all videos
    for name in ["vid1.mp4", "vid2.mp4"]:
        (tmp_path / "inbox" / name).write_bytes(b"dummy")
    r_del_all = client.delete("/api/videos")
    assert r_del_all.status_code == 200
    assert len(r_del_all.json()["deleted"]) == 2
    assert len(list((tmp_path / "inbox").iterdir())) == 0

    # 5. Test Delete Short
    test_short = tmp_path / "completed" / "short1.mp4"
    test_short.write_bytes(b"video bytes")
    r_del_short = client.delete(f"/api/shorts?path={test_short}")
    assert r_del_short.status_code == 200
    assert not test_short.exists()

    # 6. Test Tasks Dismiss and Stop
    server.ACTIVE_TASKS["task1"] = {"status": "FAILED", "logs": ["err"]}
    r_dismiss = client.post("/api/tasks/dismiss")
    assert r_dismiss.status_code == 200
    assert len(server.ACTIVE_TASKS) == 0

    # 7. Test Save Groq Key
    r_key = client.post("/api/settings/groq-key?key=gsk_valid_test_key_1234567890")
    assert r_key.status_code == 200
    assert r_key.json()["configured"] is True

    # 8. Test Cleanup Endpoint
    (tmp_path / "processing" / "temp_run").mkdir()
    (tmp_path / "processing" / "temp_run" / "chunk.mp4").write_bytes(b"test")
    r_clean = client.post("/api/cleanup", json={"clear_processing": True, "clear_completed": False, "clear_inbox": False, "reset_db": False})
    assert r_clean.status_code == 200
    assert "processing_temp_files" in r_clean.json()["cleaned"]
    assert not (tmp_path / "processing" / "temp_run").exists()


