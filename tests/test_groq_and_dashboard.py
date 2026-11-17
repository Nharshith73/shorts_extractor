"""Unit and integration tests for Groq Provider and Minimalist Web Studio."""

from pathlib import Path
import pytest
from starlette.testclient import TestClient

from app.evaluation.providers.groq import (
    GroqHighlightJudgeProvider,
    GroqProviderConfig,
    resolve_groq_api_key,
)
from app.video.engine import make_provider
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
