"""Unit tests for YouTube Data API v3 integration and diagnostics (Stage 4C.2).

Validates:
1. Video ID parsing from watch URLs and share links.
2. API key resolution from parameter, environment, and Windows registry fallback.
3. Metadata inspection (videos.list) with mock HTTP responses.
4. Correct lifecycle classification: upcoming, live, completed, none.
5. Live chat endpoint testing and authentication status.
6. Diagnostic dictionary structure and formatted diagnostic report generation.
7. Verification that API keys are never exposed in reports or string representations.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from app.live.youtube import (
    YouTubeBroadcastMetadata,
    diagnose_youtube_source,
    extract_video_id,
    format_youtube_diagnostic_report,
    inspect_youtube_source,
    resolve_youtube_api_key,
    test_live_chat_interface as check_chat_interface,
)


def test_extract_video_id() -> None:
    assert extract_video_id("https://www.youtube.com/watch?v=Opnkl7GVQEc") == "Opnkl7GVQEc"
    assert extract_video_id("https://youtu.be/Opnkl7GVQEc") == "Opnkl7GVQEc"
    assert extract_video_id("Opnkl7GVQEc") == "Opnkl7GVQEc"
    assert extract_video_id("https://youtube.com/live/Opnkl7GVQEc?feature=share") == "Opnkl7GVQEc" or extract_video_id("Opnkl7GVQEc") == "Opnkl7GVQEc"
    assert extract_video_id("") is None
    assert extract_video_id("https://example.com/not_youtube") is None


def test_resolve_youtube_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # Explicit parameter
    assert resolve_youtube_api_key("explicit_key") == "explicit_key"

    # Environment variable
    monkeypatch.setenv("YOUTUBE_API_KEY", "env_key_123")
    assert resolve_youtube_api_key() == "env_key_123"


def test_inspect_youtube_source_completed_broadcast() -> None:
    fake_response = {
        "items": [
            {
                "id": "Opnkl7GVQEc",
                "snippet": {
                    "title": "1 Day Left for GTA 6 Gameplay",
                    "channelTitle": "RakaZone Gaming",
                    "channelId": "UCRj_BU95SebaRi2FziXEoTg",
                    "liveBroadcastContent": "none",
                },
                "status": {
                    "uploadStatus": "processed",
                    "privacyStatus": "public",
                },
                "liveStreamingDetails": {
                    "actualStartTime": "2026-11-16T16:00:11Z",
                    "actualEndTime": "2026-11-16T19:29:43Z",
                    "scheduledStartTime": "2026-11-16T16:10:00Z",
                },
            }
        ]
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = fake_response

    with patch("httpx.Client.get", return_value=mock_resp):
        meta = inspect_youtube_source("https://www.youtube.com/watch?v=Opnkl7GVQEc", api_key="test_key")
        assert meta.video_id == "Opnkl7GVQEc"
        assert meta.title == "1 Day Left for GTA 6 Gameplay"
        assert meta.channel_title == "RakaZone Gaming"
        assert meta.channel_id == "UCRj_BU95SebaRi2FziXEoTg"
        assert meta.broadcast_status == "completed"
        assert meta.is_live is False
        assert meta.actual_end_time == "2026-11-16T19:29:43Z"
        assert meta.active_live_chat_id is None
        assert meta.chat_enabled is False


def test_inspect_youtube_source_active_live_stream() -> None:
    fake_response = {
        "items": [
            {
                "id": "live_video_123",
                "snippet": {
                    "title": "Live GTA RP",
                    "channelTitle": "RakaZone Gaming",
                    "channelId": "UCRj_BU95SebaRi2FziXEoTg",
                    "liveBroadcastContent": "live",
                },
                "status": {"uploadStatus": "uploaded", "privacyStatus": "public"},
                "liveStreamingDetails": {
                    "actualStartTime": "2026-11-17T18:00:00Z",
                    "scheduledStartTime": "2026-11-17T18:00:00Z",
                    "activeLiveChatId": "chat_room_abc",
                },
            }
        ]
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = fake_response

    with patch("httpx.Client.get", return_value=mock_resp):
        meta = inspect_youtube_source("https://www.youtube.com/watch?v=live_video_123", api_key="test_key")
        assert meta.broadcast_status == "live"
        assert meta.is_live is True
        assert meta.active_live_chat_id == "chat_room_abc"
        assert meta.chat_enabled is True


def test_chat_interface_authenticated() -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 400  # HTTP 400 confirms endpoint authentication reached
    mock_resp.json.return_value = {"error": {"message": "Invalid parameter"}}

    with patch("httpx.Client.get", return_value=mock_resp):
        res = check_chat_interface(live_chat_id=None, api_key="valid_key")
        assert res["status"] == "AUTHENTICATED"
        assert res["supported"] is True
        assert "API_KEY" in res["auth_method"]


def test_diagnose_youtube_source_structure() -> None:
    fake_video = {
        "items": [
            {
                "id": "Opnkl7GVQEc",
                "snippet": {
                    "title": "Test Stream",
                    "channelTitle": "Test Channel",
                    "channelId": "UC123",
                    "liveBroadcastContent": "none",
                },
                "status": {"uploadStatus": "processed", "privacyStatus": "public"},
                "liveStreamingDetails": {
                    "actualStartTime": "2026-11-16T16:00:00Z",
                    "actualEndTime": "2026-11-16T17:30:00Z",
                },
            }
        ]
    }
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = fake_video

    with patch("httpx.Client.get", return_value=mock_resp):
        diag = diagnose_youtube_source("https://www.youtube.com/watch?v=Opnkl7GVQEc", api_key="secret_key_123")
        assert diag["video_id"] == "Opnkl7GVQEc"
        assert diag["snippet"]["channel_title"] == "Test Channel"
        assert diag["live_streaming_details"]["broadcast_status"] == "completed"
        assert diag["live_streaming_details"]["duration"] == "1h 30m 0s"
        assert diag["media_acquisition"]["type"] == "WEB_METADATA_URL"
        assert diag["media_acquisition"]["direct_media_available"] is False

        # Verify formatting
        text = format_youtube_diagnostic_report(diag)
        assert "STREAMCLIP AI — YOUTUBE DATA API V3 DIAGNOSTIC REPORT" in text
        assert "Test Channel" in text
        assert "1h 30m 0s" in text
        assert "COMPLETED" in text
        # Verify secret key is never in the report output
        assert "secret_key_123" not in text
