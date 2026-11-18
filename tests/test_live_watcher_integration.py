"""Integration tests for Twitch Live Watcher, IRC Spike Detector, and Web API."""

from __future__ import annotations

import json
from pathlib import Path
import time
import pytest
from fastapi.testclient import TestClient

from app.detection.chat_spike import ChatSpikeDetector, ChatSpikeEvent
from app.live.twitch_irc import ChatMessage, TwitchIRCClient
from app.live.watcher import LiveWatcherSession
from app.web.server import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_api_watcher_status_idle(client: TestClient) -> None:
    res = client.get("/api/watcher/status")
    assert res.status_code == 200
    data = res.json()
    assert "is_running" in data


def test_api_list_shorts(client: TestClient) -> None:
    res = client.get("/api/shorts")
    assert res.status_code == 200
    data = res.json()
    assert "shorts" in data
    assert isinstance(data["shorts"], list)


def test_live_watcher_session_lifecycle(tmp_path: Path) -> None:
    session = LiveWatcherSession(
        channel="mockchannel",
        output_dir=tmp_path / "completed",
        min_messages=3,
        spike_factor=2.0,
        ai_mode="none",
        layout="blurred_background",
    )

    # Status before starting
    status = session.get_status()
    assert status["is_running"] is False
    assert status["channel"] == "mockchannel"
    assert status["platform"] == "twitch"
    assert status["spikes_detected"] == 0

    # Start session
    session.start()
    assert session.is_running is True

    # Stop session
    session.stop()
    assert session.is_running is False


def test_kick_watcher_session_init(tmp_path: Path) -> None:
    # Test explicit platform="kick"
    session_kick = LiveWatcherSession(
        channel="westcol",
        output_dir=tmp_path / "completed",
        platform="kick",
        min_messages=5,
    )
    assert session_kick.platform == "kick"
    assert session_kick.channel == "westcol"
    assert "kick_live_westcol" in str(session_kick.run_dir)

    # Test auto-detection from Kick URL
    session_url = LiveWatcherSession(
        channel="https://kick.com/xqc",
        output_dir=tmp_path / "completed",
    )
    assert session_url.platform == "kick"
    assert session_url.channel == "xqc"
    assert "kick_live_xqc" in str(session_url.run_dir)


def test_api_watcher_start_and_stop_kick(client: TestClient) -> None:
    # Start Kick watcher via API
    res_start = client.post("/api/watcher/start", json={
        "channel": "mockstreamer",
        "platform": "kick",
        "min_messages": 5,
        "spike_factor": 2.5,
    })
    assert res_start.status_code == 200
    data_start = res_start.json()
    assert data_start["status"] in ("STARTED", "ALREADY_RUNNING")
    assert data_start["platform"] == "kick"

    # Status check
    res_status = client.get("/api/watcher/status")
    assert res_status.status_code == 200
    data_status = res_status.json()
    assert data_status["platform"] == "kick"
    assert data_status["is_running"] is True

    # Stop watcher via API
    res_stop = client.post("/api/watcher/stop")
    assert res_stop.status_code == 200
    assert res_stop.json()["status"] == "STOPPED"

    # Status after stop
    res_status2 = client.get("/api/watcher/status")
    assert res_status2.json()["is_running"] is False


def test_chat_spike_with_simulated_stream(tmp_path: Path) -> None:
    session = LiveWatcherSession(
        channel="testchannel",
        output_dir=tmp_path / "completed",
        min_messages=3,
        spike_factor=2.0,
        ai_mode="none",
    )

    # Simulate baseline messages
    for i in range(10):
        session._spike_detector.add_message(float(i * 10))

    assert len(session._spike_detector.detected_spikes) == 0

    # Trigger surge
    t_surge = 120.0
    for i in range(15):
        session._spike_detector.add_message(t_surge + (i * 0.2))

    # Advance bucket
    spike = session._spike_detector.add_message(t_surge + 15.0)
    assert spike is not None
    assert spike.message_count == 15
    assert len(session._spike_detector.detected_spikes) == 1


def test_api_watcher_test_clip_standalone(client: TestClient) -> None:
    # Trigger standalone test reel cut (5s duration for fast test)
    res = client.post("/api/watcher/test-clip", json={
        "duration": 5.0,
        "layout": "blurred_background",
    })
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "COMPLETED"
    assert data["mode"] == "standalone_test"
    assert "short" in data
    assert Path(data["short"]["path"]).is_file()

    # Verify short appears in /api/shorts list
    res_shorts = client.get("/api/shorts")
    assert res_shorts.status_code == 200
    shorts_list = res_shorts.json()["shorts"]
    assert len(shorts_list) >= 1
    assert any("test_clip" in s["filename"] for s in shorts_list)


def test_api_cookies_status(client: TestClient) -> None:
    res = client.get("/api/cookies-status")
    assert res.status_code == 200
    data = res.json()
    assert "has_cookies_file" in data
    assert "has_node_runtime" in data
    assert data["has_node_runtime"] is True
