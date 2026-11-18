"""Unit tests for Twitch anonymous IRC client and rolling median chat spike detector."""

from __future__ import annotations

import time
import pytest

from app.detection.chat_spike import ChatSpikeDetector, detect_spikes_from_chat_log
from app.live.twitch_irc import ChatMessage, PRIVMSG_REGEX, TwitchIRCClient


def test_privmsg_regex_parsing() -> None:
    line = ":xqc!xqc@xqc.tmi.twitch.tv PRIVMSG #streamer :OMG WHAT A PLAY LMAOOOO"
    match = PRIVMSG_REGEX.match(line)
    assert match is not None
    assert match.group(1) == "xqc"
    assert match.group(2) == "OMG WHAT A PLAY LMAOOOO"


def test_chat_spike_detector_triggers_on_surge() -> None:
    detector = ChatSpikeDetector(
        bucket_seconds=10.0,
        history_minutes=3.0,
        spike_factor=2.5,
        min_messages=5,
        cooldown_seconds=30.0,
        lead_in_seconds=20.0,
        lookahead_seconds=10.0,
    )

    t = 0.0
    # Simulate 6 normal buckets with 2 messages each (baseline median = 2.0)
    for _ in range(6):
        detector.add_message(t + 1.0)
        detector.add_message(t + 5.0)
        t += 10.0

    assert len(detector.detected_spikes) == 0
    assert detector.current_median == 2.0

    # Bucket 7: Spike! 20 messages in a single 10s bucket
    for i in range(20):
        ev = detector.add_message(t + (i * 0.4))
        if ev:
            break

    # Advance time past bucket 7
    spike = detector.add_message(t + 11.0)
    assert spike is not None
    assert spike.message_count == 20
    assert spike.baseline_median == 2.0
    assert spike.spike_ratio >= 10.0
    assert spike.context_start == pytest.approx(spike.timestamp - 20.0, abs=1.0)
    assert spike.context_end == pytest.approx(spike.timestamp + 10.0, abs=1.0)


def test_chat_spike_detector_cooldown() -> None:
    detector = ChatSpikeDetector(
        bucket_seconds=10.0,
        history_minutes=3.0,
        spike_factor=2.0,
        min_messages=5,
        cooldown_seconds=30.0,
    )

    # Establish baseline
    t = 0.0
    for _ in range(5):
        for _ in range(2):
            detector.add_message(t + 1.0)
        t += 10.0

    # First spike at t=50.0
    for _ in range(15):
        detector.add_message(t + 2.0)
    t += 10.0
    ev1 = detector.add_message(t)
    assert ev1 is not None

    # Immediate second surge at t=60.0 (within 30s cooldown)
    for _ in range(25):
        detector.add_message(t + 2.0)
    t += 10.0
    ev2 = detector.add_message(t)
    # Debounced by cooldown!
    assert ev2 is None


def test_detect_spikes_from_chat_log_offline() -> None:
    # 3 minutes of quiet chat (1 message every 10s)
    msgs = [{"timestamp": float(i * 10)} for i in range(18)]
    # Big explosion of 30 messages around t=200s
    for i in range(30):
        msgs.append({"timestamp": 200.0 + (i * 0.2)})

    spikes = detect_spikes_from_chat_log(msgs, min_messages=5)
    assert len(spikes) == 1
    assert spikes[0].message_count == 30
    assert 190.0 <= spikes[0].timestamp <= 210.0
