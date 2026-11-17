from dataclasses import replace
from pathlib import Path

import pytest

from app.detection.config import DetectionConfig
from app.detection.models import DetectedEvent, FeatureBin
from app.detection.scoring import detect_events
from app.clips.config import CandidateConfig
from app.vod.chat import CHAT_NAMES, ChatConfig, ChatMessage, aggregate_chat, parse_chat
from app.vod.timeline import EPOCH, media_seconds, media_time
from app.vod.pipeline import plan_windows, write_review
from app.vod.review import review
from app.vod.evaluate import WeakLabel, metrics, random_baseline


def feature(second: float, duration: float = 1) -> FeatureBin:
    return FeatureBin("vod", "vod", media_time(second), duration, -40, -50, 0, 0, 0)


def test_chat_defensive_parsing_and_aggregation() -> None:
    data = {"Comments": [
        {"content_offset_seconds": 0.2, "commenter": {"_id": "a"},
         "message": {"body": "WOW Kappa", "fragments": [{"text": "WOW "}, {"text": "Kappa", "emoticon": {"emoticon_id": "25"}}]}},
        {"contentOffsetSeconds": 0.9, "commenter": {"id": "a"}, "message": {"body": "ok"}},
        {"content_offset_seconds": 1.1, "commenter": None, "message": None},
        {"content_offset_seconds": 2.1, "message": {"fragments": [{"text": "HI"}]}},
        {"content_offset_seconds": "nan"}, {"content_offset_seconds": -1}, {}, None,
    ]}
    parsed = parse_chat(data)
    assert len(parsed) == 4
    bins = [feature(0), feature(1), feature(2, 0.25)]
    signals = aggregate_chat(parsed, bins, ChatConfig())
    first = signals.values[media_time(0)]
    assert first["chat_message_rate"] == 2
    assert first["chat_unique_chatters"] == 1
    assert first["chat_emote_density"] == pytest.approx(1 / 3)
    assert first["chat_caps_ratio"] == pytest.approx(4 / 10)
    assert signals.values[media_time(1)]["chat_caps_ratio"] == 0
    assert signals.values[media_time(2)]["chat_message_rate"] == 4


def test_chat_offsets_and_unknown_missing_authors() -> None:
    parsed = parse_chat({"comments": [{"content_offset_seconds": 600.2, "message": {"body": ""}}]}, -600)
    assert parsed[0].seconds == pytest.approx(0.2)
    signals = aggregate_chat(parsed, [feature(0)], ChatConfig())
    assert signals.values[EPOCH]["chat_unique_chatters"] == 0
    assert signals.values[EPOCH]["chat_emote_density"] == 0
    with pytest.raises(ValueError):
        parse_chat({"unrelated": []})


def test_chat_reads_twitchdownloader_embedded_emote_names() -> None:
    messages = parse_chat({"comments": [{"content_offset_seconds": 0, "commenter": {"_id": None, "display_name": "Alice"},
        "message": {"body": "OMEGALUL hi"}}], "embeddedData": {"thirdParty": [{"name": "OMEGALUL"}]}})
    assert messages[0].chatter == "alice"
    assert messages[0].emotes == 1


def test_time_model_and_bounded_window_planning() -> None:
    assert media_seconds(media_time(942.335)) == pytest.approx(942.335)
    config = CandidateConfig()
    def event(second: float) -> DetectedEvent:
        return DetectedEvent("vod", "vod", media_time(second), media_time(second), media_time(second + 1), 10, {"motion": 10})
    windows = plan_windows([event(80), event(90)], config, 180)
    assert len(windows) == 1
    assert media_seconds(windows[0].start) == 55
    assert media_seconds(windows[0].end) == 111
    at_end = plan_windows([event(179)], config, 180)[0]
    assert media_seconds(at_end.end) == 180
    assert 30 <= (at_end.end - at_end.start).total_seconds() <= 60
    assert not plan_windows([event(5)], config, 10)


def test_no_chat_is_exactly_existing_scoring_and_chat_can_trigger() -> None:
    features = [feature(second) for second in range(100)]
    config = DetectionConfig(warmup_seconds=30)
    assert detect_events(features, config) == detect_events(features, config, None)
    messages = [ChatMessage(80.1, str(index), "WOW Kappa", 1) for index in range(20)]
    auxiliary = aggregate_chat(messages, features, ChatConfig())
    _, events = detect_events(features, config, auxiliary)
    assert any(media_seconds(event.peak_wall_utc) == 80 for event in events)
    assert "chat_message_rate" in events[0].components
    disabled = replace(auxiliary, weights={name: 0 for name in CHAT_NAMES})
    assert detect_events(features, config, disabled) == detect_events(features, config)


def test_weak_metrics_use_one_to_one_interval_matches_and_weights() -> None:
    labels = [WeakLabel(10, 5, 10), WeakLabel(40, 5, 90)]
    result = metrics([11, 13], labels, 0)
    assert result.recall_at_k == 0.5
    assert result.precision_at_k == 0.5
    assert result.weighted_recall == 0.1
    assert result.missed == (1,)
    assert metrics([8, 47], labels, 2).recall_at_k == 1
    assert metrics([], labels, 10).recall_at_k == 0
    assert random_baseline(100, labels, 2, 10) == random_baseline(100, labels, 2, 10)


def test_review_persists_each_response_without_playing(tmp_path: Path) -> None:
    import csv
    rows = [dict(clip_path=f"{index}.mp4", start_seconds="0", end_seconds="20", score="10",
                 top_features="motion", approved="", notes="keep this note") for index in range(3)]
    write_review(tmp_path / "review.csv", rows)
    responses = iter(("y", "s", "q"))
    review(tmp_path, lambda prompt: next(responses))
    with (tmp_path / "review.csv").open(newline="") as stream:
        restored = list(csv.DictReader(stream))
    assert [row["approved"] for row in restored] == ["y", "", ""]
    assert restored[0]["notes"] == "keep this note"
