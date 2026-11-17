from dataclasses import replace
from datetime import UTC, datetime, timedelta
import csv
import json
from pathlib import Path
import sqlite3
import wave

import numpy as np
import pytest

from app.clips.config import CandidateConfig
from app.clips.duration import encode_duration, guard_duration, peak_bounds
from app.clips.selection import ranked_live_events
from app.clips.store import Candidate, CandidateStore
from app.clips.windows import Event, Window, event_window, merge_windows
from app.detection.config import DetectionConfig
from app.detection.models import DetectedEvent, FeatureBin
from app.detection.scoring import AuxiliaryFeatures, score_feature_bins
from app.detection.store import DetectionStore
from app.media_identity import metadata_streamer, source_streamer
from app.rendering.audio import reaction_frames
from app.rendering.captions import generate_ass
from app.rendering.config import RenderConfig
from app.rendering.renderer import MediaInfo
from app.rendering.transcription import Transcript, Word, add_reaction_tags
from app.vod.render import RenderClip, bounded_render_clip, context_streamer
from app.vod.selection import peak_window, rank_peaks
from app.vod.settings import SelectionConfig
from app.vod.timeline import media_seconds, media_time


def test_peak_centred_cap_and_video_edge_shifts() -> None:
    assert peak_bounds(0, 100, 70) == (30, 90)
    assert peak_bounds(0, 100, 5, lower=0, upper=100) == (0, 60)
    assert peak_bounds(0, 100, 98, lower=0, upper=100) == (40, 100)
    assert peak_bounds(0, 100, 10, lower=0, upper=20) == (0, 20)
    assert peak_bounds(0, 100, 70, maximum=30) == (50, 80)
    assert encode_duration(30) == 30
    assert encode_duration(45) == 45
    assert encode_duration(60) < 60
    with pytest.raises(ValueError, match="60"):
        encode_duration(60.1)
    with pytest.raises(ValueError, match="60"):
        guard_duration(60.001, 60)
    guard_duration(29.97, 30)
    guard_duration(45.03, 45)
    guard_duration(59.97, 60)


def test_live_and_vod_windows_and_legacy_render_share_cap() -> None:
    event = Event(1, "s", "session", media_time(70), media_time(0), media_time(100), 20)
    config = CandidateConfig(max_seconds=60)
    window = event_window(event, config)
    assert (media_seconds(window.start), media_seconds(window.end)) == (30, 90)
    secondary = replace(event, id=2, peak=media_time(75), score=30)
    merged = merge_windows([window, event_window(secondary, config)], config)
    assert len(merged) == 1 and merged[0].primary.id == 2
    assert (merged[0].end-merged[0].start).total_seconds() == 60
    detected = DetectedEvent("s", "session", media_time(70), media_time(0), media_time(100), 20, {})
    assert peak_window(detected, config, 0, 100) == (30, 90)
    clip = RenderClip(1, "clip.mp4", Path("original.mp4"), 0, 100, {"peak_seconds": 70})
    final = bounded_render_clip(clip, MediaInfo(1920, 1080, 100, True, 30))
    assert (final.start, final.end) == (30, 90)


def test_ranked_top_k_rejects_scene_cuts_and_prefers_audio_chat() -> None:
    base = FeatureBin("s", "vod", media_time(0), 1, -50, -55, 0.001, 1, 0.01)
    features = [replace(base, wall_utc=media_time(second)) for second in range(200)]
    features[40] = replace(features[40], motion=100, cut_change=1)
    for second in (80, 81):
        features[second] = replace(features[second], audio_rms_db=-5, voice_energy_db=-7, onset_strength=1, motion=30)
    chat = AuxiliaryFeatures({row.wall_utc: {"chat_message_rate": 40 if index == 140 else 0}
                              for index, row in enumerate(features)}, {"chat_message_rate": 1}, {"chat_message_rate": 1})
    config = DetectionConfig(warmup_seconds=20)
    scored = score_feature_bins(features, config, chat, include_unclipped=True)
    assert scored[40].eligible is False and "visual_only_penalized" in scored[40].reason_codes
    assert "vocal_spike" in scored[80].reason_codes and "audio_contrast" in scored[80].reason_codes
    assert "chat_surge" in scored[140].reason_codes
    selected = rank_peaks(scored, SelectionConfig(top_k=2, min_separation_seconds=20), config,
                          CandidateConfig(min_seconds=15, pre_roll_seconds=8, post_roll_seconds=6), 0, 200)
    assert sorted(media_seconds(event.peak_wall_utc) for event, _ in selected) == [80, 140]
    assert all(event.reason_codes for event, _ in selected)


def test_sustained_loud_gameplay_loses_audio_contrast() -> None:
    base = FeatureBin("s", "vod", media_time(0), 1, -50, -60, 0, 1, 0.01)
    rows = [replace(base, wall_utc=media_time(second), audio_rms_db=-10 if second >= 40 else -50,
                    voice_energy_db=-14 if second >= 40 else -60, motion=30 if second >= 40 else 1)
            for second in range(120)]
    scored = score_feature_bins(rows, DetectionConfig(warmup_seconds=20))
    assert scored[40].eligible is True
    assert scored[100].eligible is False
    assert scored[100].ranking_components["audio_contrast"] == 0


def test_one_caps_message_does_not_become_a_chat_surge() -> None:
    base = FeatureBin("s", "vod", media_time(0), 1, -50, -55, 0, 1, 0)
    rows = [replace(base, wall_utc=media_time(second)) for second in range(100)]
    values = {row.wall_utc: dict(chat_message_rate=1 if index == 80 else 0,
                               chat_caps_ratio=1 if index == 80 else 0) for index, row in enumerate(rows)}
    chat = AuxiliaryFeatures(values, dict(chat_message_rate=1, chat_caps_ratio=1), dict(chat_message_rate=1, chat_caps_ratio=0.05))
    scored = score_feature_bins(rows, DetectionConfig(), chat)
    assert scored[80].z_scores["chat_caps_ratio"] == 10
    assert scored[80].eligible is False


def test_local_vocal_contrast_survives_wide_global_audio_baseline() -> None:
    rows = [FeatureBin("s", "vod", media_time(second), 1, -60+second%60, -62+second%60, 0.01, 1, 0)
            for second in range(201)]
    for second in range(195, 200):
        rows[second] = replace(rows[second], audio_rms_db=-50, voice_energy_db=-52)
    rows[200] = replace(rows[200], audio_rms_db=-8, voice_energy_db=-9)
    scored = score_feature_bins(rows, DetectionConfig())
    assert scored[-1].z_scores["voice_energy_db"] < 2
    assert scored[-1].eligible is True and "vocal_spike" in scored[-1].reason_codes
    assert scored[-1].qualifies is True  # Ranked evidence is not gated by legacy thresholds.


def test_live_ranking_enforces_quota_and_nms() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    def event(seconds: int, score: float) -> Event:
        at = now+timedelta(seconds=seconds)
        return Event(seconds, "s", "one", at, at, at+timedelta(seconds=1), score)
    events = [event(10, 10), event(11, 50), event(80, 30), replace(event(150, 1000), reason_codes=("visual_only_penalized",))]
    result = ranked_live_events(events, [], CandidateConfig(top_k=1), now)
    assert [row.id for row in result] == [11]
    existing = Candidate("existing", event_window(event(11, 50), CandidateConfig()), "extracted", (), now, None, None)
    assert ranked_live_events(events, [existing], CandidateConfig(top_k=1), now) == []
    waiting = replace(existing, status="waiting")
    result = ranked_live_events([event(12, 100), event(80, 30)], [waiting], CandidateConfig(top_k=2), now)
    assert [row.id for row in result] == [12, 80]  # Merging into a waiting clip uses no new quota slot.


def test_credit_ass_is_safe_persistent_and_separate_from_captions() -> None:
    config = RenderConfig(streamer="@ExampleStreamer", captions_enabled=False)
    text = generate_ass([], config, "Arial Black", 30)
    assert "Credit: @ExampleStreamer" in text
    assert "Dialogue: 10,0:00:00.00,0:00:30.00,Credit" in text
    style = next(line.removeprefix("Style: ") for line in text.splitlines() if line.startswith("Style: Credit,"))
    fields = next(csv.reader([style]))
    assert fields[18] == "7" and fields[21] == "160"
    assert fields[3].startswith("&H30")
    assert r"\pos" not in generate_ass([], replace(config, streamer=r"bad{\pos(1,1)}"), "Arial Black", 1)


def test_streamer_metadata_uses_owner_not_chat_commenters(tmp_path: Path) -> None:
    assert metadata_streamer({"streamer": {"name": "Owner"}, "comments": [{"commenter": {"name": "Viewer"}}]}) == "Owner"
    assert metadata_streamer({"comments": [{"commenter": {"name": "Viewer"}}]}) is None
    video = tmp_path / "vod.mp4"
    video.with_suffix(".info.json").write_text(json.dumps({"uploader": "KnownOwner"}))
    assert source_streamer(video) == "KnownOwner"
    (tmp_path / "analysis.json").write_text(json.dumps({"streamer": "SavedOwner"}))
    assert context_streamer(tmp_path) == "SavedOwner"


def test_reaction_fallback_tags_untranscribed_burst_but_not_engine_or_spoken_word(tmp_path: Path) -> None:
    rate = 16000
    times = np.arange(rate*4)/rate
    quiet = 0.005*np.sin(2*np.pi*100*times)
    samples = quiet.copy()
    mask = (times >= 1) & (times < 1.6)
    samples[mask] = 0.7*np.sin(2*np.pi*1100*times[mask])
    path = tmp_path / "shout.wav"
    def write(audio: np.ndarray) -> None:
        with wave.open(str(path), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(rate)
            stream.writeframes((audio*32767).astype("<i2").tobytes())
    write(samples)
    transcript = Transcript((), audio_frames=reaction_frames(path))
    words, details = add_reaction_tags(transcript, [], RenderConfig())
    assert details and words[0].text == "[HYPE]" and words[0].kind == "reaction"
    assert abs(words[0].start-1) <= 0.1 and words[-1].end <= 1.8
    spoken = [Word(0.9, 1.8, "wow")]
    assert add_reaction_tags(transcript, spoken, RenderConfig())[0] == spoken
    write(np.sin(2*np.pi*1100*times)*0.5)
    assert not add_reaction_tags(Transcript((), audio_frames=reaction_frames(path)), [], RenderConfig())[1]


def test_event_migration_and_candidate_snapshot_preserve_ranking_reasons(tmp_path: Path) -> None:
    database = tmp_path / "manifest.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("""CREATE TABLE events(id INTEGER PRIMARY KEY, source_id TEXT, session_id TEXT,
            peak_wall_utc TEXT,start_wall_utc TEXT,end_wall_utc TEXT,score REAL,components_json TEXT,created_utc TEXT)""")
        connection.execute("INSERT INTO events VALUES(1,'s','one','2026-01-01T00:00:20Z','2026-01-01T00:00:20Z','2026-01-01T00:00:21Z',10,'{}','2026-01-01T00:00:21Z')")
    store = DetectionStore(database)
    assert len(store.events()) == 1 and store.events()[0].reason_codes == ()
    event = replace(store.events()[0], reason_codes=("vocal_spike", "audio_contrast"), ranking_components={"vocal_spike": 8})
    store.replace_session_events("s", "one", [event])
    candidates = CandidateStore(database)
    pending = candidates.new_events()[0]
    assert pending.reason_codes == event.reason_codes and pending.ranking_components == event.ranking_components
    candidates.create(event_window(pending, CandidateConfig()), [], event.peak_wall_utc)
    assert candidates.all()[0].window.primary.reason_codes == event.reason_codes
    later = replace(event, peak_wall_utc=event.peak_wall_utc+timedelta(seconds=1))
    store.replace_session_events("s", "one", [later])
    candidates.record_selection_decision(candidates.new_events()[0], "suppressed_nms", later.peak_wall_utc)
    assert CandidateStore(database).new_events() == []
