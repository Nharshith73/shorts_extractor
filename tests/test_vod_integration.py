import json
from pathlib import Path
import shutil

import pytest

from app.vod.pipeline import run
from app.ingestion.stream_reader import probe_duration
from test_detection_integration import make_event_source

pytestmark = [pytest.mark.integration, pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="ffmpeg and ffprobe must be available on PATH")]


def test_vod_detects_injected_events_and_exports_accurate_clips(tmp_path: Path) -> None:
    video = tmp_path / "events.mp4"
    make_event_source(video)
    config = tmp_path / "config.yaml"
    config.write_text("detection:\n  baseline_seconds: 30\n  warmup_seconds: 30\n", encoding="utf-8")
    output = tmp_path / "output"
    result = run(video, output, config)
    timeline = json.loads((output / "events.json").read_text())
    for expected in (80, 140):
        assert any(abs(event["peak_time_seconds"] - expected) <= 3 for event in timeline["events"])
    assert result.clip_count >= 2
    for clip in (output / "clips").glob("*.mp4"):
        metadata = json.loads(clip.with_suffix(".json").read_text())
        assert 30 <= metadata["end_seconds"] - metadata["start_seconds"] <= 60
        assert abs(probe_duration(clip) - (metadata["end_seconds"] - metadata["start_seconds"])) <= 0.15
        assert metadata["components"]
        weights = dict(vocal_spike=2.5, audio_contrast=1.5, audio_spike=0.5,
                       onset_spike=0.5, chat_surge=2.0, visual_activity=0.25)
        assert metadata["score"] == pytest.approx(sum(weights[key]*value for key, value in metadata["ranking_components"].items()))
        assert metadata["reason_codes"]
    assert not (output / "segments").exists()
