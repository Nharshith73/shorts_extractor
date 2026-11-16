import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from app.detection.config import DetectionConfig
from app.detection.features import FEATURE_NAMES, extract_segment_features
from app.vod.chunked import extract_chunked
from app.vod.timeline import EPOCH
from app.ingestion.stream_reader import probe_duration

pytestmark = [pytest.mark.integration, pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="ffmpeg and ffprobe must be available on PATH")]


def test_chunked_and_unchunked_match_on_same_synthetic_file(tmp_path: Path) -> None:
    source = tmp_path / "partial.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30",
                    "-f", "lavfi", "-i", "sine=frequency=700:sample_rate=48000", "-t", "12.37", "-c:v", "libx264",
                    "-preset", "ultrafast", "-c:a", "aac", str(source)], check=True, capture_output=True)
    duration = probe_duration(source)
    config = DetectionConfig()
    chunked, _ = extract_chunked(source, config, "s", 0, duration, chunk_seconds=1.37)
    original = extract_segment_features(source, segment_id=None, source_id="s", session_id="vod", wall_start_utc=EPOCH,
                                        duration_seconds=duration, config=config)
    assert len(chunked) == len(original)
    for left, right in zip(chunked, original):
        assert left.wall_utc == right.wall_utc and left.bin_seconds == right.bin_seconds
        for name in FEATURE_NAMES:
            assert getattr(left, name) == pytest.approx(getattr(right, name), abs=1e-4, rel=1e-4)


def test_five_minute_vod_run_command_produces_reviewable_set(tmp_path: Path) -> None:
    source = tmp_path / "five-minutes.mp4"
    # Pulsed luminance and loud tone bursts at known times, no drawtext.
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
        "color=c=black:s=320x180:r=30:d=300,drawbox=color=white:t=fill:enable='between(t,80,82)+between(t,200,202)'",
        "-f", "lavfi", "-i", "sine=frequency=700:sample_rate=48000:duration=300",
        "-af", "volume='if(between(t,80,82)+between(t,200,202),4,0.01)':eval=frame",
        "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", str(source)], check=True, capture_output=True)
    config = tmp_path / "config.yaml"
    config.write_text("vod:\n  chunk_seconds: 7.37\n", encoding="utf-8")
    output = tmp_path / "out"
    subprocess.run([sys.executable, "-m", "app.vod", "run", "--video", str(source), "--out", str(output),
                    "--config", str(config), "--start", "00:00:10", "--end", "00:04:50"], check=True, capture_output=True)
    clips = json.loads((output / "clips.json").read_text())
    assert len(clips) == 3
    for clip in clips:
        assert Path(clip["clip_path"]).name.startswith(f'clip_{clip["rank"]:02d}_')
        assert 10 <= clip["start_seconds"] < clip["end_seconds"] <= 290
        assert clip["actual_pre_roll_seconds"] <= 8
        assert probe_duration(Path(clip["clip_path"])) == pytest.approx(clip["end_seconds"] - clip["start_seconds"], abs=0.15)
    assert all(any(abs(clip["peak_seconds"] - expected) <= 3 for clip in clips) for expected in (80, 200))
    page = (output / "index.html").read_text()
    assert page.count("<video") == 3
    for name in ("features.csv", "events.json", "review.csv", "analysis.json", "run.json"):
        assert (output / name).is_file()
