from __future__ import annotations

import shutil
import subprocess
from datetime import timedelta
from pathlib import Path

import pytest

from app.detection.config import DetectionConfig
from app.detection.offline import run_offline
from app.detection.runner import DetectionRunner
from app.ingestion.rolling_buffer import RollingBuffer
from app.ingestion.stream_reader import StreamConfig, StreamReader

FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not FFMPEG_AVAILABLE, reason="ffmpeg and ffprobe must be available on PATH"
    ),
]


def make_event_source(path: Path) -> None:
    """Build a three-minute source with quiet/static baseline and two known events."""
    video_inputs = [
        "color=c=black:s=320x180:r=30:d=80",
        "testsrc2=size=320x180:rate=30:d=3",
        "color=c=black:s=320x180:r=30:d=57",
        "testsrc2=size=320x180:rate=30:d=3",
        "color=c=black:s=320x180:r=30:d=37",
    ]
    audio_inputs = [
        "sine=frequency=220:sample_rate=48000:duration=80,volume=0.01",
        "anoisesrc=color=white:sample_rate=48000:duration=3:amplitude=0.9",
        "sine=frequency=220:sample_rate=48000:duration=57,volume=0.01",
        "anoisesrc=color=white:sample_rate=48000:duration=3:amplitude=0.9",
        "sine=frequency=220:sample_rate=48000:duration=37,volume=0.01",
    ]
    command = ["ffmpeg", "-hide_banner", "-y"]
    for specification in [*video_inputs, *audio_inputs]:
        command.extend(["-f", "lavfi", "-i", specification])
    video_concat = "".join(f"[{index}:v]" for index in range(5)) + "concat=n=5:v=1:a=0[v]"
    audio_concat = "".join(f"[{index}:a]" for index in range(5, 10)) + "concat=n=5:v=0:a=1[a]"
    command.extend(
        [
            "-filter_complex",
            f"{video_concat};{audio_concat}",
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-g",
            "30",
            "-keyint_min",
            "30",
            "-sc_threshold",
            "0",
            "-c:a",
            "aac",
            str(path),
        ]
    )
    subprocess.run(command, check=True, capture_output=True, text=True)


def detection_config() -> DetectionConfig:
    return DetectionConfig(
        baseline_seconds=30,
        warmup_seconds=30,
        z_threshold=2.5,
        single_feature_threshold=5,
        min_separation_seconds=20,
    )


def assert_known_events(event_times: list[float]) -> None:
    for expected in (80.0, 140.0):
        assert any(abs(actual - expected) <= 3.0 for actual in event_times)
    false_positives = [
        actual
        for actual in event_times
        if not any(abs(actual - expected) <= 3.0 for expected in (80.0, 140.0))
    ]
    assert len(false_positives) <= 2


def test_synthetic_events_are_detected_offline(tmp_path: Path) -> None:
    source = tmp_path / "events.mp4"
    make_event_source(source)
    _, events, _, _ = run_offline(source, detection_config())
    assert_known_events([event.peak_wall_utc.timestamp() for event in events])


def test_ingested_event_times_map_back_to_segments(tmp_path: Path) -> None:
    source = tmp_path / "events.mp4"
    make_event_source(source)
    segments_dir = tmp_path / "segments"
    reader = StreamReader(
        StreamConfig(
            source=str(source),
            segment_seconds=6,
            retention_seconds=600,
            segments_dir=segments_dir,
            reconnect_max_backoff_seconds=2,
            is_live=False,
        )
    )
    reader.run()
    runner = DetectionRunner(segments_dir / "manifest.sqlite3", detection_config())
    stats = runner.run_once()
    assert stats.segments_analyzed > 0
    events = runner.store.events(reader.source_id)
    assert_known_events(
        [(event.peak_wall_utc - reader.buffer.manifest.all_records(reader.source_id)[0].wall_start_utc).total_seconds() for event in events]
    )
    buffer = RollingBuffer(segments_dir, retention_seconds=600)
    for event in events:
        covering = buffer.get_segments(
            event.start_wall_utc,
            event.end_wall_utc + timedelta(milliseconds=1),
        )
        assert covering
