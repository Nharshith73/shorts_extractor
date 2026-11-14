from __future__ import annotations

import shutil
import subprocess
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Callable

import pytest

from app.ingestion.stream_reader import StreamConfig, StreamReader, probe_duration

FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not FFMPEG_AVAILABLE, reason="ffmpeg and ffprobe must be available on PATH"
    ),
]


def make_synthetic_source(path: Path, seconds: int) -> None:
    """Create keyframe-friendly gameplay-like test video with burned-in PTS and tone."""
    video_filter = (
        "testsrc2=size=320x180:rate=30,"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-f",
            "lavfi",
            "-i",
            video_filter,
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=1000:sample_rate=48000",
            "-t",
            str(seconds),
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
            "-shortest",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def wait_for(predicate: Callable[[], bool], timeout_seconds: float = 12.0) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.1)
    return False


def test_two_minute_ingestion_and_retention(tmp_path: Path) -> None:
    source = tmp_path / "synthetic-120s.mp4"
    make_synthetic_source(source, 120)
    config = StreamConfig(
        source=str(source),
        segment_seconds=6,
        retention_seconds=600,
        segments_dir=tmp_path / "segments",
        reconnect_max_backoff_seconds=2,
        is_live=False,
    )
    reader = StreamReader(config)
    reader.run()

    records = reader.buffer.manifest.all_records(reader.source_id)
    completed = [record for record in records if record.status == "complete"]
    assert len(completed) >= 18
    assert sum(record.duration for record in completed) == pytest.approx(120, abs=10)
    for record in completed:
        assert record.path is not None and record.path.exists()
        assert probe_duration(record.path) > 0

    for previous, current in zip(completed, completed[1:]):
        assert current.wall_start_utc >= previous.wall_start_utc
        unexplained_gap = (current.wall_start_utc - previous.wall_end_utc).total_seconds()
        assert unexplained_gap < 2.0
        if (
            current.session_id == previous.session_id
            and not previous.timing_reanchored
            and not current.timing_reanchored
        ):
            contiguous_delta = (
                current.wall_start_utc - previous.wall_end_utc
            ).total_seconds()
            assert abs(contiguous_delta) < 0.05
    assert not [record for record in records if record.status == "discontinuity"]

    latest_end = max(record.wall_end_utc for record in completed)
    removed = reader.buffer.cleanup_once(latest_end + timedelta(seconds=601))
    assert removed == len(completed)
    assert not reader.buffer.manifest.all_records(reader.source_id)


def test_reader_restarts_after_process_is_killed(tmp_path: Path) -> None:
    source = tmp_path / "synthetic-30s.mp4"
    make_synthetic_source(source, 30)
    reader = StreamReader(
        StreamConfig(
            source=str(source),
            segment_seconds=3,
            retention_seconds=120,
            segments_dir=tmp_path / "segments",
            reconnect_max_backoff_seconds=1,
            is_live=False,
        )
    )
    reader.start()
    try:
        assert wait_for(lambda: reader.current_process is not None)
        assert wait_for(
            lambda: any(
                record.status == "complete"
                for record in reader.buffer.manifest.all_records(reader.source_id)
            )
        )
        process = reader.current_process
        assert process is not None
        process.kill()
        assert wait_for(lambda: reader.restart_count >= 1)
        assert wait_for(lambda: reader.current_process is not None and reader.current_process is not process)
        assert wait_for(
            lambda: len(
                {
                    record.session_id
                    for record in reader.buffer.manifest.all_records(reader.source_id)
                    if record.status == "complete"
                }
            ) >= 2
        )
        completed = [
            record
            for record in reader.buffer.manifest.all_records(reader.source_id)
            if record.status == "complete"
        ]
        assert len({record.session_id for record in completed}) >= 2
        boundaries = reader.buffer.get_discontinuities(
            datetime(1970, 1, 1, tzinfo=UTC), datetime(2100, 1, 1, tzinfo=UTC)
        )
        assert any(
            boundary.previous_session_id != boundary.next_session_id
            and boundary.previous_session_id is not None
            for boundary in boundaries
        )
    finally:
        reader.stop()
