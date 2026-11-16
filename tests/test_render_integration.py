from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from app.clips.render import render_cut
from app.ingestion.stream_reader import probe_duration

pytestmark = [pytest.mark.integration, pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="ffmpeg and ffprobe must be available on PATH")]


@pytest.mark.parametrize("container", ["mp4", "ts"])
def test_input_seek_keeps_non_keyframe_cut_accurate(tmp_path: Path, container: str) -> None:
    source = tmp_path / f"source.{container}"
    clip = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-t", "32",
                    "-c:v", "libx264", "-preset", "ultrafast", "-g", "300", "-sc_threshold", "0", "-c:a", "aac", str(source)],
                   check=True, capture_output=True)
    start, duration = 20.37, 3.1
    render_cut(source, clip, start, duration)
    assert probe_duration(clip) == pytest.approx(duration, abs=0.15)
    def frame(path: Path, offset: float) -> np.ndarray:
        # Output-side reference seek independently verifies the faster input seek.
        result = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ss", str(offset), "-frames:v", "1",
                                 "-vf", "scale=32:18,format=gray", "-f", "rawvideo", "pipe:1"], check=True, capture_output=True)
        return np.frombuffer(result.stdout, dtype=np.uint8).astype(float)
    assert np.mean(abs(frame(source, start) - frame(clip, 0))) < 8
