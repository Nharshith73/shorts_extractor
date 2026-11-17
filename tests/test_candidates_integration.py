from dataclasses import replace
from datetime import timedelta
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess

import numpy as np
import pytest

from app.clips.candidate_extractor import CandidateExtractor
from app.clips.config import CandidateConfig
from app.detection.config import DetectionConfig
from app.detection.runner import DetectionRunner
from app.ingestion.stream_reader import StreamConfig, StreamReader, probe_duration
from app.ingestion.rolling_buffer import RollingBuffer, to_utc_text

pytestmark = [pytest.mark.integration, pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="ffmpeg and ffprobe must be available on PATH")]


def build_source(path: Path) -> None:
    # Steady black/220Hz baseline, three-second visual/noise bursts at 80/140s.
    durations = (80, 3, 57, 3, 37)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for index, duration in enumerate(durations):
        specification = (f"testsrc2=size=320x180:rate=30:duration={duration}" if index in (1, 3)
                         else f"color=c=black:s=320x180:r=30:d={duration}")
        command += ["-f", "lavfi", "-i", specification]
    for index, duration in enumerate(durations):
        specification = (f"sine=frequency=1000:sample_rate=48000:duration={duration},volume=6"
                         if index in (1, 3) else f"sine=frequency=220:sample_rate=48000:duration={duration},volume=0.01")
        command += ["-f", "lavfi", "-i", specification]
    video = "".join(f"[{index}:v]" for index in range(5)) + "concat=n=5:v=1:a=0[v]"
    audio = "".join(f"[{index}:a]" for index in range(5, 10)) + "concat=n=5:v=0:a=1[a]"
    command += ["-filter_complex", f"{video};{audio}", "-map", "[v]", "-map", "[a]",
                "-c:v", "libx264", "-preset", "ultrafast", "-g", "30", "-sc_threshold", "0", "-c:a", "aac", str(path)]
    subprocess.run(command, check=True, capture_output=True, timeout=120)


@pytest.fixture(scope="module")
def captured(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, str]:
    root = tmp_path_factory.mktemp("candidates-media")
    source = root / "source.mp4"
    build_source(source)
    reader = StreamReader(StreamConfig(source=str(source), segments_dir=root / "segments", retention_seconds=600))
    reader.run()
    return source, reader.buffer.segments_dir, reader.source_id


def clone_segments(original: Path, destination: Path) -> None:
    shutil.copytree(original, destination)
    with sqlite3.connect(destination / "manifest.sqlite3") as connection:
        for segment_id, path in connection.execute("SELECT id,path FROM segments WHERE path IS NOT NULL").fetchall():
            connection.execute("UPDATE segments SET path=? WHERE id=?", (str(destination / Path(path).name), segment_id))


def frame(path: Path, at: float) -> np.ndarray:
    result = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(at), "-i", str(path), "-frames:v", "1",
                             "-vf", "scale=32:18,format=gray", "-f", "rawvideo", "pipe:1"], capture_output=True, check=True)
    return np.frombuffer(result.stdout, dtype=np.uint8).astype(float)


def frequency(path: Path, at: float) -> float:
    result = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(at), "-i", str(path), "-t", "0.5",
                             "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1"], capture_output=True, check=True)
    samples = np.frombuffer(result.stdout, dtype="<f4")
    return float(np.fft.rfftfreq(len(samples), 1 / 16000)[np.argmax(abs(np.fft.rfft(samples)))])


def test_known_event_exports_have_accurate_duration_and_content(captured: tuple[Path, Path, str], tmp_path: Path) -> None:
    source, original, source_id = captured
    segments = tmp_path / "segments"
    clone_segments(original, segments)
    buffer = RollingBuffer(segments, 600)
    records = buffer.manifest.all_records(source_id)
    origin = records[0].wall_start_utc
    worker = CandidateExtractor(segments, CandidateConfig(output_dir=tmp_path / "out"))
    with worker.store.connect() as connection:
        connection.execute("""CREATE TABLE events (id INTEGER PRIMARY KEY,source_id TEXT,session_id TEXT,
            peak_wall_utc TEXT,start_wall_utc TEXT,end_wall_utc TEXT,score REAL)""")
        for index, second in enumerate((80.2, 140.2)):
            peak = origin + timedelta(seconds=second)
            connection.execute("INSERT INTO events VALUES (?,?,?,?,?,?,?)", (index + 1, source_id,
                records[0].session_id, to_utc_text(peak), to_utc_text(peak), to_utc_text(peak + timedelta(seconds=1)), 20))
    candidates = worker.run_once()
    assert len(candidates) == 2
    for candidate in candidates:
        assert candidate.status == "extracted", candidate.error
        output = Path(candidate.clip_path)
        expected_duration = (candidate.window.end - candidate.window.start).total_seconds()
        assert abs(probe_duration(output) - expected_duration) < 0.15
        start_offset = (candidate.window.start - origin).total_seconds()
        assert 30 <= expected_duration <= 60
        for offset in (2.0, 25.5, 32.0):
            assert np.mean(abs(frame(source, start_offset + offset) - frame(output, offset))) < 8
            assert abs(frequency(source, start_offset + offset) - frequency(output, offset)) < 5
        assert json.loads(output.with_suffix(".json").read_text())["events"][0]["score"] == 20
    assert not any(record.pinned for record in buffer.manifest.all_records())
    assert len(worker.run_once()) == 2


def test_ingestion_detection_extraction_contains_event_at_pre_roll(captured: tuple[Path, Path, str], tmp_path: Path) -> None:
    _, original, source_id = captured
    segments = tmp_path / "segments"
    clone_segments(original, segments)
    runner = DetectionRunner(segments / "manifest.sqlite3", DetectionConfig(baseline_seconds=30, warmup_seconds=30))
    runner.run_once()
    worker = CandidateExtractor(segments, CandidateConfig(output_dir=tmp_path / "out", pre_roll_seconds=15,
                                                        post_roll_seconds=10, min_seconds=20, max_seconds=30))
    candidates = worker.run_once()
    for expected in (80, 140):
        origin = worker.buffer.manifest.all_records(source_id)[0].wall_start_utc
        candidate = next(item for item in candidates if abs((item.window.primary.peak - origin).total_seconds() - expected) < 3)
        assert candidate.status == "extracted", candidate.error
        output = Path(candidate.clip_path)
        # The first nonblack frame is the injected visual event.
        changes = [offset / 10 for offset in range(130, 191) if frame(output, offset / 10).mean() > 5]
        region_offset = (min(candidate.window.primary.start, candidate.window.primary.peak) - candidate.window.start).total_seconds()
        assert abs(region_offset - 15) <= 0.5
        # Detection can start a region before the known visual onset. Preserve
        # that context while checking media placement against the source clock.
        expected_offset = (origin + timedelta(seconds=expected) - candidate.window.start).total_seconds()
        assert changes and abs(changes[0] - expected_offset) <= 0.5


def test_ffprobe_recovers_real_orphan(captured: tuple[Path, Path, str], tmp_path: Path) -> None:
    _, original, _ = captured
    path = tmp_path / "segment_original_recovered.ts"
    shutil.copyfile(next(original.glob("*.ts")), path)
    reader = StreamReader(StreamConfig(source="recovery-test", segments_dir=tmp_path))
    reader.recover_unfinished()
    record = reader.buffer.manifest.get_by_path(path)
    assert record.status == "complete"
    assert record.duration == pytest.approx(probe_duration(path))
