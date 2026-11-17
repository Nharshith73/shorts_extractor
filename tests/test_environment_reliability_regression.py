"""Regression tests for P1: Operational reliability, no-audio handling, and error diagnostics."""

from datetime import UTC, datetime
from pathlib import Path
import shutil
import subprocess

import pytest

from app.clips.candidate_extractor import CandidateExtractor
from app.clips.config import CandidateConfig
from app.clips.render import render_cut
from app.detection.config import DetectionConfig
from app.detection.features import decode_audio, extract_segment_features


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required for media reliability tests")
def test_media_without_audio_stream_handled_gracefully(tmp_path: Path) -> None:
    """Verify that video files with no audio track decode gracefully without error or crash."""
    silent_video = tmp_path / "silent_video.mp4"
    # Create 3s video with NO audio track
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=3:rate=10:size=160x90",
            "-c:v", "libx264", "-preset", "ultrafast",
            str(silent_video),
        ],
        check=True,
    )

    # 1. decode_audio returns empty array instead of raising CalledProcessError
    samples = decode_audio(silent_video, sample_rate=16000)
    assert samples.size == 0

    # 2. extract_segment_features succeeds and produces 3 feature bins with audio floor
    cfg = DetectionConfig()
    bins = extract_segment_features(
        silent_video,
        segment_id=1,
        source_id="src",
        session_id="sess",
        wall_start_utc=datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC),
        duration_seconds=3.0,
        config=cfg,
    )
    assert len(bins) == 3
    for b in bins:
        # Audio power floor (-160.0 dB)
        assert b.audio_rms_db == pytest.approx(-160.0, abs=1.0)
        assert b.voice_energy_db == pytest.approx(-160.0, abs=1.0)
        assert b.onset_strength == 0.0
        # Video features are still actively measured
        assert b.motion >= 0.0


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_render_cut_cleans_up_output_on_failure_and_preserves_stderr(tmp_path: Path) -> None:
    """If FFmpeg fails during render_cut, any output is cleaned up and stderr message is preserved."""
    nonexistent_source = tmp_path / "nonexistent.mp4"
    out_file = tmp_path / "failed_out.mp4"

    with pytest.raises(RuntimeError) as exc_info:
        render_cut(nonexistent_source, out_file, start_seconds=0.0, duration_seconds=5.0)

    # Output file must not exist
    assert not out_file.exists()
    # Exception must preserve FFmpeg stderr diagnostics
    assert "FFmpeg render_cut failed" in str(exc_info.value)
    assert "No such file" in str(exc_info.value) or "nonexistent" in str(exc_info.value) or "error" in str(exc_info.value).lower()


def test_missing_binary_in_candidate_extractor_identifies_binary(tmp_path: Path) -> None:
    """Missing ffmpeg/ffprobe executable is clearly reported."""
    extractor = CandidateExtractor(
        tmp_path / "segments",
        CandidateConfig(output_dir=tmp_path / "out"),
        ffmpeg_binary="nonexistent_ffmpeg_binary_xyz",
        ffprobe_binary="nonexistent_ffprobe_binary_xyz",
    )
    with pytest.raises(RuntimeError) as exc_info:
        extractor._validate_binaries()

    msg = str(exc_info.value)
    assert "Required executable(s) were not found on PATH" in msg
    assert "nonexistent_ffmpeg_binary_xyz" in msg
    assert "nonexistent_ffprobe_binary_xyz" in msg
