from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
import pytest

from app.clips.render import EncoderSelector
from app.rendering.config import FacecamRegion, RenderConfig
from app.rendering.renderer import probe_media, render_original
from app.rendering.transcription import SpeechSegment, Transcript, Word
from app.vod.artifacts import write_json, write_review
from app.vod.render import render_reviewed

pytestmark = [pytest.mark.integration, pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="ffmpeg and ffprobe must be available on PATH")]


def frame(path: Path, at: float) -> np.ndarray:
    result = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(at), "-i", str(path),
        "-frames:v", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"], check=True, capture_output=True, timeout=30)
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(1920, 1080, 3).astype(float)


@pytest.mark.parametrize("layout", ["blurred_background", "stacked", "center_crop"])
def test_vertical_original_source_render_with_fake_captions_and_spaces(tmp_path: Path, layout: str) -> None:
    root = tmp_path / "VOD output with spaces"
    root.mkdir()
    source = root / "original source.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=700:sample_rate=48000", "-t", "5", "-c:v", "libx264",
        "-preset", "ultrafast", "-g", "60", "-c:a", "aac", str(source)], check=True, capture_output=True, timeout=60)
    preview = root / "clip_01.mp4"
    write_json(preview.with_suffix(".json"), dict(video=str(source), start_seconds=1.37, end_seconds=3.77, rank=1))
    write_review(root / "review.csv", [dict(clip_path=str(preview), rank="1", approved="y", start_seconds="1.37",
        end_seconds="3.77", score="20", top_features="motion", notes="")])
    config_path = root / "render.yaml"
    config_path.write_text(f"render:\n  layout: {layout}\n  facecam_region: {{x: 0, y: 0, w: 120, h: 90}}\n  encoder: libx264\n")
    class Fake:
        def transcribe(self, audio: Path) -> Transcript:
            import wave
            assert audio.name == "audio.wav" and audio.is_file()
            with wave.open(str(audio)) as stream:
                assert stream.getnframes() / stream.getframerate() == pytest.approx(2.4, abs=0.05)
            return Transcript((SpeechSegment(0.1, 2, "watch this", -0.1, 0.01,
                (Word(0.1, 1, "watch"), Word(1, 2, "this"))),), "en", "injected-test-words")
    report = render_reviewed(root, config_path, transcriber=Fake())
    assert report["rendered"] == 1, report
    output = root / "final" / preview.name
    info = probe_media(output)
    assert (info.width, info.height) == (1080, 1920) and info.has_audio
    assert info.duration == pytest.approx(2.4, abs=0.15) and info.fps == pytest.approx(30)
    metadata = json.loads(output.with_suffix(".json").read_text())
    assert metadata["caption_status"] == "captioned" and metadata["caption_text"] == "WATCH THIS"
    assert metadata["effective_layout"] == layout and metadata["video"] == str(source)
    assert metadata["words"][0]["source_start_seconds"] == pytest.approx(1.47)
    if layout == "blurred_background":
        baseline = root / "without captions.mp4"
        config = RenderConfig(captions_enabled=False)
        with tempfile.TemporaryDirectory(dir=root, prefix="baseline with spaces ") as work:
            render_original(source, baseline, 1.37, 2.4, probe_media(source), [], config,
                            Path(work), EncoderSelector("libx264"))
        difference = abs(frame(output, 0.5) - frame(baseline, 0.5))
        assert difference[1300:1560].mean() > 1
        assert np.count_nonzero(difference[1300:1560].max(axis=2) > 50) > 1000


def test_source_without_audio_gets_silent_track_and_no_captions(tmp_path: Path) -> None:
    source = tmp_path / "silent.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=blue:s=320x180:r=30:d=2",
                    "-c:v", "libx264", "-preset", "ultrafast", str(source)], check=True, capture_output=True)
    preview = tmp_path / "clip.mp4"
    write_json(preview.with_suffix(".json"), dict(video=str(source), start_seconds=0.2, end_seconds=1.2))
    write_review(tmp_path / "review.csv", [dict(clip_path=str(preview), approved="y", rank="1", start_seconds="0.2",
        end_seconds="1.2", score="1", top_features="motion", notes="")])
    config = tmp_path / "render.yaml"
    config.write_text("render:\n  encoder: libx264\n")
    class Forbidden:
        def transcribe(self, audio: Path) -> Transcript:
            raise AssertionError("audio-free source must not trigger recognition")
    report = render_reviewed(tmp_path, config, transcriber=Forbidden())
    assert report["rendered"] == 1, report
    final = tmp_path / "final" / "clip.mp4"
    assert probe_media(final).has_audio
    metadata = json.loads(final.with_suffix(".json").read_text())
    assert metadata["caption_status"] == "source_has_no_audio" and metadata["words"] == []
