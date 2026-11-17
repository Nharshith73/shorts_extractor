from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
import pytest

from app.clips.render import EncoderSelector, render_cut
from app.ingestion.stream_reader import probe_duration
from app.rendering.config import RenderConfig
from app.rendering.renderer import extract_audio, probe_media, render_original, validate_export
from app.rendering.audio import reaction_frames
from app.rendering.transcription import Transcript
from app.vod.artifacts import write_json, write_review
from app.vod.render import render_reviewed

pytestmark = [pytest.mark.integration, pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="ffmpeg and ffprobe must be available on PATH")]


def build_source(path: Path, seconds: float) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c=black:s=320x180:r=30:d={seconds}",
                    "-f", "lavfi", "-i", f"sine=frequency=700:sample_rate=48000:duration={seconds}",
                    "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", str(path)], check=True, capture_output=True)


def top_frame(path: Path, at: float) -> np.ndarray:
    result = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(at), "-i", str(path), "-frames:v", "1",
                             "-vf", "crop=900:260:30:80", "-pix_fmt", "gray", "-f", "rawvideo", "pipe:1"], check=True, capture_output=True)
    return np.frombuffer(result.stdout, dtype=np.uint8).astype(float)


@pytest.mark.parametrize("duration", [30, 60])
def test_container_export_respects_requested_duration_and_sixty_second_cap(tmp_path: Path, duration: int) -> None:
    source, output = tmp_path / "source.mp4", tmp_path / "clip.mp4"
    build_source(source, duration + 5)
    render_cut(source, output, 2.37, duration)
    assert duration - 0.15 <= probe_duration(output) <= duration
    with pytest.raises(ValueError, match="60"):
        render_cut(source, tmp_path / "overlong.mp4", 0, 61)


def test_normalized_asr_audio_keeps_nonzero_seek_length_and_burst_time(tmp_path: Path) -> None:
    import wave
    source, audio = tmp_path / "source.mp4", tmp_path / "audio.wav"
    expression = "aevalsrc='if(between(t,2,2.5),0.7*sin(2*PI*1100*t),0.005*sin(2*PI*100*t))':s=48000:d=5"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=320x180:r=30:d=5",
                    "-f", "lavfi", "-i", expression, "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", str(source)],
                   check=True, capture_output=True)
    extract_audio(source, audio, 1.37, 2.4)
    with wave.open(str(audio)) as stream:
        assert stream.getnframes()/stream.getframerate() == pytest.approx(2.4, abs=1/16000)
    frames = reaction_frames(audio)
    peak = max(frames, key=lambda frame: frame.rms_db)
    assert 0.6 <= peak.start <= 1.2
    first_loud = next(frame for frame in frames if frame.rms_db >= peak.rms_db-8)
    assert first_loud.start == pytest.approx(2-1.37, abs=0.1)
    assert not (tmp_path / "raw_audio.wav").exists()


def test_credit_burns_in_for_whole_clip_without_speech(tmp_path: Path) -> None:
    source = tmp_path / "source with spaces.mp4"
    build_source(source, 2)
    credited, bare = tmp_path / "credited.mp4", tmp_path / "bare.mp4"
    config = RenderConfig(captions_enabled=False, streamer="TestStreamer", encoder="libx264")
    for output, settings in ((credited, config), (bare, replace(config, credit_enabled=False))):
        with tempfile.TemporaryDirectory(prefix="r ", dir=tmp_path) as work:
            render_original(source, output, 0, 1, probe_media(source), [], settings, Path(work), EncoderSelector("libx264"))
        validate_export(output, 1)
    for at in (0.1, 0.85):
        delta = abs(top_frame(credited, at)-top_frame(bare, at))
        assert delta.mean() > 1 and np.count_nonzero(delta > 50) > 1000


def test_empty_asr_burst_gets_timed_reaction_and_cli_credit(tmp_path: Path) -> None:
    source = tmp_path / "reaction.mp4"
    expression = "aevalsrc='if(between(t,0.6,1.2),0.7*sin(2*PI*1100*t),0.005*sin(2*PI*100*t))':s=48000:d=2"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=320x180:r=30:d=2",
                    "-f", "lavfi", "-i", expression, "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", str(source)],
                   check=True, capture_output=True)
    preview = tmp_path / "clip.mp4"
    write_json(preview.with_suffix(".json"), dict(video=str(source), start_seconds=0, end_seconds=2, streamer="MetadataOwner"))
    write_review(tmp_path / "review.csv", [dict(clip_path=str(preview), rank="1", approved="y", start_seconds="0", end_seconds="2",
                                              score="10", top_features="voice", notes="")])
    config = tmp_path / "render.yaml"
    config.write_text("render:\n  encoder: libx264\n  streamer: ConfigOwner\n")
    class Empty:
        calls = 0
        def transcribe(self, audio: Path) -> Transcript:
            self.calls += 1
            return Transcript((), engine="fake-empty-asr")
    recognizer = Empty()
    report = render_reviewed(tmp_path, config, transcriber=recognizer, streamer="CliOwner")
    assert report["rendered"] == 1, report
    metadata = json.loads((tmp_path / "final" / "clip.json").read_text())
    assert metadata["caption_status"] == "reaction_only" and "[HYPE]" in metadata["caption_text"]
    assert any(0.4 <= tag["start"] <= 0.8 and tag["end"] <= 1.5 for tag in metadata["reaction_tags"])
    assert metadata["credit_text"] == "Credit: @CliOwner" and metadata["credit_source"] == "cli"
    # A style change reuses both recognition and the cached reaction measurements.
    config.write_text("render:\n  encoder: libx264\n  caption_style: boxed\n")
    assert render_reviewed(tmp_path, config, transcriber=recognizer, streamer="CliOwner")["rendered"] == 1
    assert recognizer.calls == 1
