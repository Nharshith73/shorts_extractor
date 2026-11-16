import logging
from pathlib import Path
import subprocess

import pytest

from app.clips.config import CandidateConfig, load_config
from app.clips.render import EncoderSelector, render_cut


def test_auto_probe_falls_back_with_nvenc_driver_cause_and_caches(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    calls = []
    cause = b"Driver does not support the required nvenc API version. Required: 13.1 Found: 13.0\nThe minimum required Nvidia driver for nvenc is 610.00 or newer"
    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        calls.append(command)
        raise subprocess.CalledProcessError(1, command, stderr=cause)
    monkeypatch.setattr(subprocess, "run", run)
    selector = EncoderSelector("auto")
    with caplog.at_level(logging.WARNING):
        assert selector.select(Path("source.mp4"), 720) == "libx264"
        assert selector.select(Path("source.mp4"), 800) == "libx264"
    assert len(calls) == 1
    assert calls[0][calls[0].index("-t") + 1] == "1"
    assert "h264_nvenc" in calls[0] and "-encoders" not in calls[0]
    assert "Required: 13.1 Found: 13.0" in caplog.text
    assert "610.00" in caplog.text and "falling back to libx264" in caplog.text


def test_auto_requires_actual_encoded_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: subprocess.CompletedProcess(command, 0, b"frame=0\nout_time_us=0\n", b""))
    assert EncoderSelector("auto").select(Path("source.mp4"), 0) == "libx264"


def test_successful_probe_selects_nvenc(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: subprocess.CompletedProcess(command, 0, b"frame=30\nout_time_us=1000000\n", b""))
    assert EncoderSelector("auto").select(Path("source.mp4"), 0) == "h264_nvenc"


def test_input_seek_and_auto_fallback_use_software_encode(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        calls.append(command)
        if "-progress" in command:
            raise subprocess.CalledProcessError(1, command, stderr=b"NVENC unavailable")
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(subprocess, "run", run)
    render_cut(Path("source.mp4"), Path("clip.mp4"), 720.125, 30, video_encoder="auto")
    export = calls[-1]
    assert export.index("-ss") < export.index("-i")
    assert export[export.index("-ss") + 1] == "720.125000"
    assert export[export.index("-c:v") + 1] == "libx264"
    assert "-accurate_seek" in export
    assert export[export.index("-preset") + 1] == "veryfast"
    assert export[export.index("-crf") + 1] == "18"


def test_encoder_config_supports_auto_and_legacy_key(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("candidates:\n  encoder: auto\n", encoding="utf-8")
    assert load_config(path)[0].requested_encoder == "auto"
    path.write_text("candidates:\n  video_encoder: h264_nvenc\n", encoding="utf-8")
    assert load_config(path)[0].requested_encoder == "h264_nvenc"
    path.write_text("candidates:\n  encoder: auto\n  video_encoder: libx264\n", encoding="utf-8")
    with pytest.raises(ValueError, match="conflicting"):
        load_config(path)
    assert CandidateConfig(video_encoder="libx264").requested_encoder == "libx264"
