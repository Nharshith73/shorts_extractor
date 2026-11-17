import csv
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.clips.render import EncoderSelector
from app.rendering.captions import ass_time, display_word, font_family, generate_ass, phrase_groups
from app.rendering.config import FacecamRegion, RenderConfig, load_render_config
from app.rendering.layouts import LayoutGraph, build_layout
from app.rendering.renderer import MediaInfo, render_original
from app.rendering.transcription import (SpeechSegment, Transcript, WhisperTranscriber, Word,
                                         transcript_key, usable_words)
from app.vod.artifacts import write_json, write_review
from app.vod.render import parse_ranks, render_reviewed, selected_clips


def words() -> list[Word]:
    return [Word(index * 0.4, index * 0.4 + 0.35, text)
            for index, text in enumerate(("look", "at", "this", "great", "escape", "now"))]


def transcript() -> Transcript:
    return Transcript((SpeechSegment(0, 2.5, "look at this great escape now", -0.2, 0.01, tuple(words())),), "en", "fake-test")


def review_fixture(root: Path, count: int = 3) -> Path:
    source = root / "original source.mp4"
    source.write_bytes(b"original")
    rows = []
    for rank in range(1, count + 1):
        preview = root / f"clip_{rank}.mp4"
        # No preview video exists. Rendering must use the original source.
        write_json(preview.with_suffix(".json"), dict(video=str(source), rank=rank, start_seconds=10, end_seconds=13))
        rows.append(dict(clip_path=str(preview), rank=str(rank), approved="y" if rank == 1 else "n",
                         start_seconds="999", end_seconds="1000", score="20", top_features="motion", notes=""))
    write_review(root / "review.csv", rows)
    return source


def test_render_config_defaults_and_dedicated_file(tmp_path: Path) -> None:
    path = tmp_path / "render.yaml"
    path.write_text("render:\n  fonts_dir: custom fonts\n  facecam_region: {x: 10, y: 20, w: 40, h: 60}\n")
    config = load_render_config(path)
    assert (config.whisper_model, config.compute_type, config.device, config.language) == ("small", "int8", "cpu", "auto")
    assert config.fonts_dir == tmp_path / "custom fonts"
    assert config.facecam_region == FacecamRegion(10, 20, 40, 60)
    path.write_text("candidates:\n  encoder: libx264\n")
    assert load_render_config(path).encoder == "libx264"
    with pytest.raises(ValueError, match="safe area"):
        replace(config, margin_bottom=300).validate()
    with pytest.raises(ValueError, match="1..3"):
        replace(config, max_words=4).validate()


def test_ass_timings_phrase_limits_and_current_word_highlight() -> None:
    config = RenderConfig()
    groups = phrase_groups(words(), config)
    assert all(1 <= len(group) <= 3 for group in groups)
    assert len(groups) > 1
    text = generate_ass(words(), config, "Arial Black")
    dialogues = [line for line in text.splitlines() if line.startswith("Dialogue:")]
    assert len(dialogues) == len(words())
    assert ",0:00:00.00,0:00:00.40," in dialogues[0]
    assert all(line.count(r"\1c&H00FFFF&") == 1 for line in dialogues)
    assert "LOOK" in dialogues[0] and "look" not in text
    assert ass_time(59.999) == "0:01:00.00"
    assert ass_time(3661.23) == "1:01:01.23"
    split = phrase_groups([Word(0, 0.5, "hello."), Word(0.6, 1, "there"), Word(2, 3, "later")], config)
    assert len(split) == 3


@pytest.mark.parametrize("style,border,highlight", [
    ("bold_yellow", "1", r"\1c&H00FFFF&"),
    ("clean_white", "1", r"\1c&HFFFFFF&"),
    ("boxed", "3", r"\1c&H00FFFF&"),
])
def test_caption_styles_margins_and_case(style: str, border: str, highlight: str) -> None:
    config = RenderConfig(caption_style=style, uppercase=False, max_words=1)
    text = generate_ass(words(), config, "Poppins")
    line = next(line for line in text.splitlines() if line.startswith("Style:"))
    fields = next(csv.reader([line.removeprefix("Style: ")]))
    assert fields[1] == "Poppins" and fields[15] == border
    assert fields[18] == "2" and int(fields[21]) >= 1920 * 0.18
    assert fields[16:18] == ["4", "2"]
    assert highlight in text and "look" in text
    assert all(len(group) == 1 for group in phrase_groups(words(), config))


def test_caption_text_cannot_inject_ass_and_long_words_fit() -> None:
    assert r"\pos" not in display_word(r"{\pos(1,1)}Hi\Nthere", False)
    long_word = "SUPERCALIFRAGILISTICEXPIALIDOCIOUS"
    text = generate_ass([Word(0, 1, long_word)], RenderConfig(), "Arial Black")
    assert r"{\fs80}" not in text
    assert long_word in text


def test_font_preference_and_system_fallback(tmp_path: Path) -> None:
    assert font_family(tmp_path) == "Arial Black"
    (tmp_path / "Poppins-Bold.ttf").touch()
    assert font_family(tmp_path) == "Poppins"
    (tmp_path / "Anton-Regular.ttf").touch()
    assert font_family(tmp_path) == "Anton"


def test_hallucination_filter_and_word_time_clamping() -> None:
    keep = SpeechSegment(0, 4, "valid", -0.2, 0.01, (Word(-0.1, 0.3, "hello"), Word(0.2, 4, "world")))
    bad_logprob = replace(keep, avg_logprob=-2)
    bad_silence = replace(keep, no_speech_prob=0.9)
    result, rejected = usable_words(Transcript((keep, bad_logprob, bad_silence)), RenderConfig(), 3)
    assert rejected == 2
    assert result == [Word(0, 0.3, "hello"), Word(0.3, 3, "world")]
    empty, _ = usable_words(Transcript((bad_logprob, bad_silence)), RenderConfig(), 3)
    assert empty == []


def test_whisper_requests_word_timing_vad_and_clip_audio(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import faster_whisper
    observed = {}
    class Model:
        def __init__(self, name: str, **kwargs: object) -> None:
            observed.update(model=name, **kwargs)
        def transcribe(self, path: str, **kwargs: object):
            observed.update(audio=path, **kwargs)
            segment = SimpleNamespace(start=0, end=1, text=" hello", avg_logprob=-0.1, no_speech_prob=0.01,
                                      words=[SimpleNamespace(start=0.1, end=0.8, word=" hello", probability=0.9)])
            return iter([segment]), SimpleNamespace(language="en")
    monkeypatch.setattr(faster_whisper, "WhisperModel", Model)
    recognizer = WhisperTranscriber(RenderConfig())
    result = recognizer.transcribe(tmp_path / "clip.wav")
    recognizer.transcribe(tmp_path / "second.wav")
    assert result.segments[0].words[0].text == "hello"
    assert observed["device"] == "cpu" and observed["compute_type"] == "int8"
    assert observed["word_timestamps"] is True and observed["vad_filter"] is True
    assert observed["no_speech_threshold"] is None and observed["log_prob_threshold"] is None
    assert observed["language"] is None
    assert observed["audio"].endswith("second.wav")


def test_layout_graphs_and_missing_facecam_warning(caplog: pytest.LogCaptureFixture) -> None:
    default = build_layout(RenderConfig(), 1920, 1080, True)
    assert "gblur=" in default.graph and "eq=brightness=" in default.graph
    assert "subtitles=filename=captions.ass:fontsdir=fonts" in default.graph
    stacked = build_layout(RenderConfig(layout="stacked", facecam_region=FacecamRegion(20, 30, 320, 240)), 1920, 1080, False)
    assert "crop=320:240:20:30" in stacked.graph and "vstack=inputs=2" in stacked.graph
    assert "subtitles=" not in stacked.graph
    crop = build_layout(RenderConfig(layout="center_crop", crop_center_x=0.75), 1920, 1080, False)
    assert "iw*0.75-540" in crop.graph
    fallback = build_layout(RenderConfig(layout="stacked"), 1920, 1080, False)
    assert fallback.effective_layout == "blurred_background" and "falling back" in caplog.text
    with pytest.raises(ValueError, match="outside"):
        build_layout(RenderConfig(layout="stacked", facecam_region=FacecamRegion(1900, 0, 320, 240)), 1920, 1080, False)


def test_review_approval_and_rank_selection_use_sidecar_times(tmp_path: Path) -> None:
    source = review_fixture(tmp_path)
    approved = selected_clips(tmp_path)
    assert [clip.rank for clip in approved] == [1]
    assert approved[0].source == source and (approved[0].start, approved[0].end) == (10, 13)
    assert [clip.rank for clip in selected_clips(tmp_path, ranks={2, 3})] == [2, 3]
    assert len(selected_clips(tmp_path, all_clips=True)) == 3
    assert parse_ranks("1, 3,5") == {1, 3, 5}
    with pytest.raises(ValueError, match="unknown"):
        selected_clips(tmp_path, ranks={9})
    with pytest.raises(ValueError):
        parse_ranks("1,0")


def test_transcript_cache_identity_excludes_style_and_filter_thresholds(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"media")
    config = RenderConfig()
    first = transcript_key(source, 10, 20, config)
    assert first == transcript_key(source, 10, 20, replace(config, caption_style="boxed", min_avg_logprob=-0.5))
    assert first != transcript_key(source, 10, 21, config)
    assert first != transcript_key(source, 10, 20, replace(config, whisper_model="tiny"))


def test_relative_subtitle_paths_with_spaces_and_medium_encode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import app.rendering.renderer as module
    calls = []
    def run(command: list[str], **kwargs: object):
        calls.append((command, kwargs))
    monkeypatch.setattr(module.subprocess, "run", run)
    work = tmp_path / "path with spaces"
    work.mkdir()
    source = tmp_path / "original source.mp4"
    source.touch()
    render_original(source, work / "result.mp4", 720.125, 30, MediaInfo(1920, 1080, 900, True, 30),
                    words(), RenderConfig(), work, EncoderSelector("libx264"))
    command, options = calls[0]
    graph = command[command.index("-filter_complex") + 1]
    assert str(source) not in graph and str(work) not in graph
    assert ":\\" not in graph and "subtitles=filename=captions.ass:fontsdir=fonts" in graph
    assert options["cwd"] == work and command[command.index("-i") + 1] == str(source)
    assert command.index("-ss") < command.index("-i")
    assert command[command.index("-preset") + 1] == "medium"
    assert command[command.index("-b:a") + 1] == "192k"
    assert "loudnorm=I=-14:TP=-1.5:LRA=11" in command


def test_render_cache_reuse_and_no_usable_speech_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import app.vod.render as module
    source = review_fixture(tmp_path, count=1)
    config_path = tmp_path / "render.yaml"
    config_path.write_text("render:\n  encoder: libx264\n")
    monkeypatch.setattr(module, "probe_media", lambda path: MediaInfo(1920, 1080, 100, True, 30))
    seen = []
    def audio(path: Path, output: Path, start: float, duration: float) -> None:
        seen.append((path, start, duration))
        output.touch()
    monkeypatch.setattr(module, "extract_audio", audio)
    def export(path, output, start, duration, info, words, config, work, selector):
        assert path == source
        output.write_bytes(b"final render")
        return LayoutGraph("", config.layout), "Arial Black", "libx264"
    monkeypatch.setattr(module, "render_original", export)
    monkeypatch.setattr(module, "validate_export", lambda path, duration: MediaInfo(1080, 1920, duration, True, 30))
    class Fake:
        count = 0
        def transcribe(self, audio: Path) -> Transcript:
            self.count += 1
            return transcript()
    recognizer = Fake()
    assert render_reviewed(tmp_path, config_path, transcriber=recognizer)["rendered"] == 1
    config_path.write_text("render:\n  encoder: libx264\n  caption_style: boxed\n")
    render_reviewed(tmp_path, config_path, transcriber=recognizer)
    assert recognizer.count == 1 and seen == [(source, 10, 3)]
    metadata = json.loads((tmp_path / "final" / "clip_1.json").read_text())
    assert metadata["transcription"]["cache_hit"] and metadata["words"][0]["source_start_seconds"] == 10
    config_path.write_text("render:\n  encoder: libx264\n  min_avg_logprob: -0.1\n")
    render_reviewed(tmp_path, config_path, transcriber=recognizer)
    metadata = json.loads((tmp_path / "final" / "clip_1.json").read_text())
    assert metadata["caption_status"] == "no_usable_speech" and metadata["words"] == []
    assert recognizer.count == 1


def test_failed_encode_preserves_previous_export(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import app.vod.render as module
    review_fixture(tmp_path, count=1)
    config = tmp_path / "render.yaml"
    config.write_text("render:\n  captions_enabled: false\n")
    (tmp_path / "final").mkdir()
    final = tmp_path / "final" / "clip_1.mp4"
    final.write_bytes(b"previous valid output")
    monkeypatch.setattr(module, "probe_media", lambda path: MediaInfo(320, 180, 100, True, 30))
    def failed(*args, **kwargs):
        raise ValueError("failed encode")
    monkeypatch.setattr(module, "render_original", failed)
    report = render_reviewed(tmp_path, config)
    assert report["rendered"] == 0 and len(report["failed"]) == 1
    assert final.read_bytes() == b"previous valid output"


def test_no_approved_clips_does_not_transcribe(tmp_path: Path) -> None:
    review_fixture(tmp_path, count=1)
    with (tmp_path / "review.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["approved"] = ""
    write_review(tmp_path / "review.csv", rows)
    config = tmp_path / "render.yaml"
    config.write_text("{}")
    class Forbidden:
        def transcribe(self, audio: Path) -> Transcript:
            raise AssertionError("unapproved clips must not trigger recognition")
    report = render_reviewed(tmp_path, config, transcriber=Forbidden())
    assert report["selected"] == report["rendered"] == 0
    assert not (tmp_path / ".transcripts").exists()
