import argparse
from pathlib import Path
import subprocess

import pytest

from scripts import run_pipeline as launcher


def arguments(*values: str) -> argparse.Namespace:
    args = launcher.parser().parse_args(values)
    launcher.resolve_paths(args)
    return args


def test_default_workflow_reviews_before_rendering_only_approved() -> None:
    steps = launcher.build_steps(arguments("--streamer", "ExampleStreamer", "--top-k", "12"))
    commands = [command for _, command in steps]
    assert [command[3] for command in commands] == ["run", "review", "summary", "render"]
    assert str(launcher.DEFAULT_WORK / "input" / "stream.mp4") in commands[0]
    assert commands[0][-4:] == ["--streamer", "ExampleStreamer", "--top-k", "12"]
    assert commands[-1][-2:] == ["--streamer", "ExampleStreamer"]
    assert "--all" not in commands[-1]
    assert str(launcher.ROOT / "render.yaml") in commands[-1]


def test_explicit_render_all_skips_review() -> None:
    commands = [command for _, command in launcher.build_steps(arguments("--render-all"))]
    assert [command[3] for command in commands] == ["run", "summary", "render"]
    assert commands[-1][-1] == "--all"


@pytest.mark.parametrize("stage, expected", [("review", ["review", "summary", "render"]),
                                           ("render", ["summary", "render"])])
def test_resume_never_repeats_detection(stage: str, expected: list[str]) -> None:
    commands = [command for _, command in launcher.build_steps(arguments("--from-step", stage))]
    assert [command[3] for command in commands] == expected


def test_rescore_rebuilds_previews_without_detection_then_reviews_and_renders() -> None:
    commands = [command for _, command in launcher.build_steps(arguments("--from-step", "rescore", "--top-k", "12"))]
    assert [command[3] for command in commands] == ["rescore", "review", "summary", "render"]
    assert "--extract" in commands[0] and commands[0][-2:] == ["--top-k", "12"]
    assert "--video" not in commands[0]


def test_paths_with_spaces_and_optional_arguments_remain_separate(tmp_path: Path) -> None:
    video = tmp_path / "source with spaces.mp4"
    chat = tmp_path / "matching chat.json"
    output = tmp_path / "my results"
    args = arguments("--video", str(video), "--chat", str(chat), "--out", str(output),
                     "--start", "00:01:00", "--end", "00:59:00", "--clips-per-hour", "8")
    first = launcher.build_steps(args)[0][1]
    assert first[first.index("--video") + 1] == str(video)
    assert first[first.index("--chat") + 1] == str(chat)
    assert first[first.index("--out") + 1] == str(output)
    assert first[first.index("--clips-per-hour") + 1] == "8.0"
    assert first[-4:] == ["--start", "00:01:00", "--end", "00:59:00"]


def test_child_failure_stops_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(launcher, "validate_inputs", lambda args: None)
    def fail(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        calls.append(command)
        return subprocess.CompletedProcess(command, 2)
    monkeypatch.setattr(launcher.subprocess, "run", fail)
    assert launcher.execute(arguments("--no-browser")) == 2
    assert len(calls) == 1


def test_execution_opens_gallery_after_detection_and_preserves_interactive_input(monkeypatch: pytest.MonkeyPatch) -> None:
    actions: list[str] = []
    monkeypatch.setattr(launcher, "validate_inputs", lambda args: None)
    monkeypatch.setattr(launcher, "open_gallery", lambda output: actions.append("gallery"))
    def succeed(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        assert kwargs == {"cwd": launcher.ROOT, "check": False}
        # stdin/stdout remain inherited so terminal review stays interactive.
        actions.append(command[3])
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(launcher.subprocess, "run", succeed)
    assert launcher.execute(arguments()) == 0
    assert actions == ["run", "gallery", "review", "summary", "render"]


def test_dry_run_never_checks_media_or_launches_anything(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected(*args: object, **kwargs: object) -> None:
        pytest.fail("dry-run must not validate inputs, launch processing, or open a browser")
    monkeypatch.setattr(launcher, "validate_inputs", unexpected)
    monkeypatch.setattr(launcher.subprocess, "run", unexpected)
    monkeypatch.setattr(launcher.webbrowser, "open", unexpected)
    assert launcher.main(["--dry-run"]) == 0


def test_missing_video_fails_before_work_begins(tmp_path: Path) -> None:
    assert launcher.main(["--video", str(tmp_path / "missing.mp4"), "--no-browser"]) == 2
