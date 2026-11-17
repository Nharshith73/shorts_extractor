"""Run the existing VOD commands as one workflow, with optional human review."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Sequence
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORK = ROOT / "data" / "vod-benchmark" / "one-hour-test"
LOGGER = logging.getLogger(__name__)
Step = tuple[str, list[str]]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Detect highlights, extract previews, review, and render captioned Shorts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    result.add_argument("--video", type=Path, default=DEFAULT_WORK / "input" / "stream.mp4",
                        help="Downloaded source recording")
    result.add_argument("--chat", type=Path, help="Optional matching TwitchDownloader chat JSON")
    result.add_argument("--out", type=Path, default=DEFAULT_WORK / "results", help="Results directory")
    result.add_argument("--streamer", help="Source handle for the burned-in credit")
    result.add_argument("--config", type=Path, default=ROOT / "config.yaml", help="Detection/VOD configuration")
    result.add_argument("--render-config", type=Path, default=ROOT / "render.yaml", help="Final render configuration")
    ranking = result.add_mutually_exclusive_group()
    ranking.add_argument("--top-k", type=int, help="Maximum shortlisted clips, 1 to 40")
    ranking.add_argument("--clips-per-hour", type=float, help="Shortlist rate instead of a fixed count")
    result.add_argument("--start", help="Skip the intro: original-file HH:MM:SS")
    result.add_argument("--end", help="Skip the outro: original-file HH:MM:SS")
    result.add_argument("--from-step", choices=("detect", "rescore", "review", "render"), default="detect",
                        help="Resume existing results without repeating earlier stages")
    result.add_argument("--render-all", action="store_true",
                        help="Skip interactive review and render every shortlisted clip")
    result.add_argument("--no-browser", action="store_true", help="Do not open the HTML preview gallery")
    result.add_argument("--dry-run", action="store_true", help="Show commands without processing or opening files")
    return result


def resolve_paths(args: argparse.Namespace) -> None:
    """Resolve caller-relative paths before subprocesses switch to the repository."""
    for name in ("video", "chat", "out", "config", "render_config"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())


def command(*arguments: str) -> list[str]:
    return [sys.executable, "-m", "app.vod", *arguments]


def build_steps(args: argparse.Namespace) -> list[Step]:
    steps: list[Step] = []
    output = str(args.out)
    if args.from_step == "detect":
        arguments = ["run", "--video", str(args.video), "--out", output, "--config", str(args.config)]
        for name in ("chat", "streamer", "top_k", "clips_per_hour", "start", "end"):
            value = getattr(args, name)
            if value is not None:
                arguments.extend(["--" + name.replace("_", "-"), str(value)])
        steps.append(("Detect highlights and extract preview clips", command(*arguments)))
    elif args.from_step == "rescore":
        arguments = ["rescore", "--out", output, "--config", str(args.config), "--extract"]
        for name in ("top_k", "clips_per_hour"):
            value = getattr(args, name)
            if value is not None:
                arguments.extend(["--" + name.replace("_", "-"), str(value)])
        steps.append(("Replan and extract previews from saved features (no detection decode)", command(*arguments)))
    if args.from_step != "render" and not args.render_all:
        steps.append(("Review clips (y approve / n reject / s skip / q finish review)",
                      command("review", "--out", output)))
    steps.append(("Summarize review decisions", command("summary", "--out", output)))
    arguments = ["render", "--out", output, "--config", str(args.render_config)]
    if args.streamer is not None:
        arguments.extend(["--streamer", args.streamer])
    if args.render_all:
        arguments.append("--all")
    steps.append(("Render all shortlisted clips" if args.render_all else "Render approved clips only",
                  command(*arguments)))
    return steps


def validate_inputs(args: argparse.Namespace) -> None:
    required = [args.render_config]
    if args.from_step == "detect":
        required.extend([args.video, args.config])
        if args.chat is not None:
            required.append(args.chat)
    else:
        required.append(args.out / "review.csv")
        if args.from_step == "rescore":
            required.extend([args.config, args.out / "features.csv", args.out / "analysis.json"])
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Required file missing: {path}")
    for executable in ("ffmpeg", "ffprobe"):
        if shutil.which(executable) is None:
            raise FileNotFoundError(f"{executable} is missing from PATH; add your FFmpeg bin directory")


def open_gallery(output: Path) -> None:
    page = output / "index.html"
    if not page.is_file():
        LOGGER.warning("No preview gallery found: %s", page)
        return
    LOGGER.info("Preview gallery: %s", page)
    try:
        if not webbrowser.open(page.as_uri()):
            LOGGER.warning("Open the preview gallery manually in your browser")
    except webbrowser.Error as exc:
        LOGGER.warning("Open the preview gallery manually: %s", exc)


def execute(args: argparse.Namespace) -> int:
    if not args.dry_run:
        validate_inputs(args)
    steps = build_steps(args)
    gallery_opened = False
    for index, (title, arguments) in enumerate(steps, 1):
        LOGGER.info("Step %d/%d: %s", index, len(steps), title)
        LOGGER.info("%s", subprocess.list2cmdline(arguments))
        if args.dry_run:
            continue
        # Open after previews exist and before asking for terminal approvals.
        if arguments[3] not in ("run", "rescore") and not gallery_opened and not args.no_browser:
            open_gallery(args.out)
            gallery_opened = True
        completed = subprocess.run(arguments, cwd=ROOT, check=False)
        if completed.returncode:
            LOGGER.error("Stopped at '%s' (exit %d); later stages were not run", title, completed.returncode)
            return completed.returncode
    if not args.dry_run:
        LOGGER.info("Workflow finished. Final Shorts (if selected): %s", args.out / "final")
        LOGGER.info("Reports: %s and %s", args.out / "run.json", args.out / "render-report.json")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parser().parse_args(argv)
    resolve_paths(args)
    try:
        return execute(args)
    except (OSError, ValueError) as exc:
        LOGGER.error("Pipeline could not run: %s", exc)
        return 2
    except KeyboardInterrupt:
        LOGGER.info("Interrupted. Resume using --from-step review or --from-step render once previews exist.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
