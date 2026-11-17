"""Render reviewed VOD candidates from original media, never preview pixels."""

import csv
from dataclasses import asdict, dataclass, replace
import json
import logging
import math
from pathlib import Path
import subprocess
import tempfile
import time

from app.clips.candidate_extractor import worker_lock
from app.clips.render import EncoderSelector
from app.rendering.captions import credit_text, display_word
from app.rendering.audio import reaction_frames
from app.rendering.config import RenderConfig, load_render_config
from app.rendering.renderer import MediaInfo, extract_audio, probe_media, render_original, validate_export
from app.rendering.transcription import (Transcript, Transcriber, WhisperTranscriber, read_transcript,
    save_transcript, transcript_key, usable_words, add_reaction_tags)
from app.clips.duration import peak_bounds
from app.media_identity import metadata_streamer, source_streamer, streamer_name
from .artifacts import write_json

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RenderClip:
    rank: int
    name: str
    source: Path
    start: float
    end: float
    metadata: dict
    credit_source: str | None = None


def parse_ranks(value: str) -> set[int]:
    try:
        ranks = {int(part.strip()) for part in value.split(",")}
    except ValueError as exc:
        raise ValueError("ranks must be comma-separated positive integers") from exc
    if not ranks or min(ranks) <= 0:
        raise ValueError("ranks must be positive")
    return ranks


def selected_clips(output: Path, all_clips: bool = False, ranks: set[int] | None = None) -> list[RenderClip]:
    with (output / "review.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    clips: list[RenderClip] = []
    available: set[int] = set()
    names: set[str] = set()
    for index, row in enumerate(rows, 1):
        rank = int(row.get("rank") or index)
        available.add(rank)
        chosen = rank in ranks if ranks is not None else all_clips or row.get("approved", "").strip().lower() == "y"
        if not chosen:
            continue
        preview = Path(row["clip_path"])
        preview = preview if preview.is_absolute() else output / preview
        metadata = json.loads(preview.with_suffix(".json").read_text(encoding="utf-8"))
        source = Path(metadata["video"])
        source = source if source.is_absolute() else output / source
        start, end = float(metadata["start_seconds"]), float(metadata["end_seconds"])
        if not all(math.isfinite(value) for value in (start, end)) or not 0 <= start < end:
            raise ValueError(f"invalid source window for rank {rank}")
        if preview.suffix.lower() != ".mp4" or preview.name.casefold() in names:
            raise ValueError("render selection requires distinct MP4 basenames")
        names.add(preview.name.casefold())
        clips.append(RenderClip(rank, preview.name, source.resolve(), start, end, metadata))
    if ranks is not None and ranks - available:
        raise ValueError(f"unknown clip ranks: {sorted(ranks - available)}")
    return sorted(clips, key=lambda clip: clip.rank)


def clip_transcript(clip: RenderClip, output: Path, config: RenderConfig, info: MediaInfo,
                    work: Path, transcriber: Transcriber) -> tuple[Transcript, bool, float]:
    if not config.captions_enabled or not info.has_audio:
        return Transcript(()), False, 0.0
    started = time.perf_counter()
    key = transcript_key(clip.source, clip.start, clip.end, config)
    cache = output / ".transcripts" / f"{key}.json"
    existing = read_transcript(cache, key)
    if existing is not None:
        return existing, True, time.perf_counter() - started
    audio = work / "audio.wav"
    extract_audio(clip.source, audio, clip.start, clip.end - clip.start)
    transcript = transcriber.transcribe(audio)
    transcript = replace(transcript, audio_frames=reaction_frames(audio))
    save_transcript(cache, key, transcript)
    return transcript, False, time.perf_counter() - started


def bounded_render_clip(clip: RenderClip, info: MediaInfo) -> RenderClip:
    peak = clip.metadata.get("peak_seconds")
    if peak is None:
        events = clip.metadata.get("events", [])
        primary = max(events, key=lambda event: event.get("score", 0), default={})
        peak = primary.get("peak_time_seconds", (clip.start+clip.end)/2)
    peak = min(clip.end, max(clip.start, float(peak)))
    start, end = peak_bounds(clip.start, clip.end, peak, lower=0, upper=info.duration)
    metadata = {**clip.metadata, "original_start_seconds": clip.start, "original_end_seconds": clip.end,
                "peak_seconds": peak}
    return replace(clip, start=start, end=end, metadata=metadata)


def context_streamer(output: Path) -> str | None:
    path = output / "analysis.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        name = metadata_streamer(data)
        if name:
            return name
        chat = Path(data["chat"]) if data.get("chat") else None
        if chat is not None and chat.is_file():
            return metadata_streamer(json.loads(chat.read_text(encoding="utf-8-sig")))
    except (OSError, ValueError, TypeError):
        LOGGER.warning("Unable to read VOD attribution metadata")
    return None


def render_clip(clip: RenderClip, output: Path, config: RenderConfig,
                transcriber: Transcriber, selector: EncoderSelector) -> dict:
    begun = time.perf_counter()
    info = probe_media(clip.source)
    clip = bounded_render_clip(clip, info)
    duration = clip.end - clip.start
    final = output / "final"
    final.mkdir(exist_ok=True)
    path = final / clip.name
    with tempfile.TemporaryDirectory(prefix="r ", dir=final) as temporary:
        work = Path(temporary)
        transcript, cached, transcription_seconds = clip_transcript(clip, output, config, info, work, transcriber)
        words, rejected = usable_words(transcript, config, duration)
        words, reactions = add_reaction_tags(transcript, words, config) if config.captions_enabled else ([], [])
        caption_status = ("disabled" if not config.captions_enabled else "source_has_no_audio" if not info.has_audio
                          else "reaction_only" if reactions and all(word.kind == "reaction" for word in words)
                          else "captioned_with_reactions" if reactions else "captioned" if words else "no_usable_speech")
        if not words:
            LOGGER.info("Rank %d renders without captions: %s", clip.rank, caption_status)
        elif reactions:
            LOGGER.info("Rank %d uses %d heuristic reaction caption(s)", clip.rank, len(reactions))
        at = time.perf_counter()
        rendered = work / "final.mp4"
        layout, family, encoder = render_original(clip.source, rendered, clip.start, duration,
                                                 info, words, config, work, selector)
        actual = validate_export(rendered, duration)
        encoding_seconds = time.perf_counter() - at
        payload = dict(schema=1, rank=clip.rank, video=str(clip.source), start_seconds=clip.start, end_seconds=clip.end,
            clip_path=str(path), preview_metadata=clip.metadata, settings=asdict(config),
            effective_layout=layout.effective_layout, video_encoder=encoder, font_family=family,
            peak_seconds=clip.metadata["peak_seconds"], actual_pre_roll_seconds=clip.metadata["peak_seconds"]-clip.start,
            actual_post_roll_seconds=clip.end-clip.metadata["peak_seconds"],
            streamer=config.streamer, credit_text=credit_text(config), credit_source=clip.credit_source,
            credit_status="burned_in" if credit_text(config) else "disabled" if not config.credit_enabled else "missing_streamer",
            width=actual.width, height=actual.height, fps=actual.fps, actual_duration_seconds=actual.duration,
            caption_status=caption_status, caption_text=" ".join(display_word(word.text, config.uppercase) for word in words),
            reaction_tags=reactions,
            words=[dict(**asdict(word), source_start_seconds=clip.start + word.start,
                        source_end_seconds=clip.start + word.end) for word in words],
            transcription=dict(language=transcript.language, engine=transcript.engine, cache_hit=cached,
                               rejected_segments=rejected, seconds=transcription_seconds),
            encoding_seconds=encoding_seconds, total_seconds=time.perf_counter() - begun)
        write_json(work / "final.json", payload)
        if (work / "captions.ass").exists():
            (work / "captions.ass").replace(path.with_suffix(".ass"))
        else:
            path.with_suffix(".ass").unlink(missing_ok=True)
        rendered.replace(path)
        (work / "final.json").replace(path.with_suffix(".json"))
    LOGGER.info("Rendered rank %d: %.2fs in %.2fs (%.2fx), %s", clip.rank, duration,
                payload["total_seconds"], duration / max(payload["total_seconds"], 1e-9), path)
    return payload


def render_reviewed(output: Path, config_path: Path, *, all_clips: bool = False,
                    ranks: set[int] | None = None, transcriber: Transcriber | None = None,
                    streamer: str | None = None) -> dict:
    output = output.resolve()
    config = load_render_config(config_path)
    config.validate()
    if all_clips and ranks is not None:
        raise ValueError("choose --all or --ranks")
    with worker_lock(output / ".vod.lock"):
        clips = selected_clips(output, all_clips, ranks)
        context = context_streamer(output)
        recognizer = transcriber or WhisperTranscriber(config)
        selector = EncoderSelector(config.encoder)
        started = time.perf_counter()
        completed, failed = [], []
        for clip in clips:
            try:
                identities = (("cli", streamer_name(streamer)), ("config", streamer_name(config.streamer)),
                              ("clip_metadata", metadata_streamer(clip.metadata)), ("vod_metadata", context),
                              ("source_metadata", source_streamer(clip.source)))
                origin, name = next(((origin, name) for origin, name in identities if name), (None, None))
                if config.credit_enabled and not name:
                    LOGGER.warning("Rank %d has no streamer metadata; set --streamer to burn in credit", clip.rank)
                completed.append(render_clip(replace(clip, credit_source=origin), output,
                                              replace(config, streamer=name), recognizer, selector))
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                details = getattr(exc, "stderr", b"")
                details = details.decode("utf-8", errors="replace") if isinstance(details, bytes) else details
                LOGGER.error("Rank %d render failed: %s %s", clip.rank, exc, details or "")
                failed.append(dict(rank=clip.rank, error=f"{exc} {details or ''}"))
        seconds = time.perf_counter() - started
        footage = sum(clip["end_seconds"] - clip["start_seconds"] for clip in completed)
        report = dict(selected=len(clips), rendered=len(completed), failed=failed, processing_seconds=seconds,
                      footage_seconds=footage, realtime_factor=footage / max(seconds, 1e-9),
                      clips=[clip["clip_path"] for clip in completed])
        write_json(output / "render-report.json", report)
        LOGGER.info("Final render: %d/%d clips, %d failed, %.2fx real time", len(completed), len(clips), len(failed), report["realtime_factor"])
        return report
