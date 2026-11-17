"""Structured candidate context extraction (visual sampling, transcript, chat)."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import math
from pathlib import Path
import subprocess
from typing import Any, Sequence

from app.ingestion.stream_reader import probe_duration
from app.vod.pipeline import _read_json
from .dataset import compute_source_fingerprint
from .models import (
    CandidateContext,
    ChatContext,
    SpeechContext,
    TranscriptSegmentContext,
    TranscriptWordContext,
    VisualFrame,
    utc_iso_now,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FrameSamplerConfig:
    """Settings controlling visual frame extraction for context packages."""

    max_frames: int = 16
    uniform_interval_seconds: float = 2.0
    max_width: int = 640
    max_height: int = 360
    jpeg_quality: int = 85
    include_hook_frame: bool = True
    include_boundary_frames: bool = True
    include_peak_frame: bool = True
    min_frame_separation_seconds: float = 0.4


def plan_frame_timestamps(
    start_seconds: float,
    end_seconds: float,
    peak_seconds: float | None = None,
    cut_timestamps: Sequence[float] = (),
    config: FrameSamplerConfig = FrameSamplerConfig(),
) -> list[tuple[float, str]]:
    """Determine deterministic timestamp plan and frame roles for extraction."""
    planned: list[tuple[float, str]] = []

    if config.include_boundary_frames:
        planned.append((round(start_seconds, 3), "boundary_start"))

    if config.include_hook_frame and (start_seconds + 0.5 < end_seconds):
        planned.append((round(start_seconds + 0.5, 3), "hook"))

    if config.include_peak_frame and peak_seconds is not None:
        if start_seconds <= peak_seconds <= end_seconds:
            planned.append((round(peak_seconds, 3), "peak"))

    for cut_t in cut_timestamps:
        if start_seconds < cut_t < end_seconds:
            planned.append((round(cut_t, 3), "scene_cut"))

    # Uniform sampling
    curr = start_seconds + config.uniform_interval_seconds
    while curr < end_seconds - 0.2:
        planned.append((round(curr, 3), "uniform"))
        curr += config.uniform_interval_seconds

    if config.include_boundary_frames and (end_seconds - 0.1 > start_seconds):
        planned.append((round(end_seconds - 0.1, 3), "boundary_end"))

    # Sort and deduplicate with minimum separation
    planned.sort(key=lambda item: item[0])
    filtered: list[tuple[float, str]] = []
    for ts, role in planned:
        if not filtered:
            filtered.append((ts, role))
        else:
            prev_ts, prev_role = filtered[-1]
            if ts - prev_ts >= config.min_frame_separation_seconds:
                filtered.append((ts, role))
            elif role in ("peak", "hook") and prev_role == "uniform":
                # Prioritize peak/hook over uniform
                filtered[-1] = (ts, role)

    # Clamp to max_frames while strictly protecting essential anchor frames
    if len(filtered) > config.max_frames:
        essential_roles = {"boundary_start", "boundary_end", "peak", "hook", "scene_cut"}
        essential = [item for item in filtered if item[1] in essential_roles]
        non_essential = [item for item in filtered if item[1] not in essential_roles]

        if len(essential) <= config.max_frames:
            slots_for_others = config.max_frames - len(essential)
            if slots_for_others > 0 and non_essential:
                step = len(non_essential) / slots_for_others
                chosen_non_essential = [
                    non_essential[int(i * step)] for i in range(slots_for_others)
                ]
            else:
                chosen_non_essential = []
            filtered = sorted(essential + chosen_non_essential, key=lambda item: item[0])
        else:
            step = len(filtered) / config.max_frames
            indices = [int(i * step) for i in range(config.max_frames)]
            filtered = [filtered[i] for i in indices]

    return filtered


def extract_visual_frames(
    source_video: Path,
    timestamps: Sequence[tuple[float, str]],
    output_dir: Path,
    config: FrameSamplerConfig = FrameSamplerConfig(),
) -> list[VisualFrame]:
    """Extract sampled JPEG frames using FFmpeg, preserving exact source timestamps."""
    output_dir.mkdir(parents=True, exist_ok=True)
    frames: list[VisualFrame] = []

    scale_filter = f"scale=w='min(iw,{config.max_width})':h='min(ih,{config.max_height})':force_original_aspect_ratio=decrease"

    for index, (ts, role) in enumerate(timestamps):
        out_filename = f"frame_{index:02d}_{role}_{ts:.2f}s.jpg"
        out_path = output_dir / out_filename
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{ts:.3f}",
            "-i",
            str(source_video),
            "-frames:v",
            "1",
            "-vf",
            scale_filter,
            "-q:v",
            "3",
            str(out_path),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=30)
            frames.append(
                VisualFrame(
                    timestamp=ts,
                    frame_path=str(out_path.name),
                    frame_type=role,
                    width=config.max_width,
                    height=config.max_height,
                )
            )
        except Exception as exc:
            LOGGER.warning("Failed to extract frame at %fs (%s): %s", ts, role, exc)

    return frames


def extract_speech_context(
    candidate_start: float,
    candidate_end: float,
    transcripts_dir: Path | None = None,
) -> SpeechContext:
    """Slice available faster-whisper transcripts to candidate interval."""
    if transcripts_dir is None or not transcripts_dir.is_dir():
        return SpeechContext(has_speech=False, provenance="none")

    # Search for cached transcripts in directory
    transcript_files = list(transcripts_dir.glob("*.json"))
    if not transcript_files:
        return SpeechContext(has_speech=False, provenance="none")

    # Ingest word & segment entries that overlap candidate window
    selected_words: list[TranscriptWordContext] = []
    selected_segments: list[TranscriptSegmentContext] = []
    detected_lang: str | None = None

    for t_file in transcript_files:
        try:
            data = _read_json(t_file, {})
            if not isinstance(data, dict):
                continue
            detected_lang = detected_lang or data.get("language")
            for seg in data.get("segments", []):
                s_start = float(seg.get("start", 0))
                s_end = float(seg.get("end", 0))
                if s_end > candidate_start and s_start < candidate_end:
                    words_in_seg = []
                    for w in seg.get("words", []):
                        w_start = float(w.get("start", 0))
                        w_end = float(w.get("end", 0))
                        if candidate_start <= w_start <= candidate_end:
                            w_ctx = TranscriptWordContext(
                                word=str(w.get("word", "")).strip(),
                                start=w_start,
                                end=w_end,
                                probability=float(w.get("probability", 1.0)),
                            )
                            words_in_seg.append(w_ctx)
                            selected_words.append(w_ctx)
                    selected_segments.append(
                        TranscriptSegmentContext(
                            text=str(seg.get("text", "")).strip(),
                            start=s_start,
                            end=s_end,
                            avg_logprob=float(seg.get("avg_logprob", 0.0)),
                            no_speech_prob=float(seg.get("no_speech_prob", 0.0)),
                            words=tuple(words_in_seg),
                        )
                    )
        except Exception as exc:
            LOGGER.debug("Skipping unreadable transcript file %s: %s", t_file, exc)

    has_speech = len(selected_words) > 0
    return SpeechContext(
        language=detected_lang,
        segments=selected_segments,
        words=selected_words,
        has_speech=has_speech,
        provenance="cache" if has_speech else "none",
    )


def extract_chat_context(
    chat_path: Path | None,
    candidate_start: float,
    candidate_end: float,
    time_offset: float = 0.0,
) -> ChatContext:
    """Extract and aggregate Twitch chat comments during candidate interval."""
    if chat_path is None or not chat_path.is_file():
        return ChatContext(present=False)

    try:
        raw_data = json.loads(chat_path.read_text(encoding="utf-8"))
        comments = raw_data.get("comments", []) if isinstance(raw_data, dict) else raw_data
    except Exception as exc:
        LOGGER.warning("Failed to parse chat data at %s: %s", chat_path, exc)
        return ChatContext(present=False)

    start_aligned = candidate_start + time_offset
    end_aligned = candidate_end + time_offset

    window_comments = []
    unique_users = set()
    total_emotes = 0
    total_letters = 0
    caps_letters = 0

    for c in comments:
        offset = float(c.get("content_offset_seconds", 0.0))
        if start_aligned <= offset <= end_aligned:
            user = c.get("commenter", {}).get("display_name") or c.get("commenter", {}).get("name") or "unknown"
            unique_users.add(user)
            body = str(c.get("message", {}).get("body", ""))
            emotes = c.get("message", {}).get("emotes", [])
            total_emotes += len(emotes)
            for ch in body:
                if ch.isalpha():
                    total_letters += 1
                    if ch.isupper():
                        caps_letters += 1
            window_comments.append({
                "offset_seconds": round(offset - time_offset, 3),
                "author": user,
                "body": body,
            })

    message_count = len(window_comments)
    duration = max(1.0, candidate_end - candidate_start)
    emote_density = total_emotes / duration
    caps_ratio = (caps_letters / total_letters) if total_letters > 0 else 0.0

    return ChatContext(
        present=True,
        message_count=message_count,
        unique_chatters=len(unique_users),
        emote_density=round(emote_density, 4),
        caps_ratio=round(caps_ratio, 4),
        messages=window_comments[:50],  # cap embedded messages
    )


def build_candidate_context(
    candidate: dict[str, Any],
    source_video: Path,
    output_dir: Path,
    *,
    features_csv: Path | None = None,
    transcripts_dir: Path | None = None,
    chat_path: Path | None = None,
    sampler_config: FrameSamplerConfig = FrameSamplerConfig(),
) -> CandidateContext:
    """Build, persist, and return a complete multimodal CandidateContext package."""
    source_video = source_video.resolve()
    candidate_id = str(candidate.get("window_id", candidate.get("candidate_id", "unknown")))
    start = float(candidate["start_seconds"])
    end = float(candidate["end_seconds"])
    peak = float(candidate.get("peak_seconds", (start + end) / 2))

    package_dir = output_dir / candidate_id
    frames_dir = package_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # 1. Identify scene cuts from features.csv if available
    cut_timestamps: list[float] = []
    if features_csv is not None and features_csv.is_file():
        try:
            import csv
            with features_csv.open(encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    t = float(row.get("wall_seconds", row.get("time_seconds", 0)))
                    if start <= t <= end and float(row.get("cut_change", 0)) > 2.5:
                        cut_timestamps.append(t)
        except Exception as exc:
            LOGGER.debug("Could not parse cuts from features.csv: %s", exc)

    # 2. Plan and extract frames
    planned = plan_frame_timestamps(start, end, peak, cut_timestamps, sampler_config)
    frames = extract_visual_frames(source_video, planned, frames_dir, sampler_config)

    # 3. Extract speech context
    speech = extract_speech_context(start, end, transcripts_dir)

    # 4. Extract chat context
    chat = extract_chat_context(chat_path, start, end)

    # 5. Assemble and serialize CandidateContext
    context = CandidateContext(
        schema_version=1,
        candidate_id=candidate_id,
        source_video=str(source_video),
        source_fingerprint=compute_source_fingerprint(source_video),
        start_seconds=start,
        end_seconds=end,
        duration_seconds=round(end - start, 6),
        peak_seconds=peak,
        candidate_score=float(candidate.get("score", 0.0)),
        components=dict(candidate.get("components", {})),
        unclipped_score=candidate.get("unclipped_score"),
        unclipped_z_scores=dict(candidate.get("unclipped_z_scores", {})),
        rule=str(candidate.get("rule", "single")),
        frames=frames,
        speech=speech,
        chat=chat,
    )

    context_json = package_dir / "context.json"
    context_json.write_text(json.dumps(context.to_dict(), indent=2), encoding="utf-8")
    LOGGER.info("Built context package for candidate %s at %s (%d frames)", candidate_id, context_json, len(frames))
    return context
