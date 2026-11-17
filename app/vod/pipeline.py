"""Bounded VOD analysis and durable, independently repeatable rescoring."""

import csv
from dataclasses import asdict, dataclass, replace
from datetime import timedelta
import hashlib
import json
import logging
from pathlib import Path
import shutil
import time

from app.clips.candidate_extractor import worker_lock
from app.clips.config import CandidateConfig
from app.clips.render import EncoderSelector, render_cut
from app.clips.windows import Event, Window, event_window, merge_windows
from app.detection.config import load_detection_config
from app.detection.models import DetectedEvent, FeatureBin
from app.detection.scoring import AuxiliaryFeatures, pick_events, score_feature_bins
from app.ingestion.stream_reader import probe_duration
from .artifacts import REVIEW_FIELDS, read_features, write_features, write_index, write_json, write_review
from .chat import load_chat, load_chat_config
from .chunked import extract_chunked
from .memory import PeakMemory
from .selection import peak_window, rank_peaks
from .settings import load_settings
from .timeline import EPOCH, media_seconds, media_time

LOGGER = logging.getLogger(__name__)


def event_data(event: DetectedEvent, event_id: int) -> dict[str, object]:
    return dict(id=event_id, source_id=event.source_id, session_id=event.session_id,
                peak_time_seconds=media_seconds(event.peak_wall_utc),
                start_time_seconds=media_seconds(event.start_wall_utc),
                end_time_seconds=media_seconds(event.end_wall_utc), score=event.score, components=event.components,
                unclipped_z_scores=event.unclipped_z_scores, unclipped_score=event.unclipped_score)


def plan_windows(events: list[DetectedEvent], config: CandidateConfig, duration: float) -> list[Window]:
    """Legacy explicit-config planner retained for existing API consumers.

    The ranked VOD path uses peak_window and VOD-only defaults instead.
    """
    windows = []
    for index, event in enumerate(events, 1):
        value = Event(index, event.source_id, event.session_id, event.peak_wall_utc,
                      event.start_wall_utc, event.end_wall_utc, event.score)
        window = event_window(value, config)
        start, end = max(EPOCH, window.start), min(media_time(duration), window.end)
        if (end - start).total_seconds() < config.min_seconds:
            start = max(EPOCH, end - timedelta(seconds=config.min_seconds))
            end = min(media_time(duration), max(end, start + timedelta(seconds=config.min_seconds)))
        if start <= event.peak_wall_utc < end and (end - start).total_seconds() >= config.min_seconds:
            windows.append(replace(window, start=start, end=end))
    return merge_windows(windows, config)


@dataclass(frozen=True, slots=True)
class RunResult:
    duration_seconds: float
    processing_seconds: float
    detection_seconds: float
    event_count: int
    clip_count: int

    @property
    def realtime_factor(self) -> float:
        return self.duration_seconds / max(self.processing_seconds, 1e-9)


def _read_json(path: Path, default: object) -> object:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _fingerprint(video: Path) -> dict[str, object]:
    stat = video.stat()
    return dict(path=str(video), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def make_render_provenance(
    source_fingerprint: dict[str, object],
    left: float,
    right: float,
    requested_encoder: str,
    effective_encoder: str,
    render_version: str = "1.0",
) -> dict[str, object]:
    codec_settings = (
        {"preset": "veryfast", "crf": 18}
        if effective_encoder == "libx264"
        else {"preset": "p4", "cq": 18}
    )
    return {
        "render_version": render_version,
        "source_fingerprint": source_fingerprint,
        "start_seconds": round(left, 6),
        "end_seconds": round(right, 6),
        "duration_seconds": round(right - left, 6),
        "requested_encoder": requested_encoder,
        "effective_encoder": effective_encoder,
        "video_codec": effective_encoder,
        "codec_settings": codec_settings,
        "pix_fmt": "yuv420p",
        "audio_codec": "aac",
        "container_flags": ["+faststart"],
    }


def make_render_cache_key(provenance: dict[str, object]) -> str:
    serialized = json.dumps(provenance, sort_keys=True, allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:24]


def _produce(output: Path, config_path: Path, analysis: dict, features: list[FeatureBin],
             auxiliary: AuxiliaryFeatures | None, extract: bool, timings: dict[str, float]) -> tuple[list[dict], int, str | None]:
    detection = load_detection_config(config_path)
    vod, selection, windows = load_settings(config_path)
    at = time.perf_counter()
    scored = score_feature_bins(features, detection, auxiliary, include_unclipped=True)
    threshold_events = pick_events(scored, detection)
    timings["scoring_seconds"] = time.perf_counter() - at
    at = time.perf_counter()
    selected = rank_peaks(scored, selection, detection, windows, analysis["start_seconds"], analysis["end_seconds"])
    timings["selection_seconds"] = time.perf_counter() - at
    LOGGER.info("Ranked %d peaks (%d threshold events); exporting=%s", len(selected), len(threshold_events), extract)
    write_json(output / "events.json", dict(video=analysis["video"], duration_seconds=analysis["source_duration_seconds"],
        start_seconds=analysis["start_seconds"], end_seconds=analysis["end_seconds"],
        events=[dict(**event_data(event, rank), rank=rank, rule=rule) for rank, (event, rule) in enumerate(selected, 1)],
        threshold_events=[event_data(event, index) for index, event in enumerate(threshold_events, 1)]))
    previous_rows = []
    if (output / "review.csv").exists():
        with (output / "review.csv").open(newline="", encoding="utf-8") as stream:
            previous_rows = list(csv.DictReader(stream))
    previous = {row.get("window_id", row["clip_path"]): row for row in previous_rows}
    cache: dict = _read_json(output / "clip_cache.json", {})
    selector = EncoderSelector(windows.requested_encoder)
    clips_dir = output / "clips"
    clips_dir.mkdir(exist_ok=True)
    clips, rows = [], []
    at = time.perf_counter()
    for rank, (event, rule) in enumerate(selected, 1):
        left, right = peak_window(event, windows, analysis["start_seconds"], analysis["end_seconds"])
        # Editorial candidate identity (stable window_id across reviews)
        key = hashlib.sha256(f'{analysis["fingerprint"]}:{left:.6f}:{right:.6f}'.encode()).hexdigest()[:24]

        # Determine effective encoder and render provenance for this candidate
        effective_encoder = selector.select(Path(analysis["video"]), left)
        expected_provenance = make_render_provenance(
            analysis["fingerprint"], left, right, windows.requested_encoder, effective_encoder
        )
        expected_render_key = make_render_cache_key(expected_provenance)

        peak = media_seconds(event.peak_wall_utc)
        hours, remainder = divmod(int(peak), 3600)
        minutes, seconds = divmod(remainder, 60)
        clip_path = clips_dir / f"clip_{rank:02d}_{hours:02d}-{minutes:02d}-{seconds:02d}_{event.score:.2f}.mp4"
        cached = cache.get(key, {})
        has_modern_provenance = (
            cached.get("render_cache_key") == expected_render_key
            and cached.get("provenance") == expected_provenance
        )
        is_compatible_legacy = (
            "render_cache_key" not in cached
            and "provenance" not in cached
            and cached.get("video_encoder") == effective_encoder
            and (windows.requested_encoder in ("auto", cached.get("video_encoder")))
        )
        has_matching_provenance = has_modern_provenance or is_compatible_legacy
        old_file = Path(cached.get("clip_path", "."))
        reused = (
            has_matching_provenance
            and old_file.is_file()
            and old_file.stat().st_size == cached.get("size_bytes")
            and (cached.get("size_bytes") or 0) > 0
        )
        extracted = False
        encoder = cached.get("video_encoder") if reused else None
        if reused:
            if old_file != clip_path:
                temporary = clip_path.with_suffix(".reuse.mp4")
                shutil.copyfile(old_file, temporary)
                temporary.replace(clip_path)
            extracted = True
        elif extract:
            if _fingerprint(Path(analysis["video"])) != analysis["fingerprint"]:
                raise ValueError("source file changed since analysis")
            temporary = clip_path.with_suffix(".tmp.mp4")
            try:
                render_cut(Path(analysis["video"]), temporary, left, right - left, encoder_selector=selector)
                actual = probe_duration(temporary)
                if abs(actual - (right - left)) > 0.15:
                    raise ValueError(f"clip duration {actual} differs from requested {right-left}")
                temporary.replace(clip_path)
                extracted, encoder = True, selector.selected
            finally:
                temporary.unlink(missing_ok=True)
        clip = dict(window_id=key, rank=rank, video=analysis["video"], clip_path=str(clip_path),
                    source_id=event.source_id, peak_seconds=peak, start_seconds=left, end_seconds=right,
                    actual_pre_roll_seconds=peak-left, actual_post_roll_seconds=right-peak,
                    score=event.score, components=event.components, rule=rule,
                    unclipped_z_scores=event.unclipped_z_scores, unclipped_score=event.unclipped_score,
                    events=[event_data(event, rank)], video_encoder=encoder, extracted=extracted, reused=reused,
                    render_cache_key=expected_render_key if (has_modern_provenance or extracted) else None)
        write_json(clip_path.with_suffix(".json"), clip)
        if extracted:
            cache_dir = output / ".clip-cache"
            cache_dir.mkdir(exist_ok=True)
            cached_file = cache_dir / f"{expected_render_key}.mp4"
            if not reused or old_file != cached_file:
                shutil.copyfile(clip_path, cached_file)
            cache[key] = dict(
                clip_path=str(cached_file),
                size_bytes=cached_file.stat().st_size,
                video_encoder=encoder,
                render_cache_key=expected_render_key,
                provenance=expected_provenance,
            )
        old = previous.get(key, {})
        top = "; ".join(f"{name}={value:.3f}" for name, value in sorted(event.components.items(), key=lambda item: -item[1])[:3])
        rows.append(dict(clip_path=str(clip_path), start_seconds=str(left), end_seconds=str(right), score=str(event.score),
            top_features=top, approved=old.get("approved", ""), notes=old.get("notes", ""), reason_code=old.get("reason_code", ""),
            window_id=key, rank=str(rank), peak_seconds=str(peak), rule=rule,
            unclipped_z_scores=json.dumps(event.unclipped_z_scores, sort_keys=True, allow_nan=False),
            unclipped_score=str(event.unclipped_score)))
        clips.append(clip)
        write_review(output / "review.csv", rows)
        write_json(output / "clip_cache.json", cache)
    timings["extraction_seconds"] = time.perf_counter() - at
    write_review(output / "review.csv", rows)
    write_index(output, clips)
    write_json(output / "clips.json", clips)
    return clips, len(threshold_events), selector.selected


def _report(output: Path, analysis: dict, started: float, timings: dict[str, float], memory: PeakMemory,
            clips: list[dict], threshold_count: int, selected_encoder: str | None, config_path: Path) -> RunResult:
    duration = analysis["end_seconds"] - analysis["start_seconds"]
    detection_seconds = sum(value for key, value in timings.items() if key != "extraction_seconds")
    result = RunResult(duration, time.perf_counter() - started, detection_seconds, len(clips), sum(clip["extracted"] for clip in clips))
    dominant = max(timings, key=timings.get) if timings else "none"
    vod, selection, candidates = load_settings(config_path)
    write_json(output / "run.json", dict(**analysis, **asdict(result), stage_timings=timings,
        dominant_stage=dominant, peak_memory_bytes=memory.peak_bytes,
        memory_measurement="peak sampled parent + child RSS, 100ms interval",
        timing_measurement="decode times measure FFmpeg pipe reads; feature computation measured separately",
        realtime_factor=result.realtime_factor, detection_realtime_factor=duration/max(detection_seconds, 1e-9),
        threshold_event_count=threshold_count, selected_video_encoder=selected_encoder,
        detection_config=asdict(load_detection_config(config_path)), vod_config=asdict(vod),
        selection_config=asdict(selection), candidates_config=asdict(candidates)))
    LOGGER.info("Processed %.3fs in %.3fs: %.2fx; %d ranked clips, %d available; peak sampled RSS %.1f MiB",
                duration, result.processing_seconds, result.realtime_factor, len(clips), result.clip_count, memory.peak_bytes / 2**20)
    LOGGER.info("Per-stage seconds: %s; dominant=%s (video decode dominant=%s)", timings, dominant, dominant == "video_decode_seconds")
    return result


def run(video: Path, output: Path, config_path: Path, chat: Path | None = None,
        start: float = 0, end: float | None = None) -> RunResult:
    video, output = video.resolve(), output.resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    output.mkdir(parents=True, exist_ok=True)
    with worker_lock(output / ".vod.lock"), PeakMemory() as memory:
        started = time.perf_counter()
        config = load_detection_config(config_path)
        vod, _, _ = load_settings(config_path)
        duration = probe_duration(video)
        end = duration if end is None else end
        if not 0 <= start < end <= duration + 1e-6:
            raise ValueError("require 0 <= start < end <= source duration")
        source_id = hashlib.sha256(str(video).encode()).hexdigest()[:16]
        features, timings = extract_chunked(video, config, source_id, start, end, vod.chunk_seconds, vod.progress_seconds)
        at = time.perf_counter()
        auxiliary = load_chat(chat, features, load_chat_config(config_path)) if chat else None
        timings["chat_seconds"] = time.perf_counter() - at
        write_features(output / "features.csv", features, auxiliary)
        analysis = dict(video=str(video), source_id=source_id, fingerprint=_fingerprint(video), source_duration_seconds=duration,
                        start_seconds=start, end_seconds=end, chunk_seconds=vod.chunk_seconds,
                        sample_rate=config.sample_rate, video_fps=config.video_fps, feature_schema_version=1,
                        time_model="1970-01-01 UTC + original-file media seconds", chat=str(chat.resolve()) if chat else None,
                        chat_time_offset_seconds=load_chat_config(config_path).time_offset_seconds if chat else None)
        write_json(output / "analysis.json", analysis)
        clips, threshold_count, encoder = _produce(output, config_path, analysis, features, auxiliary, True, timings)
        return _report(output, analysis, started, timings, memory, clips, threshold_count, encoder, config_path)


def rescore(output: Path, config_path: Path, extract: bool = False) -> RunResult:
    output = output.resolve()
    with worker_lock(output / ".vod.lock"), PeakMemory() as memory:
        started = time.perf_counter()
        analysis = _read_json(output / "analysis.json", None)
        if analysis is None:
            raise ValueError("analysis.json missing; run VOD analysis once to persist source metadata")
        chat_config = load_chat_config(config_path)
        if analysis.get("chat") and chat_config.time_offset_seconds != analysis.get("chat_time_offset_seconds"):
            raise ValueError("chat offset changed; run analysis again to re-align raw chat bins")
        detection = load_detection_config(config_path)
        if analysis.get("sample_rate", detection.sample_rate) != detection.sample_rate or analysis.get("video_fps", detection.video_fps) != detection.video_fps:
            raise ValueError("decode sample rate or video fps changed; run analysis again")
        features, auxiliary = read_features(output / "features.csv", analysis["source_id"], chat_config)
        timings = {}
        clips, threshold_count, encoder = _produce(output, config_path, analysis, features, auxiliary, extract, timings)
        return _report(output, analysis, started, timings, memory, clips, threshold_count, encoder, config_path)
