"""Single-Stream Live Intelligence Pipeline Orchestrator."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
import threading
import time
from typing import Any, Sequence

from app.clips.candidate_extractor import CandidateExtractor
from app.clips.config import CandidateConfig
from app.clips.store import Candidate, CandidateStore
from app.detection.config import DetectionConfig
from app.detection.runner import DetectionRunner
from app.evaluation.context import FrameSamplerConfig, build_candidate_context
from app.evaluation.inference import HighlightJudgeProvider, MockHighlightJudgeProvider
from app.evaluation.models import CandidateContext
from app.evaluation.providers.qwen import QwenProviderConfig, QwenVisionLanguageProvider
from app.evaluation.schema import AIJudgmentRecord, EvidenceItem
from app.rendering.config import RenderConfig
from app.rendering.renderer import probe_media, render_original
from app.clips.render import EncoderSelector
from .config import LiveSourceConfig
from .gpu import GpuResourceManager
from .queue import LiveJobRecord, LiveJobStatus, LiveJudgmentQueue
from .review import LiveReviewStore
from .supervisor import LiveState, LiveSupervisor

LOGGER = logging.getLogger(__name__)


class SingleStreamLivePipeline:
    """Coordinates single-stream capture, detection, candidate extraction, AI judging, and rendering."""

    def __init__(
        self,
        config: LiveSourceConfig,
        provider: HighlightJudgeProvider | None = None,
        *,
        base_dir: Path | None = None,
        ffmpeg_binary: str = "ffmpeg",
        ffprobe_binary: str = "ffprobe",
    ) -> None:
        self.config = config
        self.base_dir = base_dir or Path(".")
        self.ffmpeg_binary = ffmpeg_binary
        self.ffprobe_binary = ffprobe_binary

        # Directories
        self.segments_dir = self.config.capture.segments_dir
        self.manifest_db = self.segments_dir / "manifest.sqlite3"
        self.candidates_dir = (self.base_dir / "data" / "live_candidates").resolve()
        self.context_dir = (self.base_dir / "data" / "live_context").resolve()
        self.judgments_dir = (self.base_dir / "data" / "live_judgments").resolve()
        self.shorts_dir = (self.base_dir / "data" / "live_shorts").resolve()

        for d in (self.segments_dir, self.candidates_dir, self.context_dir, self.judgments_dir, self.shorts_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Subsystems
        self.supervisor = LiveSupervisor(
            config=self.config,
            ffmpeg_binary=self.ffmpeg_binary,
            ffprobe_binary=self.ffprobe_binary,
        )

        # Detection follower
        det_cfg = DetectionConfig()
        self.detection_runner = DetectionRunner(
            manifest_path=self.manifest_db,
            config=det_cfg,
            ffmpeg_binary=self.ffmpeg_binary,
        )

        # Candidate extractor
        cand_cfg = CandidateConfig(
            output_dir=self.candidates_dir,
            min_seconds=5.0,
            grace_seconds=0.5,
        )
        self.candidate_extractor = CandidateExtractor(
            segments_dir=self.segments_dir,
            config=cand_cfg,
            retention_seconds=self.config.capture.retention_seconds,
            ffmpeg_binary=self.ffmpeg_binary,
            ffprobe_binary=self.ffprobe_binary,
        )
        self.candidate_store = CandidateStore(self.manifest_db)

        # Queue, review store, and GPU manager
        self.queue = LiveJudgmentQueue(
            db_path=self.manifest_db,
            max_queue_depth=self.config.queue.max_queue_depth,
            max_attempts=self.config.queue.max_attempts,
        )
        self.review_store = LiveReviewStore(db_path=self.manifest_db)
        self.gpu_manager = GpuResourceManager(
            serialize_vlm_and_render=self.config.gpu.serialize_vlm_and_render
        )

        # AI Judge Provider
        self.provider = provider or QwenVisionLanguageProvider(QwenProviderConfig())

        # Threading controls
        self._stop_event = threading.Event()
        self._pipeline_thread: threading.Thread | None = None
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        return bool(self._pipeline_thread and self._pipeline_thread.is_alive())

    def start(self, media_override: str | None = None) -> None:
        """Start supervisor capture and background pipeline processing loop."""
        with self._lock:
            if self.is_running:
                raise RuntimeError("SingleStreamLivePipeline is already running")

            self._stop_event.clear()
            # 1. Start supervisor capture
            self.supervisor.start(media_override=media_override)

            # 2. Recover stale queue jobs from any prior interrupted session
            self.queue.recover_stale_jobs(timeout_seconds=self.config.queue.stale_job_timeout_seconds)

            # 3. Launch background pipeline orchestrator thread
            self._pipeline_thread = threading.Thread(
                target=self._orchestration_loop,
                name="live-pipeline-orchestrator",
                daemon=True,
            )
            self._pipeline_thread.start()
            LOGGER.info("SingleStreamLivePipeline started for source %s", self.config.id)

    def stop(self, timeout_seconds: float = 15.0) -> None:
        """Gracefully stop pipeline and supervisor."""
        self._stop_event.set()

        if self._pipeline_thread and self._pipeline_thread.is_alive():
            self._pipeline_thread.join(timeout=5.0)

        self.supervisor.stop(timeout_seconds=timeout_seconds)

        if hasattr(self.provider, "close"):
            try:
                self.provider.close()
            except Exception as exc:
                LOGGER.warning("Error closing provider: %s", exc)

        LOGGER.info("SingleStreamLivePipeline stopped successfully")

    def step(self) -> dict[str, int]:
        """Perform one deterministic, synchronous pass across all pipeline stages.
        
        Returns summary stats of operations performed.
        """
        stats = {
            "segments_analyzed": 0,
            "candidates_extracted": 0,
            "contexts_generated": 0,
            "jobs_judged": 0,
            "shorts_rendered": 0,
        }

        # 1. Detection Stage: follow completed segments
        if self.config.processing.detection_enabled:
            det_stats = self.detection_runner.run_once()
            stats["segments_analyzed"] = det_stats.segments_analyzed

        # 2. Candidate Extraction Stage: follow detection events and extract clips
        candidates = self.candidate_extractor.run_once()
        extracted_candidates = [c for c in candidates if c.status == "extracted" and c.clip_path]
        stats["candidates_extracted"] = len(extracted_candidates)

        # 3. Context Generation & Queue Enqueue Stage
        for rank_idx, cand in enumerate(extracted_candidates, 1):
            clip_file = Path(cand.clip_path) if cand.clip_path else None
            if not clip_file or not clip_file.is_file():
                continue

            ctx_file = self.context_dir / cand.id / "context.json"
            if not ctx_file.is_file():
                try:
                    clip_duration = probe_media(clip_file, self.ffprobe_binary).duration
                    candidate_dict = {
                        "candidate_id": cand.id,
                        "window_id": cand.id,
                        "start_seconds": 0.0,
                        "end_seconds": clip_duration,
                        "peak_seconds": clip_duration / 2.0,
                        "score": cand.window.primary.score if cand.window and cand.window.primary else 3.0,
                    }
                    sampler = FrameSamplerConfig(max_frames=8)
                    build_candidate_context(
                        candidate=candidate_dict,
                        source_video=clip_file,
                        output_dir=self.context_dir,
                        sampler_config=sampler,
                    )
                    stats["contexts_generated"] += 1
                except Exception as exc:
                    LOGGER.warning("Could not generate context for candidate %s: %s", cand.id, exc)
                    continue

            # Record initial candidate in review store
            score = cand.window.primary.score if cand.window and cand.window.primary else 0.0
            self.review_store.record_candidate(
                candidate_id=cand.id,
                source_id=self.config.id,
                session_id=cand.window.primary.session_id if cand.window and cand.window.primary else "session",
                signal_score=score,
                signal_rank=rank_idx,
                clip_path=str(clip_file.resolve()),
            )

            # Enqueue in persistent AI judgment queue
            if self.config.processing.ai_judging_enabled:
                self.queue.enqueue(
                    candidate_id=cand.id,
                    source_id=self.config.id,
                    session_id=cand.window.primary.session_id if cand.window and cand.window.primary else "session",
                    context_path=ctx_file,
                    priority=score,
                    model_fingerprint=self.provider.model_name,
                )

        # 4. AI Judgment Worker Stage: claim and process next queued job
        if self.config.processing.ai_judging_enabled:
            job = self.queue.claim_next()
            if job:
                try:
                    ctx_path = Path(job.context_path)
                    if not ctx_path.is_file():
                        raise FileNotFoundError(f"Context package missing at {ctx_path}")

                    ctx_data = json.loads(ctx_path.read_text(encoding="utf-8"))
                    cand_ctx = CandidateContext.from_dict(ctx_data)

                    # Acquire conservative GPU inference slot
                    with self.gpu_manager.acquire_inference_slot(timeout_seconds=60.0):
                        if hasattr(self.provider, "judge_candidate_record"):
                            record = self.provider.judge_candidate_record(cand_ctx)
                        else:
                            hj = self.provider.evaluate_candidate(cand_ctx)
                            record = AIJudgmentRecord(
                                candidate_id=job.candidate_id,
                                event_category=hj.event_category,
                                summary=hj.reasoning,
                                editorial_rating=hj.editorial_rating,
                                confidence=hj.confidence,
                                context_complete=True,
                                setup_present=True,
                                payoff_present=True,
                                evidence=[
                                    EvidenceItem(
                                        timestamp=cand_ctx.peak_seconds,
                                        source="visual" if cand_ctx.components.get("motion", 0) > 1.0 else "audio",
                                        observation=hj.reasoning[:120],
                                    )
                                ],
                                hook_suggestion=hj.hook_text,
                                recommended_start_seconds=cand_ctx.start_seconds,
                                recommended_end_seconds=cand_ctx.end_seconds,
                                recommended_layout=hj.recommended_layout,
                            )

                    # Persist judgment JSON
                    judgment_path = self.judgments_dir / f"{job.candidate_id}.json"
                    judgment_dict = record.to_dict()
                    judgment_path.write_text(json.dumps(judgment_dict, indent=2), encoding="utf-8")

                    # Mark job completed in queue
                    self.queue.mark_completed(job.job_id, judgment_dict)
                    stats["jobs_judged"] += 1

                    # Update review store with AI score
                    self.review_store.record_candidate(
                        candidate_id=job.candidate_id,
                        source_id=job.source_id,
                        session_id=job.session_id,
                        signal_score=job.priority,
                        signal_rank=1,  # updated in ranking export
                        ai_record=judgment_dict,
                    )

                    # Export updated dual-ranking CSV
                    self._export_dual_ranking_csv()

                    # 5. Render Vertical Short if approved & rendering enabled
                    if self.config.processing.rendering_enabled and record.is_approved:
                        self._render_candidate_short(job.candidate_id, record)
                        stats["shorts_rendered"] += 1

                except Exception as exc:
                    LOGGER.exception("Failed processing AI judgment for job %s: %s", job.job_id, exc)
                    self.queue.mark_failed(job.job_id, str(exc), retryable=True)

        return stats

    def _render_candidate_short(self, candidate_id: str, record: AIJudgmentRecord) -> Path | None:
        """Render approved candidate as a 9:16 vertical Short using GPU/CPU slot."""
        clip_path = self.candidates_dir / f"{candidate_id}.mp4"
        if not clip_path.is_file():
            LOGGER.warning("Candidate MP4 not found for rendering: %s", clip_path)
            return None

        out_short = self.shorts_dir / f"{candidate_id}_short.mp4"
        if out_short.is_file():
            return out_short

        render_cfg = RenderConfig(layout=record.recommended_layout)
        try:
            info = probe_media(clip_path, self.ffprobe_binary)
            work_dir = self.shorts_dir / f".work_{candidate_id}"
            work_dir.mkdir(parents=True, exist_ok=True)

            with self.gpu_manager.acquire_render_slot(timeout_seconds=30.0, allow_cpu_fallback=True) as use_gpu:
                encoder = "h264_nvenc" if use_gpu else "libx264"
                selector = EncoderSelector(encoder, self.ffmpeg_binary)
                render_original(
                    source=clip_path,
                    output=out_short,
                    start=0.0,
                    duration=info.duration,
                    info=info,
                    words=[],
                    config=render_cfg,
                    work=work_dir,
                    selector=selector,
                )
            LOGGER.info("Rendered vertical Short for candidate %s -> %s", candidate_id, out_short)
            return out_short
        except Exception as exc:
            LOGGER.warning("Could not render Short for candidate %s: %s", candidate_id, exc)
            return None

    def _export_dual_ranking_csv(self) -> Path:
        """Export dual-ranking table (Signal Rank vs AI Editorial Rank) to CSV."""
        items = self.review_store.list_candidates(limit=100)
        csv_path = self.judgments_dir / "live_ai_judgments.csv"

        fieldnames = [
            "signal_rank",
            "candidate_id",
            "signal_score",
            "ai_rank",
            "ai_editorial_rating",
            "ai_confidence",
            "ai_category",
            "ai_approved",
            "human_approval",
            "ai_summary",
            "hook_suggestion",
            "recommended_layout",
            "clip_path",
        ]

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for ai_rank, item in enumerate(items, 1):
                writer.writerow({
                    "signal_rank": item.signal_rank,
                    "candidate_id": item.candidate_id,
                    "signal_score": round(item.signal_score, 3),
                    "ai_rank": ai_rank,
                    "ai_editorial_rating": item.ai_editorial_rating or "",
                    "ai_confidence": round(item.ai_confidence, 3) if item.ai_confidence else "",
                    "ai_category": item.ai_category or "",
                    "ai_approved": item.ai_approved or "",
                    "human_approval": item.human_approval,
                    "ai_summary": item.ai_summary or "",
                    "hook_suggestion": item.hook_suggestion or "",
                    "recommended_layout": item.recommended_layout or "",
                    "clip_path": item.clip_path or "",
                })
        return csv_path

    def _orchestration_loop(self) -> None:
        """Continuous live background polling loop."""
        LOGGER.info("Pipeline orchestration loop running...")
        while not self._stop_event.is_set():
            try:
                self.step()
            except Exception as exc:
                LOGGER.exception("Error in pipeline orchestration step: %s", exc)

            # Interruptible sleep
            self._stop_event.wait(timeout=2.0)
        LOGGER.info("Pipeline orchestration loop terminated")
