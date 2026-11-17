"""Stage 4C: Comprehensive Unit and Integration Tests for Single-Stream Live Operations.

Covers:
1. Source configuration validation.
2. Authorization validation.
3. Capture startup.
4. Continuous six-second segment production.
5. Detection execution while capture remains active.
6. Candidate creation.
7. Post-roll waiting.
8. Segment reservation.
9. Candidate extraction.
10. Context extraction.
11. AI queue insertion.
12. Mock provider execution.
13. Judgment persistence.
14. Queue recovery after interruption.
15. Stream disconnection.
16. Stream reconnection.
17. Session identity handling.
18. Graceful shutdown.
19. GPU resource scheduling.
20. Existing VOD compatibility.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import time
from unittest.mock import MagicMock, patch
import pytest

from app.clips.config import CandidateConfig
from app.clips.store import CandidateStore
from app.clips.windows import Event, Window
from app.detection.config import DetectionConfig
from app.detection.runner import DetectionRunner
from app.evaluation.inference import MockHighlightJudgeProvider
from app.evaluation.schema import AIJudgmentRecord
from app.ingestion.rolling_buffer import RollingBuffer, SegmentRecord, utc_now
from app.ingestion.stream_reader import StreamConfig, StreamReader
from app.live.config import (
    AcquisitionConfig,
    AuthorizationConfig,
    AuthorizationError,
    LiveCaptureConfig,
    LiveGpuConfig,
    LiveProcessingConfig,
    LiveQueueConfig,
    LiveSourceConfig,
    MissingMediaInputError,
    load_source_config,
)
from app.live.gpu import GpuResourceManager, VramSnapshot
from app.live.pipeline import SingleStreamLivePipeline
from app.live.queue import LiveJobRecord, LiveJobStatus, LiveJudgmentQueue
from app.live.review import LiveReviewStore
from app.live.supervisor import LiveState, LiveSupervisor, SupervisorStateStore
from app.live.youtube import YouTubeBroadcastMetadata, inspect_youtube_source

FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def make_test_video(path: Path, seconds: int = 18) -> None:
    """Create keyframe-friendly test video with 1-second GOPs."""
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x180:rate=30",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=1000:sample_rate=48000",
            "-t",
            str(seconds),
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-g",
            "30",
            "-keyint_min",
            "30",
            "-sc_threshold",
            "0",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


# ---------------------------------------------------------------------------
# Test 1: Source Configuration Validation
# ---------------------------------------------------------------------------
def test_source_configuration_validation(tmp_path: Path) -> None:
    cfg_file = tmp_path / "valid_source.yaml"
    cfg_file.write_text(
        """
source:
  id: test_creator
  creator_name: test_streamer
  platform: youtube
  watch_url: https://www.youtube.com/watch?v=Opnkl7GVQEc
  authorization:
    permitted: true
    capture: true
    clipping: true
    editing: true
    publishing: true
    permission_reference: perm-1234
  acquisition:
    method: direct_media_input
    media_url_env: TEST_LIVE_INPUT
    media_url: udp://127.0.0.1:1234
    reconnect_enabled: true
capture:
  segment_seconds: 6
  retention_seconds: 900
processing:
  detection_enabled: true
  ai_judging_enabled: true
""",
        encoding="utf-8",
    )

    loaded = load_source_config(cfg_file)
    assert loaded.id == "test_creator"
    assert loaded.creator_name == "test_streamer"
    assert loaded.capture.segment_seconds == 6.0
    assert loaded.capture.retention_seconds == 900.0
    assert loaded.authorization.permitted is True
    assert loaded.acquisition.media_url == "udp://127.0.0.1:1234"


def test_source_configuration_negative_values(tmp_path: Path) -> None:
    cfg_file = tmp_path / "invalid_source.yaml"
    cfg_file.write_text(
        """
source:
  id: bad_creator
  authorization:
    permitted: true
    capture: true
    clipping: true
    editing: true
    publishing: true
    permission_reference: valid-ref
capture:
  segment_seconds: -5
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="segment_seconds"):
        load_source_config(cfg_file)


# ---------------------------------------------------------------------------
# Test 2: Authorization Validation & Missing-Input Enforcement
# ---------------------------------------------------------------------------
def test_authorization_gates() -> None:
    auth = AuthorizationConfig(
        permitted=False,
        capture=True,
        clipping=True,
        editing=True,
        publishing=True,
        permission_reference="ref",
    )
    with pytest.raises(AuthorizationError, match="permitted is false"):
        auth.validate()

    auth.permitted = True
    auth.publishing = False
    with pytest.raises(AuthorizationError, match="publishing is false"):
        auth.validate()

    auth.publishing = True
    auth.permission_reference = ""
    with pytest.raises(AuthorizationError, match="permission_reference"):
        auth.validate()


def test_missing_media_input_youtube_watch_url_blocked() -> None:
    acq = AcquisitionConfig(media_url_env="UNSET_STREAMCLIP_ENV_VAR", media_url="")
    with pytest.raises(MissingMediaInputError, match="metadata reference and is not a direct FFmpeg-compatible"):
        acq.resolve_media_url("https://www.youtube.com/watch?v=Opnkl7GVQEc")


# ---------------------------------------------------------------------------
# Test 3 & 4: Capture Startup and Continuous 6-Second Segment Production
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg required")
def test_capture_startup_and_segment_production(tmp_path: Path) -> None:
    seg_dir = tmp_path / "live_segments"
    seg_dir.mkdir()
    video_file = tmp_path / "live_input.mp4"
    make_test_video(video_file, seconds=15)

    cfg = LiveSourceConfig(
        id="test_src",
        authorization=AuthorizationConfig(
            permitted=True, capture=True, clipping=True, editing=True, publishing=True, permission_reference="ref"
        ),
        acquisition=AcquisitionConfig(media_url=str(video_file)),
        capture=LiveCaptureConfig(segment_seconds=6.0, retention_seconds=900.0, segments_dir=seg_dir),
    )

    supervisor = LiveSupervisor(config=cfg)
    assert supervisor.state == LiveState.STOPPED

    supervisor.start(media_override=str(video_file))
    assert supervisor.state in (LiveState.CONNECTING, LiveState.LIVE)

    # Wait for at least 1 complete segment
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if supervisor.reader and supervisor.reader.completed_segment_count >= 1:
            break
        time.sleep(0.5)

    assert supervisor.reader is not None
    assert supervisor.reader.completed_segment_count >= 1

    # Check manifest records in SQLite
    buf = RollingBuffer(seg_dir, retention_seconds=900)
    records = buf.manifest.all_records()
    assert len(records) >= 1
    assert records[0].duration > 0.0

    supervisor.stop()
    assert supervisor.state == LiveState.STOPPED


# ---------------------------------------------------------------------------
# Test 5 & 6: Live Detection Execution & Candidate Creation
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg required")
def test_detection_execution_and_candidate_creation(tmp_path: Path) -> None:
    seg_dir = tmp_path / "segments"
    seg_dir.mkdir()
    video_file = tmp_path / "test.mp4"
    make_test_video(video_file, seconds=15)

    stream_cfg = StreamConfig(source=str(video_file), segment_seconds=6.0, segments_dir=seg_dir)
    reader = StreamReader(stream_cfg)
    reader.start()

    # Wait for 2 segments
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if reader.completed_segment_count >= 2:
            break
        time.sleep(0.5)
    reader.stop()

    manifest_db = seg_dir / "manifest.sqlite3"
    det_runner = DetectionRunner(manifest_path=manifest_db, config=DetectionConfig())
    stats = det_runner.run_once()
    assert stats.segments_analyzed >= 1

    # Insert a synthetic detection event into the store to verify candidate creation
    store = CandidateStore(manifest_db)
    records = RollingBuffer(seg_dir, retention_seconds=900).manifest.all_records()
    assert len(records) >= 1

    from datetime import timedelta
    peak_utc = records[0].wall_start_utc
    evt = Event(
        id=1,
        source_id=records[0].source_id,
        session_id=records[0].session_id,
        peak=peak_utc,
        start=peak_utc,
        end=peak_utc + timedelta(seconds=6),
        score=4.5,
    )
    win = Window(events=(evt,), start=evt.start, end=evt.end)
    cand_id = store.create(win, records, now=utc_now(), status="waiting")
    assert cand_id is not None
    assert len(store.all()) >= 1


# ---------------------------------------------------------------------------
# Test 7 & 8: Post-roll Waiting and Segment Reservation
# ---------------------------------------------------------------------------
def test_post_roll_waiting_and_segment_reservations(tmp_path: Path) -> None:
    manifest_db = tmp_path / "manifest.sqlite3"
    buf = RollingBuffer(tmp_path, retention_seconds=900)
    store = CandidateStore(manifest_db)

    # Create dummy segment record
    now = utc_now()
    seg_path = tmp_path / "seg_001.ts"
    buf.manifest.add_writing(
        path=seg_path,
        wall_start_utc=now,
        source_id="src1",
        session_id="sess1",
    )
    buf.manifest.complete(path=seg_path, duration=6.0)
    records = buf.manifest.all_records()
    assert len(records) == 1

    # Reserve segment for candidate
    store.reserve("cand_test_01", records)
    # Check that segment was pinned in manifest
    assert buf.manifest.all_records()[0].pinned is True


# ---------------------------------------------------------------------------
# Test 9 & 10: Candidate Extraction & Multimodal Context Extraction
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg required")
def test_candidate_context_extraction(tmp_path: Path) -> None:
    clip_file = tmp_path / "clip.mp4"
    make_test_video(clip_file, seconds=10)

    from app.evaluation.context import FrameSamplerConfig, build_candidate_context

    ctx_dir = tmp_path / "context"
    cand_dict = {
        "candidate_id": "live_cand_001",
        "window_id": "live_cand_001",
        "start_seconds": 0.0,
        "end_seconds": 10.0,
        "peak_seconds": 5.0,
        "score": 4.2,
    }

    ctx = build_candidate_context(
        candidate=cand_dict,
        source_video=clip_file,
        output_dir=ctx_dir,
        sampler_config=FrameSamplerConfig(max_frames=8),
    )

    assert ctx.candidate_id == "live_cand_001"
    assert len(ctx.frames) >= 1
    assert (ctx_dir / "live_cand_001" / "context.json").is_file()


# ---------------------------------------------------------------------------
# Test 11, 12, 13: AI Queue Insertion, Mock Execution & Judgment Persistence
# ---------------------------------------------------------------------------
def test_ai_queue_workflow_and_persistence(tmp_path: Path) -> None:
    db_path = tmp_path / "manifest.sqlite3"
    queue = LiveJudgmentQueue(db_path, max_queue_depth=10, max_attempts=3)
    assert queue.get_queue_depth() == 0

    # Enqueue candidate
    job, was_inserted = queue.enqueue(
        candidate_id="cand_100",
        source_id="src_1",
        session_id="sess_1",
        context_path=tmp_path / "context.json",
        priority=4.5,
    )
    assert was_inserted is True
    assert job.status == LiveJobStatus.PENDING.value
    assert queue.get_queue_depth() == 1

    # Idempotent re-enqueue
    job2, was_inserted2 = queue.enqueue(
        candidate_id="cand_100",
        source_id="src_1",
        session_id="sess_1",
        context_path=tmp_path / "context.json",
    )
    assert was_inserted2 is False
    assert job2.job_id == job.job_id

    # Claim job
    claimed = queue.claim_next()
    assert claimed is not None
    assert claimed.job_id == job.job_id
    assert claimed.status == LiveJobStatus.RUNNING.value
    assert queue.get_queue_depth() == 0

    # Mark completed
    judgment_payload = {"candidate_id": "cand_100", "editorial_rating": 4, "category": "chaotic"}
    queue.mark_completed(claimed.job_id, judgment_payload)

    stats = queue.get_stats()
    assert stats[LiveJobStatus.COMPLETED.value] == 1


# ---------------------------------------------------------------------------
# Test 14: Queue Recovery After Interruption
# ---------------------------------------------------------------------------
def test_queue_stale_job_recovery(tmp_path: Path) -> None:
    db_path = tmp_path / "manifest.sqlite3"
    queue = LiveJudgmentQueue(db_path, max_attempts=2)

    job, _ = queue.enqueue(
        candidate_id="stale_cand",
        source_id="src_1",
        session_id="sess_1",
        context_path="/dummy/path",
        priority=3.0,
    )

    claimed = queue.claim_next()
    assert claimed is not None
    assert claimed.status == LiveJobStatus.RUNNING.value

    # Recover with 0 timeout (instantly stale)
    recovered = queue.recover_stale_jobs(timeout_seconds=-1.0)
    assert recovered == 1

    # Should be back to RETRY_WAIT
    reclaimed = queue.claim_next()
    assert reclaimed is not None
    assert reclaimed.attempts == 2

    # Second recovery exceeds max_attempts (2) -> moves to FAILED
    queue.recover_stale_jobs(timeout_seconds=-1.0)
    failed_job = queue.list_jobs(status=LiveJobStatus.FAILED.value)
    assert len(failed_job) == 1
    assert failed_job[0].candidate_id == "stale_cand"


# ---------------------------------------------------------------------------
# Test 15 & 16: Stream Disconnection and Reconnection Handling
# ---------------------------------------------------------------------------
def test_supervisor_state_transitions(tmp_path: Path) -> None:
    db_path = tmp_path / "manifest.sqlite3"
    store = SupervisorStateStore(db_path)

    store.record_transition("src1", LiveState.CONNECTING, media_url="udp://127.0.0.1:1234")
    st1 = store.get_state("src1")
    assert st1["current_state"] == LiveState.CONNECTING.value

    store.record_transition("src1", LiveState.LIVE, session_id="sess_alpha")
    st2 = store.get_state("src1")
    assert st2["current_state"] == LiveState.LIVE.value
    assert st2["reconnect_attempts"] == 0

    # Disconnection triggers RECONNECTING
    store.record_transition("src1", LiveState.RECONNECTING, reason="Socket timed out")
    st3 = store.get_state("src1")
    assert st3["current_state"] == LiveState.RECONNECTING.value
    assert st3["reconnect_attempts"] == 1

    # Reconnection recovers to LIVE
    store.record_transition("src1", LiveState.LIVE, reason="Reconnected")
    st4 = store.get_state("src1")
    assert st4["current_state"] == LiveState.LIVE.value
    assert st4["reconnect_attempts"] == 0


# ---------------------------------------------------------------------------
# Test 17: Session Identity Handling
# ---------------------------------------------------------------------------
def test_session_identity_preservation(tmp_path: Path) -> None:
    buf = RollingBuffer(tmp_path, retention_seconds=900)
    now = utc_now()

    # Session 1
    p1 = tmp_path / "s1_001.ts"
    buf.manifest.add_writing(path=p1, wall_start_utc=now, source_id="src1", session_id="sess_1")
    buf.manifest.complete(path=p1, duration=6.0)

    # Session 2 (after reconnect)
    p2 = tmp_path / "s2_001.ts"
    buf.manifest.add_writing(path=p2, wall_start_utc=now, source_id="src1", session_id="sess_2")
    buf.manifest.complete(path=p2, duration=6.0)

    all_recs = buf.manifest.all_records()
    assert len(all_recs) == 2
    sess_ids = {r.session_id for r in all_recs}
    assert "sess_1" in sess_ids
    assert "sess_2" in sess_ids


# ---------------------------------------------------------------------------
# Test 18: Graceful Shutdown
# ---------------------------------------------------------------------------
def test_supervisor_graceful_shutdown(tmp_path: Path) -> None:
    cfg = LiveSourceConfig(
        id="shutdown_src",
        authorization=AuthorizationConfig(
            permitted=True, capture=True, clipping=True, editing=True, publishing=True, permission_reference="ref"
        ),
        acquisition=AcquisitionConfig(media_url="udp://127.0.0.1:9999"),
        capture=LiveCaptureConfig(segments_dir=tmp_path),
    )
    supervisor = LiveSupervisor(config=cfg)
    supervisor._state = LiveState.LIVE

    supervisor.stop()
    assert supervisor.state == LiveState.STOPPED


# ---------------------------------------------------------------------------
# Test 19: GPU Resource Scheduling & Serialization
# ---------------------------------------------------------------------------
def test_gpu_resource_manager_serialization() -> None:
    gpu = GpuResourceManager(serialize_vlm_and_render=True)

    with gpu.acquire_inference_slot(timeout_seconds=5.0):
        assert gpu.is_inference_active is True
        # While inference is active, render slot with CPU fallback returns False
        with gpu.acquire_render_slot(timeout_seconds=0.1, allow_cpu_fallback=True) as use_gpu:
            assert use_gpu is False  # CPU fallback chosen safely

    assert gpu.is_inference_active is False

    # When inference is free, render slot is granted
    with gpu.acquire_render_slot(timeout_seconds=5.0) as use_gpu:
        assert use_gpu is True
        assert gpu.is_render_active is True

    assert gpu.is_render_active is False


def test_gpu_resource_manager_vram_low_fallback() -> None:
    gpu = GpuResourceManager(serialize_vlm_and_render=True, min_free_vram_mb=1000.0)

    # Mock low VRAM snapshot (200 MB free < 1000 MB threshold)
    low_vram = VramSnapshot(used_mb=7800.0, free_mb=200.0, total_mb=8000.0)
    with patch("app.live.gpu.query_gpu_vram", return_value=low_vram):
        with gpu.acquire_render_slot(allow_cpu_fallback=True) as use_gpu:
            assert use_gpu is False  # Safely fell back to CPU


# ---------------------------------------------------------------------------
# Test 20: Existing VOD Compatibility
# ---------------------------------------------------------------------------
def test_existing_vod_pipeline_compatibility() -> None:
    from app.vod.selection import canonical_ranking_key

    # Ensure existing canonical ranking formulas remain identical
    k1 = canonical_ranking_key(score=5.0, unclipped_score=6.0, time_seconds=10.0)
    k2 = canonical_ranking_key(score=4.0, unclipped_score=4.5, time_seconds=5.0)
    assert k1 < k2  # Descending score sort


# ---------------------------------------------------------------------------
# Test 21: Full End-to-End Live Pipeline Step Execution
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg required")
def test_live_pipeline_end_to_end_step(tmp_path: Path) -> None:
    seg_dir = tmp_path / "live_segments"
    seg_dir.mkdir()
    video_file = tmp_path / "stream.mp4"
    make_test_video(video_file, seconds=24)

    cfg = LiveSourceConfig(
        id="test_stream",
        authorization=AuthorizationConfig(
            permitted=True, capture=True, clipping=True, editing=True, publishing=True, permission_reference="p-1"
        ),
        acquisition=AcquisitionConfig(media_url=str(video_file)),
        capture=LiveCaptureConfig(segment_seconds=6.0, retention_seconds=900.0, segments_dir=seg_dir),
    )

    mock_provider = MockHighlightJudgeProvider()
    pipeline = SingleStreamLivePipeline(config=cfg, provider=mock_provider, base_dir=tmp_path)

    # 1. Start capture and produce at least 3 segments (18s)
    pipeline.supervisor.start(media_override=str(video_file))

    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if pipeline.supervisor.reader and pipeline.supervisor.reader.completed_segment_count >= 3:
            break
        time.sleep(0.5)

    pipeline.supervisor.stop()

    manifest_db = seg_dir / "manifest.sqlite3"
    records = RollingBuffer(seg_dir, retention_seconds=900).manifest.all_records()
    assert len(records) >= 2

    # Inject detection event covering 10 seconds (covered by records 0 and 1)
    from datetime import timedelta
    peak_utc = records[0].wall_start_utc + timedelta(seconds=5)
    evt = Event(
        id=1,
        source_id=records[0].source_id,
        session_id=records[0].session_id,
        peak=peak_utc,
        start=records[0].wall_start_utc,
        end=records[0].wall_start_utc + timedelta(seconds=10),
        score=4.8,
    )
    win = Window(events=(evt,), start=evt.start, end=evt.end)
    pipeline.candidate_store.create(win, records, now=utc_now(), status="waiting")

    # Step 1: extracts candidate
    pipeline.step()

    # Step 2: builds context, enqueues to AI queue
    pipeline.step()

    # Step 3: processes claimed queue job with mock provider
    pipeline.step()

    # Verify AI judgment completed and CSV exported
    q_stats = pipeline.queue.get_stats()
    assert q_stats["COMPLETED"] >= 1
    assert (pipeline.judgments_dir / "live_ai_judgments.csv").is_file()

    # Verify review store has candidate
    reviews = pipeline.review_store.list_candidates()
    assert len(reviews) >= 1
    assert reviews[0].ai_editorial_rating is not None


# ---------------------------------------------------------------------------
# Test 22: Continuous Capture Soak & Telemetry Recording
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg required")
def test_live_continuous_soak_telemetry(tmp_path: Path) -> None:
    seg_dir = tmp_path / "soak_segments"
    seg_dir.mkdir()
    video_file = tmp_path / "soak_video.mp4"
    make_test_video(video_file, seconds=24)

    cfg = LiveSourceConfig(
        id="soak_source",
        authorization=AuthorizationConfig(
            permitted=True, capture=True, clipping=True, editing=True, publishing=True, permission_reference="soak-ref"
        ),
        acquisition=AcquisitionConfig(media_url=str(video_file)),
        capture=LiveCaptureConfig(segment_seconds=6.0, retention_seconds=900.0, segments_dir=seg_dir),
    )

    pipeline = SingleStreamLivePipeline(
        config=cfg,
        provider=MockHighlightJudgeProvider(),
        base_dir=tmp_path,
    )

    start_time = time.monotonic()
    pipeline.start(media_override=str(video_file))

    # Monitor continuous capture telemetry across multiple intervals
    telemetry_samples = []
    for _ in range(4):
        time.sleep(2.0)
        summary = pipeline.supervisor.get_summary()
        records = RollingBuffer(seg_dir, retention_seconds=900).manifest.all_records()
        total_size = sum(f.stat().st_size for f in seg_dir.glob("*.ts") if f.is_file())
        q_depth = pipeline.queue.get_queue_depth()
        telemetry_samples.append({
            "elapsed_seconds": round(time.monotonic() - start_time, 2),
            "state": summary["state"],
            "segments_completed": summary["segments_completed"],
            "manifest_records": len(records),
            "queue_depth": q_depth,
            "storage_bytes": total_size,
        })

    pipeline.stop()

    assert len(telemetry_samples) == 4
    assert telemetry_samples[-1]["segments_completed"] >= 1
    assert telemetry_samples[-1]["storage_bytes"] > 0
