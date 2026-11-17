# StreamClip AI — Stage 4C: Single-Stream Live Architecture

## 1. System Philosophy & Objectives

StreamClip AI is a **LIVE-FIRST internal content production platform**. 

The system continuously ingests a single authorized creator livestream, detects engaging gameplay moments in near real-time, extracts candidates, assembles multimodal context packages, evaluates editorial value via the local Qwen2.5-VL model, renders vertical 9:16 Shorts, and surfaces them for operator review and approval.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        SINGLE-STREAM LIVE ARCHITECTURE PIPELINE                        │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│   Authorized Media Stream (RTMP / SRT / HLS / STREAMCLIP_LIVE_INPUT)                  │
│                                ↓                                                       │
│   [StreamReader (FFmpeg)] ──> Continuous 6-Second MPEG-TS Segments                     │
│                                ↓                                                       │
│   [RollingBuffer + SQLite Manifest] ──> Segment Tracking & Pin Protections             │
│                                ↓                                                       │
│   [DetectionRunner] ───────> Audio/Video Feature Extraction & Rolling MAD Scoring      │
│                                ↓                                                       │
│   [CandidateExtractor] ────> Event Windowing & Post-Roll Segment Reservation           │
│                                ↓                                                       │
│   [Candidate MP4 Cut] ─────> Candidate Context Assembly (8 Visual Frames, Transcript)  │
│                                ↓                                                       │
│   [LiveJudgmentQueue] ─────> Persistent SQLite AI Queue (Priority Dispatch)            │
│                                ↓                                                       │
│   [GpuResourceManager] ────> Conservative GPU Lock (Serializes VLM & NVENC)            │
│                                ↓                                                       │
│   [Qwen2.5-VL Provider] ───> Grounded AI Judgment JSON (1-5 Rating, Hook, Payoff)     │
│                                ↓                                                       │
│   [Review & Approval] ─────> Dual-Rankings (Signal vs AI Rank) & Vertical 9:16 Short  │
│                                                                                        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Core Subsystems & Operational Contracts

### 2.1 Single Stream Supervisor (`app/live/supervisor.py`)
Maintains a persistent, SQLite-backed lifecycle state machine:
- `STOPPED`: Idle state before capture.
- `DISCOVERING`: Validating legal authorization gates and configuration.
- `WAITING_FOR_LIVE`: Awaiting media stream availability.
- `CONNECTING`: Spawning StreamReader and establishing connection.
- `LIVE`: Actively producing completed playable segments.
- `RECONNECTING`: Socket stalled or process terminated; reconnecting with exponential backoff.
- `ENDED`: Broadcast terminated cleanly (EOF).
- `FAILED`: Unrecoverable network or binary error.
- `AUTHORIZATION_REVOKED`: Authorization permission checks failed.

### 2.2 Live Capture & Segment Reservation (`app/ingestion/`)
- Ingestion operates in 6-second segments with keyframe alignment.
- When an event occurs, candidate extraction reserves all required segments in `segment_reservations` and `candidate_segment_pins` within `manifest.sqlite3`.
- The rolling buffer retention cleaner runs independently and will never delete pinned segments required by pending candidate extraction.

### 2.3 Persistent AI Judgment Queue (`app/live/queue.py`)
- Backed by SQLite table `live_ai_jobs` in `manifest.sqlite3`.
- States: `PENDING`, `RUNNING`, `COMPLETED`, `RETRY_WAIT`, `FAILED`, `CANCELLED`.
- Atomic claiming prevents race conditions across workers.
- Bounded retries (default 3 attempts) with automatic stale-job recovery if a worker crashes while in `RUNNING`.
- High-priority events (higher detector signal scores) are dispatched first.

### 2.4 Conservative GPU Resource Manager (`app/live/gpu.py`)
- Hardware Profile: NVIDIA GeForce RTX 4070 Laptop GPU (8,188 MiB VRAM).
- Model Footprint: Qwen2.5-VL-7B (Q4_K_M) consumes ~7.42 GB VRAM.
- Hardware Rendering: NVENC 1080p transcode consumes ~350 MB VRAM.
- Policy:
  1. Capture and CPU detection run continuously and independently.
  2. VLM inference and GPU NVENC rendering are strictly serialized via a global mutex.
  3. If VRAM free is below 500 MB, the renderer automatically switches to CPU encoding (`libx264`) without stalling capture.

### 2.5 Dual-Ranked Review Store (`app/live/review.py`)
- Preserves both **Signal Rank** (canonical detector z-score) and **AI Editorial Rank** (Qwen multimodal rating & confidence).
- Persists human approvals (`approved`, `rejected`, `pending`) to maintain editorial oversight before any final video distribution.

---

## 3. Tested vs Planned Capabilities

| Capability | Status | Implementation Details |
| :--- | :---: | :--- |
| **Source Authorization Validation** | **TESTED** | Full validation of 5 permission gates in `source.yaml`. |
| **Direct Media Feed Acquisition** | **TESTED** | Ingestion via `STREAMCLIP_LIVE_INPUT` or direct RTMP/SRT/HLS. |
| **YouTube Metadata Inspection** | **TESTED** | Official YouTube Data API inspection; reports unauthenticated/blocked state cleanly. |
| **Continuous Live Capture** | **TESTED** | Continuous 6-second MPEG-TS segments with SQLite tracking. |
| **Detection-to-Candidate Handoff**| **TESTED** | Follows complete segments, creates windows, protects segments. |
| **Context Extraction & Frame Sampling** | **TESTED** | Extracts 8 visual frames and audio features per candidate. |
| **Persistent SQLite AI Queue** | **TESTED** | Atomic claims, bounded retries, stale recovery, backpressure. |
| **Conservative GPU Scheduling** | **TESTED** | Mutual exclusion between VLM and NVENC; CPU render fallback. |
| **Dual Rankings & Human Review** | **TESTED** | Side-by-side signal vs AI ranking, approve/reject CLI. |
| **Multi-Stream Orchestration** | *PLANNED* | Deferred to post-Phase 4. |
| **Automatic Social Publishing** | *PLANNED* | Deferred; requires operator sign-off. |
