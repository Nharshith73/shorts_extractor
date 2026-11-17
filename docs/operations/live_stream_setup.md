# StreamClip AI — Live Stream Setup & Operations Guide

## 1. Prerequisites & Environment

1. **System Binaries**:
   - `ffmpeg` and `ffprobe` must be available on system `PATH`.
2. **Local AI Model Weights**:
   - `Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf` located at `D:/models/qwen2.5-vl-7b-gguf/`.
   - `mmproj-Qwen2.5-VL-7B-Instruct-Q8_0.gguf` located at `D:/models/qwen2.5-vl-7b-gguf/`.
   - `llama-server.exe` located at `D:/tools/llama.cpp/`.
3. **Configuration**:
   - `source.yaml`: Live source configuration and authorization gates.
   - `judge.yaml`: Multimodal model parameters and server ports.

---

## 2. Source Configuration (`source.yaml`)

```yaml
source:
  id: primary_creator
  creator_name: authorized_creator
  platform: youtube
  watch_url: https://www.youtube.com/watch?v=fvKv0L8aUJs

  authorization:
    permitted: true
    capture: true
    clipping: true
    editing: true
    publishing: true
    permission_reference: permission-record-reference

  acquisition:
    method: direct_media_input
    media_url_env: STREAMCLIP_LIVE_INPUT
    media_url: ""  # Optional fallback direct feed URL
    reconnect_enabled: true
    reconnect_max_attempts: 0

capture:
  segment_seconds: 6
  retention_seconds: 900
  segments_dir: data/live_segments

processing:
  detection_enabled: true
  transcription_enabled: true
  ai_judging_enabled: true
  rendering_enabled: true
  human_approval_required: true

queue:
  max_queue_depth: 50
  max_attempts: 3
  stale_job_timeout_seconds: 120.0

gpu:
  serialize_vlm_and_render: true
  max_vram_mb: 7800
```

> [!IMPORTANT]
> **YouTube Watch URLs vs Live Media Input**:
> A YouTube watch-page URL is a web metadata reference. It cannot be fed directly to FFmpeg without unauthorized circumvention. An authorized direct stream must be supplied via:
> - The `STREAMCLIP_LIVE_INPUT` environment variable (e.g. `set STREAMCLIP_LIVE_INPUT=rtmp://localhost/live/stream`), OR
> - Creator-provided RTMP/SRT/HLS ingest endpoint set in `source.yaml` (`media_url`).

---

## 3. Operator CLI Workflows

### 3.1 Production Preflight Audit
Before initiating live operations, run the preflight verification command to audit all 11 critical production criteria without initiating persistent media capture:

```powershell
# Standard terminal report with PASS/WARN/FAIL status table
python -m app.live preflight --config source.yaml

# Structured JSON report for automated orchestration pipelines
python -m app.live preflight --config source.yaml --json
```

**Audited Criteria:**
1. **Source Configuration Validity**: Parses YAML structure, numeric intervals, and retention windows.
2. **Authorization Record Completeness**: Verifies `permission_reference` and legal consent metadata.
3. **Permission Status**: Enforces all 5 mandatory gates (`permitted`, `capture`, `clipping`, `editing`, `publishing`).
4. **FFmpeg & ffprobe Availability**: Probes system binaries and verifies `h264_nvenc` hardware acceleration.
5. **SQLite Database Availability**: Validates segment manifest path and enforces WAL mode journal persistence.
6. **Capture Directory Permissions**: Probes read/write/delete access across live segments, candidates, context, judgments, and shorts directories.
7. **Required Environment Variables**: Inspects `STREAMCLIP_LIVE_INPUT` and `YOUTUBE_API_KEY`.
8. **Media Input Protocol**: Distinguishes between web metadata URLs (YouTube watch pages) and direct media feeds (RTMP/SRT/HLS/UDP).
9. **Qwen Server Availability**: Probes live server health at `http://127.0.0.1:8080/health` or verifies offline disk assets (`llama-server.exe`, Qwen GGUF model, mmproj) for automatic cold start.
10. **GPU Resource Availability**: Queries NVIDIA GPU VRAM headroom ($\ge 500$ MB free).
11. **Model & Render Configuration**: Validates `judge.yaml`, `render.yaml`, and subtitle font assets.

**Preflight Exit Codes:**
- `0`: `FIRST LIVE STREAM READY` — All infrastructure, models, directories, and direct media feeds are active.
- `2`: `FIRST LIVE STREAM VERIFIED WITH LIMITATIONS` — Infrastructure and models are 100% verified; awaiting direct creator stream feed (`STREAMCLIP_LIVE_INPUT`).
- `1`: `FIRST LIVE STREAM NOT READY` — Critical failure (missing binaries, unpermitted gates, or schema error).

### 3.2 Validate Source Authorization & Connectivity
```powershell
python -m app.live validate-source --config source.yaml
```

### 3.3 Start Live Capture & Pipeline
```powershell
# Production run with real local Qwen VLM
python -m app.live start --config source.yaml --foreground

# Development run with local test media and mock AI
python -m app.live start --config source.yaml --media-override "path/to/test.mp4" --mock-ai --foreground
```

### 3.3 Check Real-time Pipeline Status
```powershell
python -m app.live status
```

### 3.4 Inspect Extracted Candidates & Dual Rankings
```powershell
python -m app.live candidates
```

Output format:
```text
Sig#  AI#   Candidate ID               Score   Rating  Category       AI App  Human Status
-------------------------------------------------------------------------------------
#1    #1    e44c13e6573ac16eb456bec1   4.85    4/5     chaotic        y       PENDING
#2    #2    2aae2654e2ecfe1a8dc53ede   4.12    1/5     ordinary       n       PENDING
```

### 3.5 Human Editorial Approval
```powershell
# Approve candidate for distribution
python -m app.live approve e44c13e6573ac16eb456bec1 --notes "Strong pursuit sequence"

# Reject candidate
python -m app.live reject 2aae2654e2ecfe1a8dc53ede --reason "No dialogue or action"
```

### 3.6 Inspect AI Judgment Queue
```powershell
python -m app.live queue
```

### 3.7 Gracefully Stop Capture
```powershell
python -m app.live stop --config source.yaml
```
