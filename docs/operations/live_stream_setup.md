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
  watch_url: https://www.youtube.com/watch?v=Opnkl7GVQEc

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

### 3.1 Validate Source Authorization & Connectivity
```powershell
python -m app.live validate-source --config source.yaml
```

### 3.2 Start Live Capture & Pipeline
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
