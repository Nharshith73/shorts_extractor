# StreamClip AI — Live Stream Troubleshooting & Diagnostic Guide

## 1. Common Operational Issues & Remedies

### 1.1 "AuthorizationError: Live capture not permitted"
- **Cause**: One or more flags under `source.authorization` in `source.yaml` are set to `false`, or `permission_reference` is missing.
- **Resolution**: Verify signed creator agreements and update `source.yaml`:
  ```yaml
  authorization:
    permitted: true
    capture: true
    clipping: true
    editing: true
    publishing: true
    permission_reference: "creator-agreement-id"
  ```

### 1.2 "MissingMediaInputError: YouTube watch URL is a web metadata reference"
- **Cause**: StreamClip AI does not engage in undocumented scraping or access control circumvention against YouTube web pages.
- **Resolution**: Set the environment variable `STREAMCLIP_LIVE_INPUT` to an authorized RTMP, SRT, or HLS feed provided by the creator or local ingest proxy:
  ```powershell
  $env:STREAMCLIP_LIVE_INPUT = "rtmp://127.0.0.1:1935/live/stream"
  python -m app.live validate-source
  ```

### 1.3 Supervisor Remains in `RECONNECTING`
- **Diagnostic Command**:
  ```powershell
  python -m app.live status
  ```
- **Checklist**:
  1. Verify network connectivity to the streaming ingest port.
  2. Check if the creator stopped broadcasting (supervisor will transition to `ENDED` once EOF is acknowledged).
  3. Inspect segment directory permissions: ensure `data/live_segments/` is writable.

### 1.4 AI Judgment Queue Backpressure
- **Symptom**: `queue.get_queue_depth()` exceeds `max_queue_depth` (50).
- **Remedy**:
  - StreamClip AI prioritizes continuous stream capture above all else. Under queue backpressure, capture and candidate reservation continue uninterrupted.
  - To accelerate queue drain:
    1. Ensure GPU offloading is active (`n_gpu_layers: 99`).
    2. Check if another process is occupying VRAM.
    3. The pipeline will automatically serialize VLM inference and NVENC rendering.

### 1.5 Stale Jobs in `RUNNING` State
- **Cause**: Operator terminated the process ungracefully during an active AI evaluation.
- **Resolution**:
  - `LiveJudgmentQueue` automatically detects jobs in `RUNNING` status whose update timestamp exceeds `stale_job_timeout_seconds` (120s) and resets them to `RETRY_WAIT`.
  - Stale jobs can also be recovered manually during pipeline initialization.

---

## 2. Diagnostic Inspection Checklist

1. **Verify Database Integrity**:
   ```powershell
   sqlite3 data/live_segments/manifest.sqlite3 "PRAGMA integrity_check;"
   ```
2. **Inspect Process Ownership**:
   ```powershell
   Get-Process -Name "*ffmpeg*", "*python*"
   ```
3. **Inspect Real-time VRAM**:
   ```powershell
   nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv
   ```
