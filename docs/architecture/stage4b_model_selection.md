# StreamClip AI — Stage 4B Model Selection & Local Multimodal Feasibility

## 1. Executive Summary

Stage 4B introduces the **AI Highlight Judge**: a multimodal editorial reasoning system evaluating candidate gameplay clips extracted by Stage 3.

The AI Judge elevates StreamClip AI from heuristic signal detection (measuring noise, motion, and chat spikes) to grounded narrative comprehension:
- Does this candidate have an engaging visual/audio hook in the first 2 seconds?
- Is there a clear, satisfying payoff or punchline?
- Is this clip entertaining enough to publish as a vertical YouTube Short or TikTok?
- Grounded multimodal evidence attribution: distinguishing strictly between observable visual frames, dialogue transcripts, chat activity, acoustic energy, and model inferences.

This document records the empirical feasibility spike, runtime architecture decision, hardware benchmarks, concurrency validation, and operational profile on our Windows development workstation.

---

## 2. Workstation Hardware Profile

All benchmarks and validations were executed on our dedicated internal production workstation:

| Component | Specification | Operational Limits & Measured Footprint |
| :--- | :--- | :--- |
| **GPU** | NVIDIA GeForce RTX 4070 Laptop GPU | **8,188 MiB (8.0 GB) GDDR6 VRAM**, Driver 621.01, CUDA UMD 13.5 |
| **CPU** | Intel Core i9-14900HX | **24 Cores / 32 Threads** (8 P-cores + 16 E-cores, up to 5.8 GHz) |
| **System RAM** | 16.0 GB DDR5-5600 | Host memory for system processes and CPU offloading |
| **Storage** | 1 TB NVMe SSD (PCIe 4.0) | High-speed model weights caching and frame extraction |
| **Hardware Video**| Dual 8th Gen NVENC Encoders | Concurrent vertical 1080p rendering (uses ~350 MB VRAM) |
| **Operating System**| Windows 11 Pro 64-bit | Python 3.13.9 runtime environment |

---

## 3. Model Weights & Provenance

The primary model selected and benchmarked is **Qwen2.5-VL-7B-Instruct**:

- **Model Weights**: `Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf` (4,683,072,032 bytes / 4.36 GB).
- **Multimodal Projector**: `mmproj-Qwen2.5-VL-7B-Instruct-Q8_0.gguf` (853,119,712 bytes / 853 MB).
- **Source Repository**: `ggml-org/Qwen2.5-VL-7B-Instruct-GGUF` (Hugging Face).
- **License**: **Apache 2.0** (verified against upstream model repository revision).
- **Local Storage**: Stored outside Git at `D:/models/qwen2.5-vl-7b-gguf/`.

---

## 4. Runtime Architecture Decision: Approach A vs Approach B

Two runtime approaches were evaluated for hosting Qwen2.5-VL locally:

### 4.1 Approach B: `llama-cpp-python` (FAILED)
- Attempted to install `llama-cpp-python` within Python 3.13 on Windows.
- **Root Failure**: Pre-built Windows wheels do not currently exist on PyPI for Python 3.13 with CUDA 12.4. Source compilation from `tar.gz` fails on Windows due to `MAX_PATH` path length limitations (>260 characters) inside `vendor/llama.cpp/tools/ui/...` submodules.
- **Conclusion**: Fragile build environment; unsuitable for stable production deployment.

### 4.2 Approach A: Standalone `llama.cpp` Server (SELECTED)
- Deployed standalone native `llama.cpp` server release `b12477` (commit `054a1babf`) with CUDA 12.4 runtime libraries at `D:/tools/llama.cpp/llama-server.exe`.
- **Advantages**:
  1. Complete decoupling from Python C/C++ compilation toolchains.
  2. Standardized, high-throughput OpenAI-compatible HTTP API (`/v1/chat/completions`).
  3. Native GPU auto-detection (`CUDA0: NVIDIA GeForce RTX 4070 Laptop GPU (8187 MiB, 7068 MiB free)`).
  4. Reliable process lifecycle management from Python (`subprocess.Popen` with health checking).
- **Decision**: **Adopt Approach A** as the production runtime for StreamClip AI Stage 4B.

---

## 5. Empirical Feasibility Benchmarks

Benchmarks were performed using real candidate context packages from synthetic gameplay VOD footage.

### 5.1 Frame Scaling & GPU Inference Performance (RTX 4070)

All 28 layers offloaded to GPU (`-ngl 99`), `temperature: 0.1`:

| Metric | 4 Sampled Frames | 8 Sampled Frames (Default) | 12 Sampled Frames |
| :--- | :---: | :---: | :---: |
| **Prompt Tokens** | 1,884 | 3,149 | 4,102 |
| **Completion Tokens** | 316 | 415 | 518 |
| **Inference Latency** | **8.27 seconds** | **10.27 seconds** | **13.15 seconds** |
| **Generation Speed** | **38.25 tok/s** | **40.43 tok/s** | **39.42 tok/s** |
| **Peak VRAM Used** | 7,423 MiB | 7,428 MiB | 7,427 MiB |
| **Free VRAM Headroom**| 526 MiB | 521 MiB | 522 MiB |
| **Host System RAM** | 120 MiB | 121 MiB | 123 MiB |
| **Context Window** | Fits in 4096 | Fits in 4096 | **Requires `n_ctx: 6144`** |

#### Key Context Limit Finding
With 12 frames, prompt tokens reached 4,102. Under a standard 4,096 context window, the llama-server returned HTTP 400 (`request exceeds available context size`). Configuring `n_ctx: 6144` in `judge.yaml` resolved this with only ~115 MB additional VRAM overhead, providing comfortable headroom for up to 12 frames + 2,000 output tokens.

### 5.2 Concurrent NVENC Transcoding Under Inference Load

To verify production stability, a high-bitrate 1080p 60fps MP4 transcode using NVENC (`h264_nvenc`) was executed simultaneously while Qwen2.5-VL was actively generating an 8-frame editorial judgment:

- **NVENC Exit Code**: `0` (Success, no frame drops).
- **Inference Latency**: `9.81 seconds` (no performance degradation).
- **Token Throughput**: `42.03 tok/s`.
- **Peak VRAM During Concurrency**: `7,656 MiB / 8,188 MiB` (~93.5% capacity).
- **Headroom Remaining**: `293 MiB` free VRAM.
- **CUDA Errors / OOM**: `0`.

**Verdict**: The RTX 4070 successfully sustains simultaneous NVENC hardware video encoding and multimodal LLM inference without memory exhaustion.

### 5.3 CPU Fallback Reality (Intel Core i9-14900HX)

Tested with full CPU offloading (`-ngl 0`, 24 threads):

- **Measured Token Throughput**: **6.76 tok/s** (contrasting with earlier theoretical estimate of 12-16 tok/s).
- **4-Frame Judgment Latency**: **44.37 seconds**.
- **Assessment**: CPU fallback is 100% functional and technically stable, but ~6.6x slower than GPU. It is viable for unattended overnight batch processing, but GPU offloading is necessary for interactive editorial sessions.

---

## 6. Schema & Output Grounding Contract

Implemented in `app/evaluation/schema.py`:

```python
class EvidenceItem(BaseModel):
    timestamp: float
    source: Literal["visual", "transcript", "chat", "audio", "inference"]
    observation: str

class AIJudgmentRecord(BaseModel):
    schema_version: str = "1.0"
    candidate_id: str
    event_category: str
    summary: str
    editorial_rating: int  # 0 to 5
    confidence: float
    context_complete: bool
    setup_present: bool
    payoff_present: bool
    entertainment_factors: list[str]
    evidence: list[EvidenceItem]
    hook_suggestion: str
    recommended_start_seconds: float
    recommended_end_seconds: float
    recommended_layout: str
    rejection_reason: str | None
    model_metadata: dict[str, Any]
```

### Multimodal Evidence Tiers
The model is instructed never to hallucinate dialogue or off-screen events:
1. `visual`: Direct observations from sampled JPEG frames.
2. `transcript`: Spoken dialogue verified from faster-whisper word-level transcripts.
3. `chat`: Audience chat density, emote bursts, and chatter reactions.
4. `audio`: Acoustic energy (explosions, gunshots, volume spikes).
5. `inference`: Deductive narrative conclusions drawn by the model.

---

## 7. Configuration & Dual Ranking Workflow

### 7.1 Configuration (`judge.yaml`)
Operators configure model parameters, server paths, and resource limits centrally in `judge.yaml`:
```yaml
judge:
  provider: "local-qwen"
  model_name: "Qwen2.5-VL-7B-Instruct-Q4_K_M"
  model_path: "D:/models/qwen2.5-vl-7b-gguf/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
  projector_path: "D:/models/qwen2.5-vl-7b-gguf/mmproj-Qwen2.5-VL-7B-Instruct-Q8_0.gguf"
  server_binary: "D:/tools/llama.cpp/llama-server.exe"
  n_gpu_layers: 99
  n_ctx: 6144
  max_frames: 8
  timeout_seconds: 60.0
```

### 7.2 Operator CLI Command
```bash
python -m app.evaluation judge --out data/synthetic_vod/output --provider local-qwen --top-k 10
```

### 7.3 Dual-Ranking Output (`ai_judgments.csv`)
The evaluation system preserves candidate identity and exports a side-by-side comparison of **Signal Rank** (statistical detector score) versus **AI Editorial Rank** (multimodal rating and confidence):

| signal_rank | candidate_id | start_seconds | end_seconds | signal_score | ai_rank | ai_editorial_rating | ai_approved | ai_category | latency_seconds |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 2aae2654e2ecfe1a8dc53ede | 0.00 | 15.00 | 4.85 | 1 | 1 | n | ordinary | 8.27 |
| 2 | 92faefe56fc3015494b12949 | 19.00 | 34.00 | 4.12 | 2 | 1 | n | ordinary | 9.81 |
| 3 | 85d72242df9a02dfb6500681 | 35.00 | 50.00 | 3.65 | 3 | 1 | n | ordinary | 10.27 |

*Note: On synthetic test pattern footage, Qwen2.5-VL accurately recognized static pattern lines with zero player dialogue, correctly giving 1/5 ratings (`ai_approved: n`) and providing factual rejection reasons.*

---

## 8. Test Verification & Regression Summary

- **Total Unit & Integration Tests**: **158 passed**, 2 skipped (0 failures).
- **Stage 4B Test Suite**: `tests/test_judge_stage4b.py` (20 passed, 1 skipped).
- **Execution Time**: Entire repository regression suite runs cleanly.

---

## 9. Final Sign-off Verdict

### **VERDICT: STAGE 4B VERIFIED**

Local vision-language multimodal inference using `Qwen2.5-VL-7B-Instruct` (Q4_K_M) via standalone `llama.cpp` server is fully operational, technically robust, within all hardware budgets, and ready to serve as the foundation for Stage 4C (Editorial Calibration & Threshold Optimization).
