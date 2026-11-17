# StreamClip AI — Stage 4: AI Highlight Judge Technical Architecture Specification

## 1. Executive Summary & Purpose

StreamClip AI is an automated gaming-content production engine designed to transform hours of raw gameplay VODs and livestreams into high-retention vertical short-form videos (YouTube Shorts, TikTok, Instagram Reels).

In the current Phase 0/0.1 pipeline, candidate highlight windows are selected via heuristic signal processing (audio energy, voice RMS, motion dynamics, cut changes, Twitch chat frequency). The decision to approve or reject a candidate currently relies on human review via `review.csv`.

**Stage 4 (AI Highlight Judge)** introduces an autonomous editorial evaluation layer between Candidate Selection (Stage 3) and Vertical Shorts Rendering (Stage 5). It analyzes the candidate clips using local multimodal models (Vision-Language Models and LLMs) to evaluate entertainment value, narrative coherence, hook potential, and visual clarity—without relying on cloud APIs or recurring external costs.

> **Status Notice**: This document specifies the planned technical architecture for Stage 4. Stage 4 implementation code is **not yet implemented** in Phase 0.1 and will be introduced in subsequent phases following formal sign-off.

```
+---------------------------------------------------------------------------------------------------+
|                                  STREAMCLIP AI PIPELINE                                           |
|                                                                                                   |
|  [Stage 1: Ingestion]  -->  [Stage 2: Detection]  -->  [Stage 3: Candidate Selection & Previews]   |
|                                                                      |                            |
|                                                                      v                            |
|                                                     +---------------------------------+           |
|                                                     |   STAGE 4: AI HIGHLIGHT JUDGE   |           |
|                                                     |   - Multimodal Analysis         |           |
|                                                     |   - Narrative & Hook Scoring    |           |
|                                                     |   - Auto-Review / Triage        |           |
|                                                     |   - Layout & Crop Optimization  |           |
|                                                     +---------------------------------+           |
|                                                                      |                            |
|                                                                      v                            |
|                                                     [Stage 5: Vertical Shorts Rendering]          |
|                                                     (1080x1920 30fps + ASS Word Captions)         |
+---------------------------------------------------------------------------------------------------+
```

---

## 2. System Interfaces and Data Flow

### 2.1 Input Data Artifacts
Stage 4 ingests the artifacts produced by Stage 2 and Stage 3:
1. **Candidate Metadata (`clips.json`)**: Contains list of candidate dictionaries (`window_id`, `start_seconds`, `end_seconds`, `peak_seconds`, `score`, `components`, `unclipped_z_scores`, `rule`).
2. **Review Ledger (`review.csv`)**: Editorial ledger containing existing human decisions and prior triage notes.
3. **Extracted Horizontal Preview Clips (`clips/*.mp4`)**: Frame-accurate cuts extracted from source VOD with H.264/AAC.
4. **Speech Transcripts (`.transcripts/*.json`)**: Faster-whisper word-level transcripts with timing, average log-probability, and `no_speech_prob`.
5. **Auxiliary Context**: Chat activity logs (Twitch chat JSON/CSV), game metadata, and streamer voice activity profiles.

### 2.2 Output Deliverables
1. **Judge Ledger Updates (`review.csv`)**:
   - `approved`: Pre-populated with `y` (auto-approved high confidence), `n` (auto-rejected), or left blank / `m` (requires manual human review).
   - `notes`: Structured rationale explaining the editorial judgment (e.g., `"[AI] High excitement, clear clutch kill, punchy hook"`).
   - `reason_code`: Canonical categorization code (`clutch_play`, `funny_reaction`, `abrupt_death`, `low_action_filler`, `garbled_audio`).
2. **Detailed Editorial Metadata (`judgments/<window_id>.json`)**:
   - Detailed per-clip editorial scoring (Hook, Narrative, Humor, Visual clarity, Virality potential).
   - Recommended Shorts title and 3-word visual hook overlay text.
   - Dynamic layout recommendation (`blurred_background`, `stacked`, `center_crop`) with suggested facecam bounding box.
   - Trim refinements (suggested micro-trims to snap precisely to speech boundaries or punchlines).

---

## 3. Multimodal Analysis Architecture

The AI Highlight Judge employs a hierarchical, 3-tier multimodal evaluation pipeline to minimize compute overhead while maximizing editorial precision:

```
[Candidate Ingestion]
        |
        v
+-------------------------------------------------------+
| Tier 1: Transcript & Semantic Pre-Filter (Fast NLP)   |
| - Fast local LLM (Qwen2.5-3B-Instruct / Llama-3.2-3B) |
| - Evaluates dialogue humor, climax, coherence         |
| - Discards unambiguous dead air / silence / filler   |
+-------------------------------------------------------+
        |  (Passes threshold >= 0.40)
        v
+-------------------------------------------------------+
| Tier 2: Visual Action & Keyframe Extraction           |
| - Keyframe sampler: 1 fps uniform + peak frame + cut  |
| - 8-16 downsampled frames (512x288) per clip          |
| - Action verification (killfeed, HUD status, action)  |
+-------------------------------------------------------+
        |
        v
+-------------------------------------------------------+
| Tier 3: Multimodal Vision-Language Reasoning          |
| - Local VLM (Qwen2.5-VL-7B-Instruct / LLaVA-v1.6-7B)  |
| - Synthesizes visual dynamics, face reaction, audio   |
| - Emits strictly validated JSON structured judgment   |
+-------------------------------------------------------+
```

### 3.1 Tier 1: Semantic & Dialogue Analysis
- Analyzes word-timed transcript from faster-whisper.
- Detects narrative markers: excitement vocabulary, vocal reactions ("No way!", "Watch this!"), storytelling structure, or punchline deliveries.
- Calculates speech pacing (words per minute), pause cadence, and voice pitch variation.

### 3.2 Tier 2: Visual Feature Extraction
- Rather than feeding high-bandwidth 60fps video directly to the VLM (which would saturate VRAM and exceed token context limits), Stage 4 extracts a compact frame matrix:
  - **Hook Frame**: Frame at 0.5s (initial viewer impression).
  - **Peak Action Frame**: Frame at `peak_seconds` (highest signal excitement).
  - **Uniform Context Frames**: 1 frame every 2.0 seconds.
  - **Resolution**: Staged as 512x288 or 640x360 tiles in RGB grid format.

### 3.3 Tier 3: Multimodal Editorial Reasoning
- The VLM receives the structured system prompt, the composite image grid, the transcript with timestamps, and the signal telemetry (audio RMS z-score, motion delta).
- Produces a deterministic evaluation against GTA-specific and gaming-content editorial rubrics.

---

## 4. Structured JSON Output Schema

All AI Judge outputs are strictly constrained to valid JSON matching the following schema. Any response failing validation is automatically rejected and routed to manual review.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "StreamClipHighlightJudgment",
  "type": "object",
  "required": [
    "window_id",
    "decision",
    "confidence",
    "editorial_scores",
    "primary_category",
    "reasoning",
    "suggested_title",
    "hook_text",
    "recommended_layout",
    "suggested_bounds"
  ],
  "properties": {
    "window_id": {
      "type": "string",
      "description": "24-character hex identifier matching candidate window_id"
    },
    "decision": {
      "type": "string",
      "enum": ["APPROVE", "REJECT", "UNCERTAIN_NEEDS_HUMAN"]
    },
    "confidence": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 1.0,
      "description": "Calibrated decision confidence score"
    },
    "editorial_scores": {
      "type": "object",
      "required": ["hook_strength", "narrative_coherence", "entertainment_value", "visual_clarity", "pacing"],
      "properties": {
        "hook_strength": { "type": "number", "minimum": 0.0, "maximum": 10.0 },
        "narrative_coherence": { "type": "number", "minimum": 0.0, "maximum": 10.0 },
        "entertainment_value": { "type": "number", "minimum": 0.0, "maximum": 10.0 },
        "visual_clarity": { "type": "number", "minimum": 0.0, "maximum": 10.0 },
        "pacing": { "type": "number", "minimum": 0.0, "maximum": 10.0 }
      }
    },
    "primary_category": {
      "type": "string",
      "enum": [
        "clutch_kill",
        "hilarious_fail",
        "car_chase_action",
        "funny_dialogue",
        "glitch_bug",
        "streamer_rage",
        "boring_inventory_management",
        "repetitive_driving",
        "dead_air_silence"
      ]
    },
    "reasoning": {
      "type": "string",
      "maxLength": 300,
      "description": "Concise 1-2 sentence human-readable rationale"
    },
    "suggested_title": {
      "type": "string",
      "maxLength": 80,
      "description": "High-CTR YouTube Shorts / TikTok title"
    },
    "hook_text": {
      "type": "string",
      "maxLength": 30,
      "description": "1-3 uppercase punchy words for the top video overlay"
    },
    "recommended_layout": {
      "type": "string",
      "enum": ["blurred_background", "stacked", "center_crop"]
    },
    "suggested_bounds": {
      "type": "object",
      "required": ["refined_start_seconds", "refined_end_seconds"],
      "properties": {
        "refined_start_seconds": { "type": "number" },
        "refined_end_seconds": { "type": "number" },
        "facecam_detected": { "type": "boolean" },
        "facecam_region": {
          "type": ["object", "null"],
          "properties": {
            "x": { "type": "integer" },
            "y": { "type": "integer" },
            "w": { "type": "integer" },
            "h": { "type": "integer" }
          }
        }
      }
    }
  }
}
```

### Example Valid JSON Output
```json
{
  "window_id": "4a7c88b9012e8471c2ff9012",
  "decision": "APPROVE",
  "confidence": 0.92,
  "editorial_scores": {
    "hook_strength": 8.5,
    "narrative_coherence": 8.0,
    "entertainment_value": 9.0,
    "visual_clarity": 8.5,
    "pacing": 9.0
  },
  "primary_category": "clutch_kill",
  "reasoning": "Player escapes high-speed 5-star police chase and executes a clean sniper shot while airborne. High vocal energy and clean payoff.",
  "suggested_title": "HE DID NOT JUST HIT THAT ESCAPE 😱",
  "hook_text": "WAIT FOR THE SHOT",
  "recommended_layout": "stacked",
  "suggested_bounds": {
    "refined_start_seconds": 142.20,
    "refined_end_seconds": 168.50,
    "facecam_detected": true,
    "facecam_region": { "x": 1540, "y": 60, "w": 340, "h": 260 }
  }
}
```

---

## 5. Local Inference Engine & Hardware Budgeting

### 5.1 Local Model Selection
To ensure 100% offline self-hosting and zero API cost:
1. **Primary VLM Choice**: `Qwen2.5-VL-7B-Instruct` (Quantized via GGUF Q4_K_M or AWQ / EXL2)
   - *Strengths*: State-of-the-art spatial bounding box detection (for automatic facecam localization), superior gaming HUD understanding, and structured JSON reliability.
2. **Fallback VLM Choice**: `LLaVA-v1.6-Mistral-7B` or `MiniCPM-V-2.6` (for 6GB-8GB VRAM constrained GPUs).
3. **Serving Infrastructure**:
   - `llama.cpp` server or `ollama` running locally as a detached background daemon.
   - OpenAI-compatible `/v1/chat/completions` REST endpoint with grammar-based JSON schema enforcement (`response_format={"type": "json_object"}`).

### 5.2 GPU & VRAM Memory Budgeting
On a standard operator workstation with an NVIDIA GPU (e.g., RTX 3060 12GB, RTX 4070 12GB, or RTX 3080/4080 16GB):

| Component | VRAM Allocation | Compute Device | Notes |
| :--- | :--- | :--- | :--- |
| **Qwen2.5-VL-7B (Q4_K_M)** | 5.2 GB | GPU (CUDA) | Handles vision + reasoning |
| **faster-whisper (small)** | 0.0 GB (or 1.2 GB) | CPU (int8) / GPU | Runs on CPU int8 by default |
| **NVENC Hardware Encoder** | 0.4 GB | NVENC ASIC | Independent dedicated silicon |
| **FFmpeg GPU Surface/Filters**| 0.6 GB | GPU (CUDA/VMM) | Video frame decoding/scaling |
| **CUDA Driver & Overhead** | 0.8 GB | GPU | Base operating overhead |
| **Total Peak VRAM Required** | **7.0 GB** | GPU | Leaves 5.0 GB headroom on 12GB cards |

*Graceful CPU Fallback*: If running on non-NVIDIA or integrated graphics machines, Tier 1 uses CPU GGUF (`Qwen2.5-3B-Instruct` via `llama-cpp-python` with AVX-512) and skips deep pixel analysis, relying on transcript and Stage 2 motion/energy heuristics.

---

## 6. Editorial Calibration and Human-Agreement Metrics

To evaluate Stage 4's performance against human editors before enabling autonomous publishing, the system implements standard offline evaluation metrics against historical `review.csv` decisions:

### 6.1 Core Metrics
1. **Approval Agreement Rate (Precision)**:
   $$\text{Precision} = \frac{\text{True Positives (AI Approved \& Human Approved)}}{\text{Total AI Approved}}$$
   *Target Threshold*: $\ge 85\%$ (ensure low false-positive rate so low-quality shorts are never rendered).
2. **Highlight Capture Rate (Recall)**:
   $$\text{Recall} = \frac{\text{True Positives (AI Approved \& Human Approved)}}{\text{Total Human Approved}}$$
   *Target Threshold*: $\ge 80\%$ (ensure the model doesn't overlook genuine viral moments).
3. **Cohen's Kappa ($\kappa$)**:
   Measures inter-annotator agreement between AI Judge and human operators, accounting for chance agreement:
   $$\kappa = \frac{p_o - p_e}{1 - p_e}$$
   *Target Threshold*: $\kappa \ge 0.70$ (substantial agreement).

### 6.2 Calibration & Auto-Triage Thresholds
The AI Judge computes a continuous confidence score $C \in [0, 1]$:
- **$C \ge 0.85$**: **Auto-Approve** $\to$ Automatically set `approved: "y"` in `review.csv` and schedule for Stage 5 vertical rendering.
- **$0.40 \le C < 0.85$**: **Triage Flag** $\to$ Set `approved: ""` (pending), attach detailed AI reasoning notes, and prioritize at top of human review queue.
- **$C < 0.40$**: **Auto-Reject** $\to$ Set `approved: "n"`, record reason code (e.g., `filler_dialogue`, `repetitive_driving`), and skip rendering.

---

## 7. Reliability, Fallbacks, and Guardrails

1. **Non-Destructive Execution**: Stage 4 operates as a pure additive filter. It never deletes raw clips, never modifies source VODs, and never overwrites an existing human approval in `review.csv`.
2. **Schema Validation & Fallback**: If the local VLM outputs invalid JSON or hallucinates nonexistent keys, the parser catches `jsonschema.ValidationError`, logs the raw response to `judgments/errors/<window_id>.log`, marks the candidate as `UNCERTAIN_NEEDS_HUMAN`, and leaves `approved: ""` for manual inspection.
3. **Rate Limiting & Concurrency**: Inferences run sequentially with a configurable batch queue (default concurrency = 1) to prevent GPU out-of-memory (OOM) conditions during concurrent FFmpeg renders.
4. **Offline Isolation**: All model weights, tokenizers, and inference backends are strictly local. No network calls are made during evaluation.
