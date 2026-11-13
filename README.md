# StreamClip AI 🎮

**Turn livestream highlights into engaging vertical Shorts—locally, with zero paid LLM API calls.**

> **Status:** Planning / pre-implementation. This README describes the intended architecture and milestones; the application is not yet implemented.

StreamClip AI is a proposed Python-first pipeline for monitoring gameplay livestreams (initially GTA 6), finding exciting moments, evaluating candidate clips with a **local vision-language model**, and automatically editing selected moments into vertical videos for YouTube Shorts, Instagram Reels, and TikTok.

## Goals

- Monitor a live video source and maintain a timestamped rolling buffer.
- Identify potential highlights using inexpensive audio, motion, and optional chat signals.
- Extract candidate clips with enough context before and after the event.
- Use **one local multimodal model request per shortlisted candidate**, where practical, to evaluate the moment and propose an edit plan.
- Generate 9:16 MP4 Shorts with readable subtitles, optional hook text, sensible framing, and clean audio.
- Review and approve clips before exporting or publishing.
- Keep **paid LLM API calls at zero**; local compute, storage, bandwidth, and electricity still have costs.

## End-to-end workflow

```text
Twitch / YouTube Live / permitted test video input
                       |
                       v
[1] Ingestion + timestamped rolling video buffer
                       |
                       v
[2] Lightweight excitement detection
    - Audio energy / sudden vocal peaks
    - Gameplay motion / visual changes
    - Chat activity spikes (when accessible)
                       |
                       v
[3] Candidate clip extraction
    - Include lead-in and reaction
    - Merge overlapping events
    - Save source timestamps and metadata
                       |
                       v
[4] Local AI highlight judge + editing plan
    - Sample frames + transcript + event metadata
    - Rank excitement, clarity, payoff
    - Suggest hook, cut boundaries, layout
                       |
                       v
[5] Automated Shorts editor
    - Trim and reframe to 9:16
    - Transcribe speech locally
    - Add styled, timed subtitles
    - Normalize audio; render MP4
                       |
                       v
[6] Local review dashboard
    - Preview / approve / reject / regenerate
                       |
                       v
[7] Export and optional platform publishing
    - YouTube Shorts / Instagram Reels / TikTok
```

### Model usage

| Stage | LLM calls | Primary tools |
| --- | --- | --- |
| Stream ingestion | 0 | FFmpeg, Python |
| Excitement detection | 0 | NumPy, librosa, OpenCV; optional chat API |
| Candidate extraction | 0 | FFmpeg |
| Highlight evaluation + edit plan | ~1 **local** request per shortlisted clip | Local vision-language model via Ollama |
| Speech transcription | 0 LLM calls | faster-whisper (local ASR model) |
| Captions, effects, rendering | 0 | FFmpeg, ASS/libass, OpenCV |
| Review and export | 0 | Streamlit, Python |

**No paid LLM APIs are required.** The local model is still an AI inference workload; actual throughput depends on CPU, RAM, GPU, model size, and frame sampling.

## Proposed tech stack

| Area | Planned technology |
| --- | --- |
| Language | Python 3.11+ |
| Media capture and rendering | FFmpeg / ffprobe |
| Audio feature extraction | NumPy, librosa |
| Video analysis | OpenCV; optionally PySceneDetect |
| Local speech-to-text | faster-whisper |
| Local multimodal reasoning | Ollama + a compatible vision-language model (benchmark before choosing) |
| Captions | ASS subtitles rendered through FFmpeg/libass |
| Backend | FastAPI (later milestone) |
| Data | SQLite for metadata, local files for video |
| Review interface | Streamlit |
| Background jobs | Simple Python workers first; queue later if needed |

**Model selection:** Benchmark a small local vision-language model against actual gameplay clips. Sparse frames can miss fast action, so retain original footage and use additional frames or a second pass when confidence is low.

## Staged implementation plan

### Stage 1 — Stream ingestion and rolling buffer

**Build:** A source adapter that reads a permitted stream or local video, writes small timestamped segments, retains a configurable recent window, and handles reconnects.

**Acceptance test:** Capture a continuous 30-minute test source without losing usable segments; verify segment timestamps and retention cleanup.

### Stage 2 — Lightweight excitement detection

**Build:** Audio energy/voice spikes, frame motion changes, optional chat-rate spikes, and a combined anomaly score based on each stream's baseline.

**Acceptance test:** Produce an event timeline and find obvious exciting moments in labeled gameplay footage without relying on a language model.

### Stage 3 — Candidate clip extraction

**Build:** Extract approximately 20–60 second windows around events, including lead-in and payoff; merge overlaps and deduplicate candidates.

**Acceptance test:** Export playable MP4 candidates with correct source timestamps and sufficient context.

### Stage 4 — Local AI highlight judge

**Build:** Sample frames, add transcript snippets and event signals, then request a structured JSON assessment and editing plan from a local multimodal model.

**Acceptance test:** Compare ranked clips against a human-reviewed sample; measure how many recommended clips are genuinely worth posting.

Example *proposed* output:

```json
{
  "is_worth_clipping": true,
  "excitement_score": 91,
  "event": "Unexpected escape during a police chase",
  "reason": "The player narrowly avoids a collision and escapes",
  "suggested_hook": "HOW DID THEY ESCAPE?",
  "clip_start_seconds": 12.5,
  "clip_end_seconds": 43.0,
  "layout": "gameplay_with_facecam",
  "caption_style": "gaming_bold"
}
```

The application must validate JSON, timestamps, and claims before rendering. AI scores are editorial heuristics, **not** reliable predictions of virality.

### Stage 5 — Automated Shorts editor

**Build:** Context-aware trimming; 9:16 layouts; local transcription and optional word alignment; bold subtitles; optional on-screen hook; audio normalization; MP4 export.

**Default editing style:** Keep the original gameplay action prominent, avoid excessive effects, and preserve setup → action → payoff. Prefer letterbox/blurred-background or stacked layouts when a center crop would hide important gameplay.

**Acceptance test:** Render a 1080×1920 video with synchronized, legible captions and no important gameplay cut out.

### Stage 6 — Review dashboard

**Build:** Stream status, candidate timeline, highlight scores, clip previews, and approve/reject/regenerate actions.

**Acceptance test:** Review and export clips through a local UI without manually editing files.

### Stage 7 — Publishing and feedback

**Build:** Export platform-ready MP4s and metadata; later add official platform publishing integrations and performance tracking.

**Acceptance test:** Successfully export and manually publish a test clip. Automated posting requires separate platform integration and approvals.

## Proposed repository layout

```text
streamclip-ai/
├── app/
│   ├── ingestion/
│   │   ├── stream_reader.py
│   │   └── rolling_buffer.py
│   ├── detection/
│   │   ├── audio_detector.py
│   │   ├── motion_detector.py
│   │   ├── chat_detector.py
│   │   └── excitement_scorer.py
│   ├── clips/
│   │   └── candidate_extractor.py
│   ├── ai/
│   │   ├── local_model.py
│   │   └── highlight_judge.py
│   ├── editing/
│   │   ├── subtitle_generator.py
│   │   ├── vertical_layout.py
│   │   └── video_renderer.py
│   ├── publishing/
│   │   └── exporter.py
│   └── api/
│       └── main.py
├── dashboard/
│   └── app.py
├── data/
│   ├── segments/
│   ├── candidates/
│   └── output/
├── tests/
├── config.yaml
└── requirements.txt
```

> The tree is a **target structure**, not a list of files already implemented.

## Key engineering principles

1. **Filter cheaply first.** Continuous analysis uses lightweight signal processing; local multimodal inference only runs on shortlisted moments.
2. **Keep footage and metadata together.** Preserve source ID, event timestamps, clip boundaries, and model output.
3. **Make editing deterministic.** The local model proposes an edit plan; Python and FFmpeg execute and validate it.
4. **Preserve context.** Include the setup and payoff rather than clipping only the loudest second.
5. **Measure quality.** Track human approval rate, false positives, processing latency, and render failures.
6. **Start with one source.** Prove the pipeline on a recording, then on one live stream, before monitoring multiple sources.
7. **Separate publishing from processing.** Clip creation should work even if platform publishing is unavailable.

## Initial MVP targets

| Metric | Target (to validate) |
| --- | --- |
| Concurrent sources | 1 initially; scale to 3 |
| Detection latency | Aim for <60 seconds after an event |
| Candidate length | Approximately 20–60 seconds |
| Export resolution | 1080×1920 (9:16) |
| Paid LLM API requests | 0 |
| Local model requests | Aim for 1 per shortlisted candidate |
| Human approval | Required for initial releases |

## Rights and platform considerations

Technical prototyping and distribution are separate workstreams. Use test footage or sources you are authorized to process. Before publishing third-party livestream footage, verify the necessary rights, applicable game-publisher policies, music usage, and each platform's reused-content and publishing requirements. Public stream availability does not itself grant redistribution rights.

## Next step

**Implement Stage 1:** confirm operating system and hardware, install FFmpeg and Python, build the segmented recorder and rolling-buffer retention, and test with a local recording before connecting a live source.

## Running Stage 1

Stage 1 is implemented as a local FFmpeg-supervised capture process. It needs Python 3.11+ and both `ffmpeg` and `ffprobe` on `PATH`. On Windows, download a trusted FFmpeg build, add its `bin` directory to `PATH`, open a new PowerShell window, and confirm:

```powershell
python --version
ffmpeg -version
ffprobe -version
```

Create an environment and install the two Python dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Set the six values in `config.yaml`, or pass a permitted source directly on the command line. A local file with `is_live: false` is intentionally read with FFmpeg's `-re` flag, so it simulates a real-time source.

```powershell
python -m app.ingestion --source .\path\to\authorized-recording.mp4 --config config.yaml
```

Segments are written to `segments_dir` as MPEG-TS files with a timestamp in the filename. `manifest.sqlite3` in that directory stores the authoritative UTC capture-start time, duration, status, and source ID. Copy-mode segmentation preserves source packets and only starts a new segment at existing keyframes, so actual durations can vary with the source GOP size.

Run unit tests without FFmpeg:

```powershell
python -m pytest -m "not integration"
```

After FFmpeg is installed, run the complete test suite (the integration capture test is approximately two minutes):

```powershell
python -m pytest
```

The 30-minute local acceptance run creates its own test source by default and reports playable/corrupt segment counts, timestamp drift, and retention cleanup. It is intentionally not run as part of ordinary testing:

```powershell
python scripts\soak_test.py --duration-seconds 1800
```

## License

To be determined before public release.
