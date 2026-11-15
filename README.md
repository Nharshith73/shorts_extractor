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

Set the seven values in `config.yaml`, or pass a permitted source directly on the command line. `reanchor_threshold_seconds` defaults to `2.0`: within a session, segment timestamps are chained from media duration unless the filesystem timestamp differs by more than this amount. A local file with `is_live: false` is intentionally read with FFmpeg's `-re` flag, so it simulates a real-time source.

```powershell
python -m app.ingestion --source .\path\to\authorized-recording.mp4 --config config.yaml
```

Segments are written to `segments_dir` as MPEG-TS files with a timestamp in the filename. `manifest.sqlite3` stores the corrected UTC capture-start time, raw filesystem creation time (`fs_created_utc`), duration, status, source ID, persistent pin flag, re-anchor flag, and FFmpeg session ID. A new session ID is created after every reconnect, so downstream stages can identify candidate windows spanning a capture gap. Existing manifests are migrated in place on startup. Copy-mode segmentation preserves source packets and only starts a new segment at existing keyframes, so actual durations can vary with the source GOP size.

Run unit tests without FFmpeg:

```powershell
python -m pytest -m "not integration"
```

After FFmpeg is installed, run the complete test suite (the integration capture test is approximately two minutes):

```powershell
python -m pytest
```

The 30-minute local acceptance run creates its own test source by default and reports per-session manifest/file counts, timestamp correction, capture-end alignment, and retention cleanup. It deletes and recreates its work directory at the beginning of every run; `--work-dir` is therefore limited to a child of this repository's `data` directory. It is intentionally not run as part of ordinary testing. Both duration and retention can be shortened for a local check:

```powershell
python scripts\soak_test.py --duration-seconds 1800
python scripts\soak_test.py --duration-seconds 120 --retention-seconds 60
```

## Running Stage 2

Stage 2 performs local, lightweight audio and video analysis only—no LLM, speech-to-text, chat API, or publishing integration is involved. It needs the Stage 1 `ffmpeg`/`ffprobe` setup plus NumPy and SciPy from `requirements.txt`.

Detection settings are under `detection:` in `config.yaml`. `scale_floors` and `relative_scale_floor` keep near-constant audio or video baselines from turning tiny codec fluctuations into large z-scores; each feature uses the largest of scaled MAD, its absolute floor, and the relative floor. Raw one-second features are persisted in the Stage 1 manifest database, making score thresholds and weights re-tunable without decoding media again.

Follow newly completed segments while Stage 1 is running:

```powershell
python -m app.detection --config config.yaml
```

Analyze the current backlog once and exit:

```powershell
python -m app.detection --config config.yaml --once
```

For offline tuning, write an event timeline and raw feature CSV from an authorized local recording:

```powershell
python -m app.detection.offline --video .\recording.mp4 --out .\timeline.json --features .\features.csv
```

The offline runner logs its real-time factor (footage seconds divided by processing seconds). Benchmark a 10-minute local file on the intended CPU; the initial target is at least `10x` real time.

Evaluate a timeline against `labels.csv` (`time_seconds,note`), with optional chronological tune/holdout splitting:

```powershell
python -m app.detection.evaluate --events .\timeline.json --labels .\labels.csv --tolerance 8
python -m app.detection.evaluate --events .\timeline.json --labels .\labels.csv --tolerance 8 --split
```

Explain the existing feature CSV around each label without decoding the video again:

```powershell
python -m app.detection.explain --features .\features.csv --labels .\data\labels\labels.csv --tolerance 8
```

The explanation prints the effective config, per-feature maxima and times, rule outcomes, and blocking reasons. Its background comparison samples 200 full-width windows with replacement using seed 42; they do not overlap any labeled window. Scale-floor suppression is a per-bin counterfactual with the configured absolute and relative floors disabled.

## Running Stage 3

Stage 3 reads the existing `events` table and exports source-orientation MP4 candidates with JSON sidecars. Install the existing requirements and make `ffmpeg` and `ffprobe` available on `PATH`.

```powershell
python -m app.clips --config config.yaml
python -m app.clips --config config.yaml --once
```

Settings live under `candidates:`. Defaults are 15 seconds of pre-roll, 10 seconds of post-roll, 20–60 second clips, a 3 second merge gap, a 0.5 second live-edge grace period, and a 180 second wait timeout. `output_dir` defaults to `data/candidates`, relative to the config file. The final cut is re-encoded using `libx264 -preset veryfast -crf 18` and AAC, with `+faststart`; `video_encoder: h264_nvenc` selects NVIDIA encoding with preset p4 and CQ 18 when supported by the installed FFmpeg/GPU.

Windows start before the earlier of the region start and peak, and end after the region end. Long windows are limited to 60 seconds while retaining the peak and at least its configured pre-roll; short windows extend to 20 seconds. Windows from the same source/session merge when they overlap or are at most 3 seconds apart and the union is at most 60 seconds. Otherwise they remain separate. Merged windows use the highest-scoring event's peak when choosing the side of a discontinuity. Gap trimming may reduce the normal pre-roll; windows shorter than the minimum after trimming are `rejected_gap`.

The candidate ledger snapshots event IDs/scores and deduplicates by source and peak time, which survives detection replacing its event rows. Later events can merge into a waiting candidate; exported candidates retain their original snapshot. Available segments plus one neighboring segment on each side are reserved immediately; later segments are reserved on each poll. Shared reservations keep overlapping candidates protected independently. Candidates wait for completed coverage through the end plus grace, become `expired` when coverage has already disappeared, and become `timeout` if coverage does not arrive. Terminal statuses release their reservations, preserving manual pins. Successful exports produce `<id>.mp4` and `<id>.json`. Interrupted `extracting` rows are checked and retried by the next worker; output replacement is atomic per file. One OS-locked worker per manifest prevents simultaneous exports of the same candidate.

Ingestion startup now probes unfinished manifest rows and orphan segment files before retention starts. Playable MPEG-TS files are recovered with their measured duration and original session when identifiable; corrupt/missing files become abandoned. A restart boundary is recorded. A segment directory must belong to one ingestion source; an orphan filename identifies its session but cannot identify its source independently. Recovery verifies playable media, not that an interrupted final segment contains every frame the source originally intended.

Stage 3 unit and media tests run with the normal suite:

```powershell
python -m pytest -v -rs
```

## License

To be determined before public release.
