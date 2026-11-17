# StreamClip AI 🎮

**Turn livestream highlights into engaging vertical Shorts—locally, with zero paid LLM API calls.**

> **Status:** Stages 1–3, offline VOD mode, and reviewed-clip vertical rendering with captions are implemented. Local AI judging, a full dashboard, and publishing remain planned.

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

Create an environment and install the Python dependencies:

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

Settings live under `candidates:`. Defaults are 15 seconds of pre-roll, 10 seconds of post-roll, 20–60 second clips, a 3 second merge gap, a 0.5 second live-edge grace period, and a 180 second wait timeout. `output_dir` defaults to `data/candidates`, relative to the config file. MP4/VOD cuts use input-side accurate seeking (`-ss` before `-i`) and re-encoding, AAC, and `+faststart`. Transport streams (`.ts`, `.mts`, `.m2ts`, `.mpegts`) retain output-side seeking because unindexed TS seeks can skip required video keyframes; buffered joins are short.

The shipped config uses `candidates.encoder: auto`. Before its first export, each worker/VOD run tries a real one-second `h264_nvenc` encode at the source resolution. Auto selects NVENC only when the encode succeeds and produces frames; an encoder appearing in `ffmpeg -encoders` is insufficient. A failed probe logs the FFmpeg cause (including driver/API incompatibilities) and falls back to `libx264 -preset veryfast -crf 18`. The decision is cached for that worker/run. Explicit `encoder: libx264` skips the probe; explicit `encoder: h264_nvenc` requires usable NVIDIA encoding with preset p4 and CQ 18 and fails if it is unavailable. The older `video_encoder` key remains supported, but conflicting values for both keys are rejected.

Windows start before the earlier of the region start and peak, and end after the region end. Long windows are limited to 60 seconds while retaining the peak and at least its configured pre-roll; short windows extend to 20 seconds. Windows from the same source/session merge when they overlap or are at most 3 seconds apart and the union is at most 60 seconds. Otherwise they remain separate. Merged windows use the highest-scoring event's peak when choosing the side of a discontinuity. Gap trimming may reduce the normal pre-roll; windows shorter than the minimum after trimming are `rejected_gap`.

The candidate ledger snapshots event IDs/scores and deduplicates by source and peak time, which survives detection replacing its event rows. Later events can merge into a waiting candidate; exported candidates retain their original snapshot. Available segments plus one neighboring segment on each side are reserved immediately; later segments are reserved on each poll. Shared reservations keep overlapping candidates protected independently. Candidates wait for completed coverage through the end plus grace, become `expired` when coverage has already disappeared, and become `timeout` if coverage does not arrive. Terminal statuses release their reservations, preserving manual pins. Successful exports produce `<id>.mp4` and `<id>.json`. Interrupted `extracting` rows are checked and retried by the next worker; output replacement is atomic per file. One OS-locked worker per manifest prevents simultaneous exports of the same candidate.

Ingestion startup now probes unfinished manifest rows and orphan segment files before retention starts. Playable MPEG-TS files are recovered with their measured duration and original session when identifiable; corrupt/missing files become abandoned. A restart boundary is recorded. A segment directory must belong to one ingestion source; an orphan filename identifies its session but cannot identify its source independently. Recovery verifies playable media, not that an interrupted final segment contains every frame the source originally intended.

Stage 3 unit and media tests run with the normal suite:

```powershell
python -m pytest -v -rs
```

## Running VOD mode

VOD mode analyzes a downloaded local file without capture, real-time throttling, or the rolling buffer. One command writes raw features, ranked event peaks, clips and JSON sidecars, `review.csv`, and a static `index.html` with embedded video players.

```powershell
python -m app.vod run --video E:\test_footage\gameplay.mp4 --out data\vod-gameplay --config config.yaml
python -m app.vod run --video E:\test_footage\gameplay.mp4 --chat E:\test_footage\chat.json --out data\vod-with-chat --config config.yaml
```

Open `<out>/index.html` in a browser to watch the shortlist. Clips are named `clip_<rank>_<HH-MM-SS>_<score>.mp4`. The fixed synthetic epoch (`1970-01-01T00:00:00Z` plus media seconds) preserves original-file offsets, including when an intro/outro is skipped:

```powershell
python -m app.vod run --video E:\test_footage\gameplay.mp4 --out data\vod-trimmed --start 00:02:00 --end 00:15:00
```

`vod.chunk_seconds` defaults to 300 seconds. Bounded FFmpeg audio/video reads carry unfinished one-second bins, audio filter state and previous-frame state across chunks. The existing bin-local spectral flux receives the complete bin even if it spans chunks. Final partial bins are retained. Progress includes percentage, elapsed time and real-time factor. `run.json` records audio/video decode and feature times, chat, scoring, selection and extraction, along with sampled peak RSS for Python plus child processes (100ms sampling). Decode timing measures blocked pipe reads, not CPU profiling; decoder work can overlap Python work. Detection speed has not been tuned here.

`selection:` defaults to `clips_per_hour: 12`, `min_score: null`, and `min_separation_seconds: 30`. The budget is `ceil(clips_per_hour × analyzed hours)`, clamped to 3–40. Highest scores are selected with temporal separation after warmup, without requiring either threshold rule. A non-null min_score is the only score cutoff; short ranges, warmup, separation or a cutoff can leave fewer eligible clips. `events.json` contains the ranked shortlist and the original threshold events. Sidecars record `multi`, `single`, `multi+single` or `ranked_only`.

Final clip windows may overlap by at most **2 seconds**, independent of the peak separation setting. Selection keeps the higher-ranked window and continues down the ranking to fill the budget. Ranking uses capped combined score first, then the uncapped weighted combined score to break ties, then timestamp. The existing feature z-score clipping stays at `[0, 10]`. Sidecars and `review.csv` also store `unclipped_z_scores` and `unclipped_score`: the signed median/MAD z-scores before clipping, with the same scale floors, and their weighted sum. Raw negative values are retained for diagnostics; only the tie-break score uses that uncapped sum. Rescore updates these fields and the overlap policy from saved raw features without decoding the source.

VOD windows have independent defaults under `vod:`: pre-roll 8s, post-roll 6s, minimum 15s and maximum 45s. Pre-roll is strictly from the peak, never the earlier region edge. Minimum-length padding extends right only; if the file/trim end cannot accommodate it, that peak is skipped. Sidecars include actual pre/post offsets. Ranked clips remain independent rather than merging leftward and exceeding the peak pre-roll. Encoder settings still come from `candidates.encoder`; live ingestion/candidate window settings retain their behavior.

Raw `features.csv` (including chat columns), `analysis.json` (source fingerprint, trim and decode parameters), and the window cache allow fast rescoring:

```powershell
python -m app.vod rescore --out data\vod-gameplay --config config.yaml
python -m app.vod rescore --out data\vod-gameplay --config config.yaml --extract
```

Rescore does not decode media. Without `--extract`, it updates metadata/review pages and reuses existing clips; missing windows are shown as not extracted. `--extract` renders only uncached windows. Changing decode sample rate, video FPS or chat alignment requires a new `run`. Fresh extraction checks the source fingerprint; cached windows can be reused even with the source offline. Review decisions follow window IDs through rank/filename changes. `.clip-cache` retains validated exports, so ranked copies use extra disk space; older unlisted files remain.

Optional chat comes from a [TwitchDownloader JSON export](https://github.com/lay295/TwitchDownloader). Configure it under `detection.chat` in `config.yaml`: weights, scale floors, and `time_offset_seconds`. The offset is **added** to every comment offset: use `-600` if the local file begins at VOD second 600. Verify alignment yourself; the tool cannot infer a trimmed video's original offset. Missing/invalid timestamps are skipped, unknown authors do not inflate distinct-chatter counts, and empty text has zero emote/caps ratios.

Chat bins contain messages per second, distinct known authors, annotated/known emote tokens divided by total tokens, and uppercase letters divided by all alphabetic characters. Emote density is approximate. The same median/MAD baseline, warmup and clipping apply to chat. Zero chat weights disable those signals. With no chat file, underlying audio/video feature and threshold scoring calculations retain their behavior; VOD ranking additionally considers below-threshold peaks.

Review paths and metadata in the terminal; it does not open or play media. Enter `y` to approve, `n` to reject, `s` to skip, or `q` to save and quit. Rejection requires a reason: `b` boring/nothing happens, `e` cut too early, `l` cut too late, `m` moment missing, `n` noise/audio artifact, or `o` other. Responses and reason_code are persisted; notes can be edited in the CSV.

```powershell
python -m app.vod review --out data\vod-gameplay
python -m app.vod summary --out data\vod-gameplay
```

Summary reports approval rate among reviewed clips, unreviewed count, reasons breakdown and mean feature z-scores for approved versus rejected clips. Missing sidecars are counted and omitted from those averages.

Weak-label CSVs contain `start_seconds,duration_seconds,weight` (for example VOD offsets, Twitch clip durations, and view counts). Evaluation ranks **event peaks**, accepts a hit within the label interval plus tolerance, and performs one-to-one matching. Precision divides by the number of available predictions up to K; view-weighted recall divides matched label weights by total label weight. These are weak proxies for editorial quality. The random baseline samples the same number of uniform timestamps across the full file, averaged over 200 trials with seed 42; it can also land in warmup time.

```powershell
python -m app.vod evaluate --out data\vod-gameplay --labels weak_labels.csv --tolerance 10 --top-k 10
python -m pytest -v -rs
```

Media buffers are bounded by chunk_seconds; feature rows are retained for scoring. Chat JSON is still loaded as a whole, so large exports can dominate memory. Memory reports are sampled process-tree peaks, not exact allocation maxima. A ten-minute benchmark does not guarantee performance for other codecs, resolutions, hardware or clip counts. Below-threshold ranking deliberately favors a reviewable shortlist over guaranteed excitement.

## Running render

Final rendering reads the current `review.csv` and each preview's JSON sidecar. By default only rows with `approved == y` are rendered. An explicit rank selection overrides approval; `--all` selects every current review row. `--all` and `--ranks` are mutually exclusive.

```powershell
python -m pip install --only-binary=:all: -r requirements.txt
python -m app.vod render --out data\vod-gameplay
python -m app.vod render --out data\vod-gameplay --config render.yaml
python -m app.vod render --out data\vod-gameplay --ranks 1,3,5 --config render.yaml
python -m app.vod render --out data\vod-gameplay --all --config render.yaml
```

The default `--config config.yaml` uses rendering defaults and inherits `candidates.encoder`. Customize the dedicated `render.yaml`, which contains all `render:` keys. Keep using `config.yaml` for ingestion: its strict top-level schema is unchanged, and a render-only config is not an ingestion config.

Each final is cut frame-accurately from the **original source file** using the sidecar's media start/end, avoiding another generation of preview compression. The original must remain available. Output is `<out>/final/<preview-name>.mp4`, with a JSON settings/timing sidecar and an ASS file when captioned. Previews, review decisions, detection features and selection are preserved. `render-report.json` reports failures and throughput; the command returns nonzero if any selected export fails. Successful outputs replace prior exports only after media validation.

Speech recognition uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) on a temporary 16 kHz mono WAV containing only the selected clip. The first captioned run can download the selected model; later runs use the model cache. Default CPU int8 needs no CUDA. No model is downloaded for a selection with no approved clips or with captions disabled. Raw transcripts are cached in `<out>/.transcripts/`, keyed by source fingerprint, source window, model/language/device/compute settings and package versions. Changing caption style, layout, or hallucination thresholds reuses recognition results; changing the source or ASR settings creates a new cache entry.

| `render:` setting | Default / meaning |
| --- | --- |
| `whisper_model` | `small`; model name or local model directory |
| `compute_type`, `device`, `language` | `int8`, `cpu`, `auto`; devices also accept `auto` and `cuda` |
| `min_avg_logprob`, `max_no_speech_prob` | `-1.0`, `0.6`; reject a segment if either check fails |
| `captions_enabled`, `caption_style` | `true`, `bold_yellow`; other styles are `clean_white` and `boxed` |
| `uppercase`, `max_words`, `font_size` | `true`, `3`, `80`; 1–3 visible words per phrase |
| `margin_horizontal`, `margin_bottom` | `80`, `400` pixels; validation protects the bottom 18% UI area |
| `outline`, `shadow`, `fonts_dir` | `4`, `2`, `assets/fonts`; relative font paths resolve beside the config |
| `layout` | `blurred_background`; also `stacked` or `center_crop` |
| `facecam_region` | `null`; otherwise `{x, y, w, h}` in original-source pixels |
| `crop_center_x` | `0.5`; horizontal source center fraction, 0–1 |
| `target_lufs` | `-14`; single-pass loudnorm with -1.5 dBTP and LRA 11 |
| `encoder`, `crf`, `preset` | `auto`, `18`, `medium`; auto reuses the real NVENC probe and CPU fallback |

Captions use word timestamps and `vad_filter=True`, then discard suspect recognition segments. Phrase grouping considers word count, length, punctuation and pauses; discrete ASS events highlight only the current word. Caption timings are clip-relative, with original-source offsets also recorded in JSON. A clip without usable speech is still exported, with `caption_status: no_usable_speech` and no subtitles. A source without audio receives a silent AAC track and `source_has_no_audio`. The sidecar records rejected segment count, cache reuse, effective layout, font, encoder, caption text and word times.

Put licensed Anton or Poppins-Bold TTF/OTF files in `assets/fonts` if desired; none are downloaded automatically. The fallback family is Arial Black, subject to system substitution if unavailable. Font files are staged in an explicit relative `fonts` directory. FFmpeg runs beside `captions.ass` and references `subtitles=filename=captions.ass:fontsdir=fonts`, so neither drive letters nor backslashes enter the filter. This uses libass, never drawtext.

The blurred layout preserves the gameplay aspect ratio over a darkened blurred fill. Stacked uses a 1080px-high gameplay panel above an 840px-high facecam panel; gameplay retains a blurred fill and facecam fills its panel by cropping. Missing facecam coordinates cause a logged fallback to blurred background; out-of-frame coordinates fail clearly. Center crop can hide HUD/action, and a static facecam region does not track moving overlays. Final encoding is 1080×1920 at 30 fps, H.264, AAC 192k/48 kHz, and `+faststart`. NVENC uses p4/CQ instead of x264 preset/CRF. Single-pass loudness is approximate, not a measured two-pass guarantee.

Run all tests normally. The real speech test is opt-in and can download the `small` model; the synthetic render tests inject fake words and need no model:

```powershell
python -m pytest -v -rs
$env:STREAMCLIP_SPEECH_SAMPLE = 'E:\test_footage\short-speech.wav'
python -m pytest tests\test_render_transcription.py -v -rs
```

This is a deterministic renderer with manual layout choices. It does not judge clip quality, infer facecam positions, generate hooks, track subjects, or publish. VAD and confidence filtering reduce hallucinations but cannot prove every caption is correct. Transcript caches and finals need manual disk-space management.

## License

To be determined before public release.
