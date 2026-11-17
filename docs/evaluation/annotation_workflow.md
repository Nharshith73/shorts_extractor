# StreamClip AI — Ground-Truth Annotation Workflow Guide

## 1. Overview

The operator annotation workflow allows human reviewers to establish ground-truth gameplay labels for training, baseline measurement, and AI judge validation.

The workflow provides:
- Single-command CLI labeling.
- Automated fast preview clip generation (`-c:v libx264 -preset ultrafast`).
- Input validation (interval sanity, bounds check against VOD duration).
- Atomic manifest updates with revision history tracking.
- Instant synchronization with companion CSV files.

---

## 2. CLI Usage

The annotation command is executed via `app.evaluation annotate`:

```powershell
python -m app.evaluation annotate \
  --video <path/to/vod.mp4> \
  --start <seconds> \
  --end <seconds> \
  [--category <category>] \
  [--rating <0-5>] \
  [--setup | --no-setup] \
  [--payoff | --no-payoff] \
  [--context-complete | --no-context-complete] \
  [--approved {y,n}] \
  [--rejection-reason <reason>] \
  [--notes <text>] \
  [--preview] \
  [--synthetic]
```

### 2.1 Argument Reference

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--video` | Path | *(Required)* | Path to the source gameplay MP4/MKV. |
| `--start` | Float | *(Required)* | Start time of the highlight interval in seconds. |
| `--end` | Float | *(Required)* | End time of the highlight interval in seconds. |
| `--category` | String | `unknown` | Event category (`funny`, `impressive`, `fail`, `chaotic`, etc.). |
| `--rating` | Integer | `3` | Editorial rating from 0 to 5. |
| `--setup` / `--no-setup` | Bool | `True` | Flag indicating whether context/setup is present. |
| `--payoff` / `--no-payoff` | Bool | `True` | Flag indicating whether the climax/punchline occurred. |
| `--context-complete` | Bool | `True` | Flag indicating whether the clip is standalone viable. |
| `--approved` | `y` / `n` | `y` | Editorial decision for Shorts publishing. |
| `--rejection-reason` | String | `""` | Standardized reason if rejected (e.g. `Dead air`, `Weak payoff`). |
| `--notes` | String | `""` | Detailed operator observations. |
| `--preview` | Flag | `False` | Render a lightweight MP4 snippet to `data/evaluation/previews/`. |
| `--synthetic` | Flag | `False` | Mark annotation as synthetic test data. |
| `--manifest` | Path | `data/evaluation/manifest.json` | Path to manifest file. |

---

## 3. Standard Annotation Scenarios

### Scenario A — Labeling a High-Action Stunt (Approved)

```powershell
python -m app.evaluation annotate `
  --video data/gameplay/gtav_stream_01.mp4 `
  --start 142.5 `
  --end 168.0 `
  --category impressive `
  --rating 5 `
  --setup `
  --payoff `
  --context-complete `
  --approved y `
  --notes "Insane stunt jump over highway police blockade" `
  --preview
```
*Creates a 5-star approved highlight with a fast preview clip generated in `data/evaluation/previews/`.*

---

### Scenario B — Labeling a False-Positive Signal Spike (Rejected)

```powershell
python -m app.evaluation annotate `
  --video data/gameplay/gtav_stream_01.mp4 `
  --start 310.0 `
  --end 325.0 `
  --category ordinary `
  --rating 1 `
  --no-setup `
  --no-payoff `
  --context-complete `
  --approved n `
  --rejection-reason "High audio spike from coughing during empty loading screen" `
  --notes "Microphone burst, zero visual gameplay interest"
```
*Records an editorial rejection with a specific failure mode to train false-positive suppression.*

---

### Scenario C — Updating an Existing Moment (Revision Tracking)

If a reviewer wants to tighten boundaries or upgrade a rating:

```powershell
python -m app.evaluation annotate `
  --video data/gameplay/gtav_stream_01.mp4 `
  --start 144.0 `
  --end 165.0 `
  --category impressive `
  --rating 5 `
  --approved y `
  --notes "Trimmed 1.5s dead air from intro for tighter TikTok hook"
```
*The dataset automatically increments `annotation_version` to `2` and archives the previous parameters into the `history` log.*

---

## 4. Quality Control & Validation Checks

When `annotate_interval()` executes, the following automatic validations run:
1. **Timestamp Sanity**:
   - `start_seconds >= 0.0`.
   - `end_seconds > start_seconds`.
   - `end_seconds <= source_duration + 0.25s`.
2. **Fingerprint Capture**:
   - Source file byte size and modification time are captured and pinned.
3. **Atomic Persistence**:
   - The manifest is written to a temporary file (`manifest.tmp.json`) and swapped via atomic filesystem replacement to guarantee no file corruption occurs if interrupted.
4. **CSV Export**:
   - `data/evaluation/manifest.csv` is immediately updated to ensure spreadsheet workflows remain in lockstep.
