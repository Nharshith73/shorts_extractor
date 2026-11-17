# StreamClip AI — Evaluation Dataset Format Specification

## 1. Overview

The StreamClip AI Evaluation Dataset provides ground-truth gameplay annotations to validate candidate highlight extraction, scoring consistency, and AI judging accuracy.

The dataset is local-first, version-controlled, and manifest-driven. Ground-truth moments are stored in `data/evaluation/manifest.json` with an automatically synchronized companion spreadsheet at `data/evaluation/manifest.csv`.

```
data/evaluation/
├── manifest.json            # Master JSON evaluation manifest with revision history
├── manifest.csv             # Companion tabular export for human spreadsheet review
├── baseline/                # Baseline evaluation reports from detector runs
│   ├── evaluation_report.json
│   ├── evaluation_comparison.csv
│   └── evaluation_summary.md
└── previews/                # Optional lightweight MP4 preview snippets for verification
```

---

## 2. Manifest Schema (`manifest.json`)

The master manifest contains top-level metadata and a dictionary of labeled samples keyed by unique `sample_id`:

```json
{
  "schema_version": 1,
  "updated_at": "2026-11-17T15:32:58.674290+00:00",
  "total_samples": 5,
  "human_samples": 0,
  "synthetic_samples": 5,
  "samples": {
    "eval_source_v_b7ded4a3bd53cd63a673": {
      "sample_id": "eval_source_v_b7ded4a3bd53cd63a673",
      "source_fingerprint": {
        "path": "D:\\clipsgta\\shorts_extractor\\data\\synthetic_vod\\source_vod.mp4",
        "size": 44081259,
        "mtime_ns": 1794928598179554000
      },
      "source_id": "source_vod",
      "start_seconds": 0.0,
      "end_seconds": 12.0,
      "event_category": "chaotic",
      "editorial_rating": 4,
      "context_completeness": true,
      "setup_presence": true,
      "payoff_presence": true,
      "approved": "y",
      "rejection_reason": "",
      "notes": "Opening high-action sequence",
      "annotation_version": 1,
      "is_synthetic": true,
      "created_at": "2026-11-17T15:32:28.291264+00:00",
      "updated_at": "2026-11-17T15:32:28.291277+00:00",
      "history": []
    }
  }
}
```

### 2.1 Sample Fields

| Field | Type | Description |
| :--- | :--- | :--- |
| `sample_id` | `string` | Deterministic unique ID (`eval_<source_id>_<sha256>`). |
| `source_fingerprint` | `object` | Source media provenance (`path`, `size`, `mtime_ns`). |
| `source_id` | `string` | Base identifier of the source gameplay file. |
| `start_seconds` | `float` | Start time of the highlight interval in seconds. |
| `end_seconds` | `float` | End time of the highlight interval in seconds. |
| `event_category` | `string` | Primary gameplay event taxonomy (see Section 3). |
| `editorial_rating` | `integer` | Editorial rating from 0 to 5 (see Section 4). |
| `context_completeness`| `boolean`| `true` if clip contains necessary setup and context. |
| `setup_presence` | `boolean` | `true` if setup / anticipation is visible. |
| `payoff_presence` | `boolean` | `true` if clear climax / punchline / kill occurs. |
| `approved` | `string` | `"y"` (approved highlight), `"n"` (rejected), or `""`. |
| `rejection_reason` | `string` | Standardized reason if `approved == "n"`. |
| `notes` | `string` | Freeform reviewer comments and observations. |
| `annotation_version` | `integer` | Monotonically increasing revision counter. |
| `is_synthetic` | `boolean` | `true` for test/synthetic labels; `false` for authentic human gameplay annotations. |
| `history` | `list[object]` | Archived revisions preserving prior reviews (see Section 5). |

---

## 3. Standard Event Categories

Annotations use an 11-category taxonomy tailored to GTA gameplay and creator streaming:

| Category | Description | Typical GTA Examples |
| :--- | :--- | :--- |
| `funny` | Humorous moment, banter, glitch, or unexpected comedic dialogue. | Streamer laughing hysterically, ragdoll physics glitch, funny NPC dialogue. |
| `unexpected` | Abrupt plot twist, surprise ambush, or bizarre occurrence. | Random plane crashing into streamer, sudden cop ambush during heist. |
| `impressive` | High skill execution, vehicle stunt, or surgical sniper shot. | Stunt jump through a billboard, landing helicopter on knife edge. |
| `dramatic` | High stakes tension, narrow escape, or clutch getaway. | Escaping 5-star wanted level with smoking engine and 1% health. |
| `chaotic` | Rapid-fire explosions, multi-car pileups, or all-out shootout. | Chain reaction RPG explosions, chaotic intersection firefight. |
| `skillful` | Precise tactical gameplay, recoil control, or macro execution. | 1v4 team wipe in GTA Online adversarial mode. |
| `interaction` | Memorable dialogue or banter between streamer and player/chat. | Roleplay negotiation, hilarious hostage interaction. |
| `fail` | Comedic or disastrous failure, missed jump, or self-inflicted wipe. | Jumping off roof and missing parachute, accidentally blowing up own getaway car. |
| `story_moment` | Narrative progression, mission completion, or cinematic sequence. | Major heist finale completion, dramatic storyline reveal. |
| `ordinary` | Routine driving, navigation, inventory sorting, or dead air. | Straight-line highway driving with silence, browsing clothing store menu. |
| `unknown` | Ambiguous or unclassified moment requiring secondary inspection. | Undefined audio spike with dark screen or off-camera noise. |

---

## 4. Editorial Rating Rubric (0 to 5 Scale)

The `editorial_rating` measures standalone publication value for vertical Shorts/TikTok content:

| Rating | Tier | Editorial Definition | Publication Action |
| :---: | :--- | :--- | :--- |
| **5** | **Viral / Must-Post** | Exceptional moment with instant hook (<2s), escalating tension, clear payoff, and high visual/audio energy. | Prioritize for immediate vertical render and publishing. |
| **4** | **Strong Highlight** | Solid entertaining moment with clear setup and payoff. Good audio reaction or high gameplay drama. | Approved for Shorts queue. |
| **3** | **Decent Clip** | Entertaining moment, but setup is slightly long or payoff is muted. Good secondary filler content. | Approved if inventory is low; candidate for tighter boundary trimming. |
| **2** | **Marginal Moment** | Mildly interesting event, but lacks a decisive punchline, hook, or payoff. High drop-off risk on mobile feeds. | Rejected (`approved = "n"`). Reason: `"Weak payoff"` or `"Drab hook"`. |
| **1** | **Sub-par / Dull** | High acoustic or visual signal triggered detector, but editorial content is routine or uneventful. | Rejected (`approved = "n"`). Reason: `"Low intensity transit"`. |
| **0** | **Total Dead Air / Junk**| False trigger caused by mic bump, static screen, menu browsing, loading screen, or pause. | Rejected (`approved = "n"`). Reason: `"Dead air"` or `"Menu navigation"`. |

---

## 5. Revision Tracking and History Preservation

When an operator updates an existing annotation (e.g. re-evaluating boundaries or changing a decision), the previous state is automatically archived into the `history` array:

```json
"history": [
  {
    "version": 1,
    "archived_at": "2026-11-17T15:30:10.123456+00:00",
    "author": "initial_reviewer",
    "event_category": "chaotic",
    "editorial_rating": 2,
    "approved": "n",
    "rejection_reason": "Weak setup",
    "notes": "Felt too abrupt at start",
    "start_seconds": 5.0,
    "end_seconds": 15.0
  }
]
```

This guarantees full auditability, prevents reviewer overwrites from erasing historical consensus, and enables calculating human agreement reliability.

---

## 6. Source Fingerprint Integrity

Each annotation captures the source video's fingerprint:
- `path`: Normalized path at time of labeling.
- `size`: Exact byte size of the media file.
- `mtime_ns`: Modification timestamp.

When evaluating predictions or loading annotations, the system compares file size and filename against the target VOD. If a video file was modified, replaced, or truncated, the fingerprint verification prevents invalid label matching against mismatched footage.

---

## 7. Synthetic vs. Authentic Dataset Separation

- Every record contains the `is_synthetic: bool` flag.
- Synthetic datasets (`data/evaluation/manifest.json` generated for test harnesses) are tracked with `is_synthetic = true`.
- When authentic human gameplay footage is ingested, records must be authored with `is_synthetic = false`.
- Evaluation reports explicitly banner whether the underlying ground truth was authentic human gameplay or synthetic validation data.
