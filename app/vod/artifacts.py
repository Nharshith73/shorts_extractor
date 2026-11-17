"""Durable raw features and local review artifacts."""

import csv
import html
import json
from pathlib import Path
from urllib.parse import quote

from app.detection.features import FEATURE_NAMES
from app.detection.models import FeatureBin
from app.detection.scoring import AuxiliaryFeatures
from .chat import CHAT_NAMES, ChatConfig
from .timeline import media_seconds, media_time

REVIEW_FIELDS = ("clip_path", "start_seconds", "end_seconds", "score", "top_features", "approved", "notes",
                 "reason_code", "window_id", "rank", "peak_seconds", "rule", "unclipped_z_scores", "unclipped_score", "reason_codes")


def write_json(path: Path, data: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, default=str, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def write_review(path: Path, rows: list[dict[str, str]]) -> None:
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_features(path: Path, features: list[FeatureBin], auxiliary: AuxiliaryFeatures | None) -> None:
    names = [*FEATURE_NAMES, *(CHAT_NAMES if auxiliary else [])]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["time_seconds", "bin_seconds", *names])
        writer.writeheader()
        for feature in features:
            writer.writerow(dict(time_seconds=media_seconds(feature.wall_utc), bin_seconds=feature.bin_seconds,
                                 **{name: getattr(feature, name) for name in FEATURE_NAMES},
                                 **(auxiliary.values.get(feature.wall_utc, {}) if auxiliary else {})))


def read_features(path: Path, source_id: str, chat_config: ChatConfig) -> tuple[list[FeatureBin], AuxiliaryFeatures | None]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        has_chat = set(CHAT_NAMES).issubset(reader.fieldnames or [])
        rows = list(reader)
    features = [FeatureBin(source_id, "vod", media_time(float(row["time_seconds"])), float(row["bin_seconds"]),
                           **{name: float(row[name]) for name in FEATURE_NAMES}) for row in rows]
    auxiliary = AuxiliaryFeatures({feature.wall_utc: {name: float(row[name]) for name in CHAT_NAMES}
                                   for feature, row in zip(features, rows)}, chat_config.weights, chat_config.scale_floors) if has_chat else None
    return features, auxiliary


def write_index(output: Path, clips: list[dict]) -> None:
    cards = []
    for clip in clips:
        path = Path(clip["clip_path"])
        src = quote(path.relative_to(output).as_posix())
        top = ", ".join(f"{name} {value:.2f}" for name, value in sorted(clip["components"].items(), key=lambda item: -item[1])[:3])
        player = f'<video controls preload="none" src="{src}"></video>' if clip["extracted"] and path.exists() else '<p>Not extracted. Run rescore --extract.</p>'
        cards.append(f'<article><h2>#{clip["rank"]} · score {clip["score"]:.2f}</h2>{player}'
                     f'<p>Source {clip["peak_seconds"]:.3f}s · window {clip["start_seconds"]:.3f}–{clip["end_seconds"]:.3f}s</p>'
                     f'<p>{html.escape(top)} · {html.escape(clip["rule"])}</p></article>')
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    page += '<title>StreamClip VOD review</title><style>body{font:16px system-ui;max-width:1000px;margin:2rem auto;padding:1rem;background:#15191f;color:#eee}article{background:#232b35;padding:1rem;margin:1rem 0;border-radius:8px}video{width:100%;max-height:65vh}p{color:#c3cdda}</style>'
    page += f'<h1>VOD review · {len(clips)} clips</h1><p>Review decisions are saved with python -m app.vod review.</p>'
    page += "".join(cards) if cards else '<p>No eligible peaks in the selected range.</p>'
    page += '</html>'
    (output / "index.html").write_text(page, encoding="utf-8")
