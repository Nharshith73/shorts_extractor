"""Interactive CSV-only review; never launches a media player."""

import csv
import logging
from pathlib import Path
from typing import Callable

from app.clips.candidate_extractor import worker_lock
from .pipeline import write_review

LOGGER = logging.getLogger(__name__)
REASONS = dict(b="boring/nothing happens", e="cut too early", l="cut too late",
               m="moment missing from window", n="noise or audio artifact", o="other")


def review(output: Path, read_input: Callable[[str], str] = input) -> None:
    path = output / "review.csv"
    with worker_lock(output / ".vod.lock"):
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        for index, row in enumerate(rows, 1):
            LOGGER.info("Clip %d/%d: %s", index, len(rows), row["clip_path"])
            LOGGER.info("  %ss .. %ss score=%s top=%s approved=%s notes=%s", row["start_seconds"],
                        row["end_seconds"], row["score"], row["top_features"], row["approved"], row["notes"])
            while True:
                try:
                    choice = read_input("y approve / n reject / s skip / q quit: ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    write_review(path, rows)
                    return
                if choice in ("y", "n", "s", "q"):
                    break
                LOGGER.info("Enter y, n, s, or q")
            if choice in ("y", "n"):
                if choice == "n":
                    while True:
                        try:
                            reason = read_input("Reason b boring / e early / l late / m missing / n noise / o other: ").strip().lower()
                        except (EOFError, KeyboardInterrupt):
                            write_review(path, rows)
                            return
                        if reason in REASONS:
                            row["reason_code"] = reason
                            break
                        LOGGER.info("Choose a reason code: %s", REASONS)
                else:
                    row["reason_code"] = ""
                row["approved"] = choice
            write_review(path, rows)
            if choice == "q":
                return


def summary(output: Path) -> dict:
    import json
    with (output / "review.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    groups: dict[str, list[dict[str, float]]] = {"y": [], "n": []}
    counts = {"y": 0, "n": 0}
    reasons: dict[str, int] = {}
    missing = 0
    for row in rows:
        approved = row.get("approved", "").lower()
        if approved not in groups:
            continue
        counts[approved] += 1
        if approved == "n":
            code = row.get("reason_code") or "unspecified"
            reasons[code] = reasons.get(code, 0) + 1
        path = Path(row["clip_path"]).with_suffix(".json")
        if path.exists():
            groups[approved].append(json.loads(path.read_text(encoding="utf-8"))["components"])
        else:
            missing += 1
    reviewed = sum(counts.values())
    averages = {}
    for label, entries in groups.items():
        averages["approved" if label == "y" else "rejected"] = {
            name: sum(entry[name] for entry in entries if name in entry) / sum(name in entry for entry in entries)
            for name in sorted({name for entry in entries for name in entry})}
    result = dict(approved=counts["y"], rejected=counts["n"], unreviewed=len(rows)-reviewed,
                  approval_rate=counts["y"] / reviewed if reviewed else 0.0,
                  reasons=reasons, feature_averages=averages, missing_sidecars=missing)
    LOGGER.info("Reviewed %d/%d; approval rate %.1f%% (%d approved, %d rejected)", reviewed, len(rows),
                100 * result["approval_rate"], counts["y"], counts["n"])
    LOGGER.info("Rejection reasons: %s", {REASONS.get(code, code): count for code, count in reasons.items()})
    LOGGER.info("Average per-feature z-scores: %s", averages)
    if missing:
        LOGGER.warning("%d reviewed clips lack sidecars; omitted from z-score averages", missing)
    return result
