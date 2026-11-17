"""Manifest-based evaluation dataset for ground-truth gameplay annotations."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from .models import AnnotationRecord, utc_iso_now

LOGGER = logging.getLogger(__name__)

DEFAULT_MANIFEST_PATH = Path("data/evaluation/manifest.json")


def compute_source_fingerprint(video: Path) -> dict[str, Any]:
    """Extract standard source fingerprint matching StreamClip VOD conventions."""
    resolved = video.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def generate_sample_id(source_id: str, start_seconds: float, end_seconds: float) -> str:
    """Generate a stable deterministic sample ID for a time window in a source."""
    raw = f"{source_id}:{start_seconds:.6f}:{end_seconds:.6f}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]
    return f"eval_{source_id[:8]}_{digest}"


class EvaluationDataset:
    """Persistent manifest-based dataset for human and synthetic ground-truth moments."""

    def __init__(self, manifest_path: Path = DEFAULT_MANIFEST_PATH) -> None:
        self.manifest_path = manifest_path.resolve()
        self._records: dict[str, AnnotationRecord] = {}
        if self.manifest_path.is_file():
            self.load()

    @property
    def records(self) -> list[AnnotationRecord]:
        return list(self._records.values())

    def __len__(self) -> int:
        return len(self._records)

    def load(self) -> None:
        """Load manifest from JSON."""
        try:
            content = self.manifest_path.read_text(encoding="utf-8")
            data = json.loads(content) if content.strip() else {}
        except Exception as exc:
            LOGGER.error("Failed to parse evaluation manifest %s: %s", self.manifest_path, exc)
            raise

        records_data = data.get("samples", {})
        self._records = {
            sample_id: AnnotationRecord.from_dict(sample_dict)
            for sample_id, sample_dict in records_data.items()
        }
        LOGGER.info("Loaded %d annotations from %s", len(self._records), self.manifest_path)

    def save(self) -> None:
        """Persist dataset atomically to JSON and companion CSV."""
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_data = {
            "schema_version": 1,
            "updated_at": utc_iso_now(),
            "total_samples": len(self._records),
            "human_samples": sum(1 for r in self._records.values() if not r.is_synthetic),
            "synthetic_samples": sum(1 for r in self._records.values() if r.is_synthetic),
            "samples": {k: v.to_dict() for k, v in sorted(self._records.items())},
        }
        temp_file = self.manifest_path.with_suffix(".tmp.json")
        temp_file.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
        temp_file.replace(self.manifest_path)

        # Companion CSV export for easy spreadsheet viewing
        csv_path = self.manifest_path.with_suffix(".csv")
        self.export_csv(csv_path)

    def export_csv(self, path: Path) -> None:
        """Export manifest records as CSV."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "sample_id",
            "source_id",
            "start_seconds",
            "end_seconds",
            "duration_seconds",
            "event_category",
            "editorial_rating",
            "context_completeness",
            "setup_presence",
            "payoff_presence",
            "approved",
            "rejection_reason",
            "notes",
            "is_synthetic",
            "annotation_version",
            "created_at",
            "updated_at",
        ]
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for r in self._records.values():
                writer.writerow({
                    "sample_id": r.sample_id,
                    "source_id": r.source_id,
                    "start_seconds": r.start_seconds,
                    "end_seconds": r.end_seconds,
                    "duration_seconds": r.duration_seconds,
                    "event_category": r.event_category,
                    "editorial_rating": r.editorial_rating,
                    "context_completeness": "y" if r.context_completeness else "n",
                    "setup_presence": "y" if r.setup_presence else "n",
                    "payoff_presence": "y" if r.payoff_presence else "n",
                    "approved": r.approved,
                    "rejection_reason": r.rejection_reason,
                    "notes": r.notes,
                    "is_synthetic": "y" if r.is_synthetic else "n",
                    "annotation_version": r.annotation_version,
                    "created_at": r.created_at,
                    "updated_at": r.updated_at,
                })

    def get(self, sample_id: str) -> AnnotationRecord | None:
        return self._records.get(sample_id)

    def add_or_update(
        self,
        record: AnnotationRecord,
        *,
        author: str = "operator",
        source_duration: float | None = None,
    ) -> AnnotationRecord:
        """Add a new annotation or update an existing one while preserving revision history."""
        record.validate(video_duration=source_duration)

        existing = self._records.get(record.sample_id)
        if existing is not None:
            # Preserve history of previous revision
            history_entry = {
                "version": existing.annotation_version,
                "archived_at": utc_iso_now(),
                "author": author,
                "event_category": existing.event_category,
                "editorial_rating": existing.editorial_rating,
                "approved": existing.approved,
                "rejection_reason": existing.rejection_reason,
                "notes": existing.notes,
                "start_seconds": existing.start_seconds,
                "end_seconds": existing.end_seconds,
            }
            record.history = list(existing.history) + [history_entry]
            record.annotation_version = existing.annotation_version + 1
            record.created_at = existing.created_at
            record.updated_at = utc_iso_now()

        self._records[record.sample_id] = record
        self.save()
        return record

    def filter_by_source(
        self,
        source_id: str | None = None,
        source_fingerprint: dict[str, Any] | None = None,
    ) -> list[AnnotationRecord]:
        """Return annotations matching source, enforcing fingerprint integrity when supplied."""
        matches = []
        for r in self._records.values():
            if source_id is not None and r.source_id != source_id:
                continue
            if source_fingerprint is not None:
                # Compare fingerprint size and filename to prevent accidental reuse against different files
                rfp = r.source_fingerprint
                if rfp.get("size") != source_fingerprint.get("size"):
                    continue
                r_name = Path(str(rfp.get("path", ""))).name
                s_name = Path(str(source_fingerprint.get("path", ""))).name
                if r_name and s_name and r_name != s_name:
                    continue
            matches.append(r)
        return matches

    def human_records(self) -> list[AnnotationRecord]:
        return [r for r in self._records.values() if not r.is_synthetic]

    def synthetic_records(self) -> list[AnnotationRecord]:
        return [r for r in self._records.values() if r.is_synthetic]
