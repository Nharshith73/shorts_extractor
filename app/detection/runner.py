"""Restartable manifest follower for Stage 2 feature extraction and scoring."""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from .config import DetectionConfig
from .features import extract_segment_features
from .scoring import detect_events
from .store import DetectionStore

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DetectionRunStats:
    """Work performed during one pass through currently pending segments."""

    segments_analyzed: int
    footage_seconds: float
    processing_seconds: float

    @property
    def realtime_factor(self) -> float:
        if self.processing_seconds <= 0:
            return float("inf")
        return self.footage_seconds / self.processing_seconds


class DetectionRunner:
    """Follow complete Stage 1 segments without mutating their retention state."""

    def __init__(
        self,
        manifest_path: Path,
        config: DetectionConfig,
        *,
        ffmpeg_binary: str = "ffmpeg",
    ) -> None:
        self.config = config
        self.ffmpeg_binary = ffmpeg_binary
        self.store = DetectionStore(manifest_path)

    def _require_ffmpeg(self) -> None:
        if shutil.which(self.ffmpeg_binary) is None and not Path(self.ffmpeg_binary).is_file():
            raise RuntimeError(f"ffmpeg was not found: {self.ffmpeg_binary}")

    def recover_dirty_sessions(self) -> int:
        """Find and recompute derived events for any sessions marked dirty."""
        recovered = 0
        for source_id, session_id, revision in self.store.dirty_sessions():
            try:
                session_features = self.store.session_features(source_id, session_id)
                _, events = detect_events(session_features, self.config)
                self.store.replace_session_events(
                    source_id, session_id, events, expected_revision=revision
                )
                recovered += 1
                LOGGER.info(
                    "Recovered dirty detection session %s/%s at revision %d (%d events)",
                    source_id,
                    session_id,
                    revision,
                    len(events),
                )
            except Exception:
                LOGGER.exception("Failed to recover dirty session %s/%s", source_id, session_id)
        return recovered

    def run_once(self) -> DetectionRunStats:
        """Analyze each currently complete, not-yet-analyzed segment in order."""
        self._require_ffmpeg()
        started = time.monotonic()
        footage_seconds = 0.0
        analyzed = 0

        # Recover any dirty sessions left from an interrupted previous run
        self.recover_dirty_sessions()

        for segment in self.store.pending_segments():
            if segment.path is None or not segment.path.is_file():
                LOGGER.warning("Pending segment is unavailable; waiting for retry: id=%s", segment.id)
                break
            try:
                features = extract_segment_features(
                    segment.path,
                    segment_id=segment.id,
                    source_id=segment.source_id,
                    session_id=segment.session_id,
                    wall_start_utc=segment.wall_start_utc,
                    duration_seconds=segment.duration,
                    config=self.config,
                    ffmpeg_binary=self.ffmpeg_binary,
                )
            except Exception:
                LOGGER.exception("Detection failed for segment id=%s; preserving it for retry", segment.id)
                break
            revision = self.store.save_features_and_mark_analyzed(segment, features)
            session_features = self.store.session_features(segment.source_id, segment.session_id)
            _, events = detect_events(session_features, self.config)
            self.store.replace_session_events(
                segment.source_id, segment.session_id, events, expected_revision=revision
            )
            footage_seconds += segment.duration
            analyzed += 1

        # Final check if any sessions were dirtied by concurrent arrivals
        self.recover_dirty_sessions()
        stats = DetectionRunStats(
            segments_analyzed=analyzed,
            footage_seconds=footage_seconds,
            processing_seconds=time.monotonic() - started,
        )
        if analyzed:
            LOGGER.info(
                "Detection analyzed %d segments / %.1fs footage in %.2fs (%.2fx real time)",
                stats.segments_analyzed,
                stats.footage_seconds,
                stats.processing_seconds,
                stats.realtime_factor,
            )
        return stats

    def follow(self) -> None:
        """Poll the manifest forever so Stage 1 and Stage 2 can run concurrently."""
        while True:
            stats = self.run_once()
            if stats.segments_analyzed == 0:
                time.sleep(self.config.reprocess_poll_seconds)
