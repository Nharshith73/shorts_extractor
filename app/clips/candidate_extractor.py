"""Follow Stage 2 events and export frame-accurate source-orientation clips."""

from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import subprocess
import tempfile
from typing import Iterator, Sequence

from app.ingestion.rolling_buffer import RollingBuffer, SegmentRecord, to_utc_text, utc_now
from app.ingestion.stream_reader import probe_duration
from .config import CandidateConfig
from .store import Candidate, CandidateStore
from .windows import Window, availability, event_window, merge_windows, trim_gaps

LOGGER = logging.getLogger(__name__)


@contextmanager
def worker_lock(path: Path) -> Iterator[None]:
    """One worker per manifest; OS releases the lock when a process crashes."""
    with path.open("a+b") as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if __import__("os").name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("another candidate worker is running for this manifest") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if __import__("os").name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def padded_segments(window: Window, records: Sequence[SegmentRecord]) -> list[SegmentRecord]:
    session = sorted((record for record in records if record.status == "complete"
                      and record.source_id == window.primary.source_id and record.session_id == window.primary.session_id),
                     key=lambda record: record.wall_start_utc)
    indices = [index for index, record in enumerate(session)
               if record.wall_start_utc < window.end and record.wall_end_utc > window.start]
    return session[max(0, indices[0] - 1):indices[-1] + 2] if indices else []


class CandidateExtractor:
    def __init__(self, segments_dir: Path, config: CandidateConfig, retention_seconds: float = 900,
                 *, ffmpeg_binary: str = "ffmpeg", ffprobe_binary: str = "ffprobe") -> None:
        config.validate()
        self.config = config
        self.buffer = RollingBuffer(segments_dir, retention_seconds)
        self.store = CandidateStore(self.buffer.manifest.database_path)
        self.ffmpeg = ffmpeg_binary
        self.ffprobe = ffprobe_binary
        config.output_dir.mkdir(parents=True, exist_ok=True)

    def _trim(self, window: Window, records: Sequence[SegmentRecord]) -> Window:
        boundaries = self.buffer.get_discontinuities(window.start, window.end)
        return trim_gaps(window, boundaries, records)

    def run_once(self, now: datetime | None = None) -> list[Candidate]:
        current = now or utc_now()
        with worker_lock(self.buffer.segments_dir / ".candidates.lock"):
            records = self.buffer.manifest.all_records()
            for window in merge_windows([event_window(event, self.config) for event in self.store.new_events()], self.config):
                for existing in self.store.all():
                    if existing.status != "waiting":
                        continue
                    merged = merge_windows([existing.window, window], self.config)
                    if len(merged) == 1:
                        window = self._trim(merged[0], records)
                        self.store.extend(existing.id, window, padded_segments(window, records))
                        break
                else:
                    window = self._trim(window, records)
                    rejected = (window.end - window.start).total_seconds() < self.config.min_seconds
                    self.store.create(window, padded_segments(window, records), current,
                                      "rejected_gap" if rejected else "waiting")
            for candidate in self.store.all():
                if candidate.status in ("waiting", "extracting"):
                    self._process(candidate, current)
        return self.store.all()

    def _process(self, candidate: Candidate, now: datetime) -> None:
        records = self.buffer.manifest.all_records()
        window = self._trim(candidate.window, records)
        if (window.end - window.start).total_seconds() < self.config.min_seconds:
            self.store.update(candidate.id, "rejected_gap", window=window, error="gap leaves less than min_seconds")
            return
        self.store.reserve(candidate.id, padded_segments(window, records))
        config = replace(self.config, grace_seconds=0) if window.has_gap else self.config
        state = availability(window, records, config, candidate.created, now, self.buffer.retention_seconds)
        if state != "ready":
            self.store.update(candidate.id, state, window=window,
                              error=None if state == "waiting" else f"source coverage {state}")
            return
        self.store.update(candidate.id, "extracting", window=window)
        try:
            output = self._export(candidate.id, window)
            self.store.update(candidate.id, "extracted", window=window, clip_path=str(output.resolve()))
            LOGGER.info("Extracted candidate %s to %s", candidate.id, output)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            LOGGER.exception("Candidate %s export failed", candidate.id)
            self.store.update(candidate.id, "failed", window=window, error=str(exc))

    def _export(self, candidate_id: str, window: Window) -> Path:
        records = self.buffer.get_segments(window.start, window.end,
                                           source_id=window.primary.source_id, session_id=window.primary.session_id)
        if not records:
            raise ValueError("no complete segments cover candidate")
        output = self.config.output_dir / f"{candidate_id}.mp4"
        with tempfile.TemporaryDirectory(prefix=f"candidate-{candidate_id}-", dir=self.config.output_dir) as temporary:
            joined = Path(temporary) / "joined.ts"
            rendered = Path(temporary) / "clip.mp4"
            self.buffer.concatenate(window.start, window.end, joined, ffmpeg_binary=self.ffmpeg,
                                    source_id=window.primary.source_id, session_id=window.primary.session_id)
            offset = (window.start - records[0].wall_start_utc).total_seconds()
            duration = (window.end - window.start).total_seconds()
            codec = ["-c:v", self.config.video_encoder]
            codec += ["-preset", "veryfast", "-crf", "18"] if self.config.video_encoder == "libx264" else ["-preset", "p4", "-cq", "18"]
            subprocess.run([self.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(joined),
                            "-ss", f"{offset:.6f}", "-t", f"{duration:.6f}", "-map", "0:v:0", "-map", "0:a:0?",
                            *codec, "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart", str(rendered)],
                           check=True, capture_output=True, timeout=300)
            actual = probe_duration(rendered, self.ffprobe)
            if abs(actual - duration) > 0.15:
                raise ValueError(f"export duration {actual:.3f}s differs from requested {duration:.3f}s")
            metadata = dict(candidate_id=candidate_id, source_id=window.primary.source_id,
                            session_id=window.primary.session_id, start_wall_utc=to_utc_text(window.start),
                            end_wall_utc=to_utc_text(window.end), has_gap=window.has_gap,
                            segment_ids=[record.id for record in records],
                            events=json.loads(self.store._snapshot(window)), clip_path=str(output.resolve()))
            sidecar = Path(temporary) / "clip.json"
            sidecar.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            rendered.replace(output)
            sidecar.replace(output.with_suffix(".json"))
        return output
