"""SQLite persistence for Stage 2 features, events, and analysis progress."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable

from app.ingestion.rolling_buffer import (
    SegmentManifest,
    SegmentRecord,
    from_utc_text,
    to_utc_text,
    utc_now,
)

from .models import DetectedEvent, FeatureBin


class DetectionStore:
    """Own the Stage 2 tables stored beside the Stage 1 segment manifest."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        # Ensure a pre-Stage-2 manifest receives the normal Stage 1 migration
        # before this store reads its segment fields.
        SegmentManifest(database_path)
        self._create_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def _create_schema(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS features (
                    id INTEGER PRIMARY KEY,
                    segment_id INTEGER NOT NULL,
                    source_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    wall_utc TEXT NOT NULL,
                    bin_seconds REAL NOT NULL,
                    audio_rms_db REAL NOT NULL,
                    voice_energy_db REAL NOT NULL,
                    onset_strength REAL NOT NULL,
                    motion REAL NOT NULL,
                    cut_change REAL NOT NULL,
                    UNIQUE(segment_id, wall_utc)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_features_session_time
                ON features (source_id, session_id, wall_utc)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    peak_wall_utc TEXT NOT NULL,
                    start_wall_utc TEXT NOT NULL,
                    end_wall_utc TEXT NOT NULL,
                    score REAL NOT NULL,
                    components_json TEXT NOT NULL,
                    created_utc TEXT NOT NULL,
                    UNIQUE(source_id, peak_wall_utc)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_events_session_time
                ON events (source_id, session_id, peak_wall_utc)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS analysis_state (
                    segment_id INTEGER PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    analyzed_utc TEXT NOT NULL
                )
                """
            )

    @staticmethod
    def _segment(row: sqlite3.Row) -> SegmentRecord:
        raw_path = row["path"]
        return SegmentRecord(
            id=int(row["id"]),
            path=Path(raw_path) if raw_path else None,
            wall_start_utc=from_utc_text(str(row["wall_start_utc"])),
            fs_created_utc=from_utc_text(str(row["fs_created_utc"])),
            duration=float(row["duration"]),
            status=str(row["status"]),
            source_id=str(row["source_id"]),
            pinned=bool(row["pinned"]),
            session_id=str(row["session_id"]),
            timing_reanchored=bool(row["timing_reanchored"]),
        )

    def pending_segments(self) -> list[SegmentRecord]:
        """Return complete, unanalysed segment rows in capture order."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT segments.* FROM segments
                LEFT JOIN analysis_state ON analysis_state.segment_id = segments.id
                WHERE segments.status = 'complete' AND analysis_state.segment_id IS NULL
                ORDER BY segments.wall_start_utc, segments.id
                """
            ).fetchall()
        return [self._segment(row) for row in rows]

    def save_features_and_mark_analyzed(
        self, segment: SegmentRecord, features: Iterable[FeatureBin]
    ) -> None:
        """Atomically persist all bins for a segment and mark it processed."""
        rows = [
            (
                segment.id,
                feature.source_id,
                feature.session_id,
                to_utc_text(feature.wall_utc),
                feature.bin_seconds,
                feature.audio_rms_db,
                feature.voice_energy_db,
                feature.onset_strength,
                feature.motion,
                feature.cut_change,
            )
            for feature in features
        ]
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO features
                (segment_id, source_id, session_id, wall_utc, bin_seconds, audio_rms_db,
                 voice_energy_db, onset_strength, motion, cut_change)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(segment_id, wall_utc) DO UPDATE SET
                    bin_seconds=excluded.bin_seconds,
                    audio_rms_db=excluded.audio_rms_db,
                    voice_energy_db=excluded.voice_energy_db,
                    onset_strength=excluded.onset_strength,
                    motion=excluded.motion,
                    cut_change=excluded.cut_change
                """,
                rows,
            )
            connection.execute(
                """
                INSERT INTO analysis_state (segment_id, source_id, session_id, analyzed_utc)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(segment_id) DO UPDATE SET analyzed_utc=excluded.analyzed_utc
                """,
                (segment.id, segment.source_id, segment.session_id, to_utc_text(utc_now())),
            )

    def session_features(self, source_id: str, session_id: str) -> list[FeatureBin]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM features
                WHERE source_id = ? AND session_id = ?
                ORDER BY wall_utc, id
                """,
                (source_id, session_id),
            ).fetchall()
        return [
            FeatureBin(
                segment_id=int(row["segment_id"]),
                source_id=str(row["source_id"]),
                session_id=str(row["session_id"]),
                wall_utc=from_utc_text(str(row["wall_utc"])),
                bin_seconds=float(row["bin_seconds"]),
                audio_rms_db=float(row["audio_rms_db"]),
                voice_energy_db=float(row["voice_energy_db"]),
                onset_strength=float(row["onset_strength"]),
                motion=float(row["motion"]),
                cut_change=float(row["cut_change"]),
            )
            for row in rows
        ]

    def replace_session_events(
        self, source_id: str, session_id: str, events: Iterable[DetectedEvent]
    ) -> None:
        """Refresh one session's derived events from its persisted raw features."""
        now = to_utc_text(utc_now())
        rows = [
            (
                event.source_id,
                event.session_id,
                to_utc_text(event.peak_wall_utc),
                to_utc_text(event.start_wall_utc),
                to_utc_text(event.end_wall_utc),
                event.score,
                json.dumps(event.components, sort_keys=True),
                now,
            )
            for event in events
        ]
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM events WHERE source_id = ? AND session_id = ?",
                (source_id, session_id),
            )
            connection.executemany(
                """
                INSERT INTO events
                (source_id, session_id, peak_wall_utc, start_wall_utc, end_wall_utc,
                 score, components_json, created_utc)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def events(self, source_id: str | None = None) -> list[DetectedEvent]:
        query = "SELECT * FROM events"
        arguments: tuple[object, ...] = ()
        if source_id is not None:
            query += " WHERE source_id = ?"
            arguments = (source_id,)
        query += " ORDER BY peak_wall_utc, id"
        with self._connect() as connection:
            rows = connection.execute(query, arguments).fetchall()
        return [
            DetectedEvent(
                source_id=str(row["source_id"]),
                session_id=str(row["session_id"]),
                peak_wall_utc=from_utc_text(str(row["peak_wall_utc"])),
                start_wall_utc=from_utc_text(str(row["start_wall_utc"])),
                end_wall_utc=from_utc_text(str(row["end_wall_utc"])),
                score=float(row["score"]),
                components=json.loads(str(row["components_json"])),
            )
            for row in rows
        ]
