"""Candidate ledger and reference-counted segment reservations in SQLite."""

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Iterator, Sequence

from app.ingestion.rolling_buffer import SegmentRecord, from_utc_text, to_utc_text
from .windows import Event, Window


@dataclass(frozen=True, slots=True)
class Candidate:
    id: str
    window: Window
    status: str
    segment_ids: tuple[int, ...]
    created: datetime
    clip_path: str | None
    error: str | None


class CandidateStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        with self.connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS candidates (
                    id TEXT PRIMARY KEY, source_id TEXT NOT NULL, session_id TEXT NOT NULL,
                    event_ids TEXT NOT NULL, start_wall_utc TEXT NOT NULL, end_wall_utc TEXT NOT NULL,
                    score REAL NOT NULL, status TEXT NOT NULL, has_gap INTEGER NOT NULL DEFAULT 0,
                    segment_ids TEXT NOT NULL, clip_path TEXT, created_utc TEXT NOT NULL, error TEXT,
                    event_snapshot TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS candidate_events (
                    source_id TEXT NOT NULL, peak_wall_utc TEXT NOT NULL, candidate_id TEXT NOT NULL,
                    PRIMARY KEY (source_id, peak_wall_utc)
                );
                CREATE TABLE IF NOT EXISTS candidate_segment_pins (
                    candidate_id TEXT NOT NULL, segment_id INTEGER NOT NULL,
                    PRIMARY KEY(candidate_id, segment_id)
                );
                CREATE TABLE IF NOT EXISTS segment_reservations (
                    segment_id INTEGER PRIMARY KEY, was_pinned INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_candidates_status ON candidates(status);
                CREATE TABLE IF NOT EXISTS candidate_selection_decisions (
                    source_id TEXT NOT NULL, peak_wall_utc TEXT NOT NULL, score REAL NOT NULL,
                    reason TEXT NOT NULL, created_utc TEXT NOT NULL,
                    PRIMARY KEY(source_id, peak_wall_utc)
                );
            """)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=15)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def new_events(self) -> list[Event]:
        with self.connect() as connection:
            if not connection.execute("SELECT 1 FROM sqlite_master WHERE name='events'").fetchone():
                return []
            rows = connection.execute("""
                SELECT e.* FROM events e LEFT JOIN candidate_events c
                ON c.source_id=e.source_id AND c.peak_wall_utc=e.peak_wall_utc
                LEFT JOIN candidate_selection_decisions d ON d.source_id=e.source_id AND d.peak_wall_utc=e.peak_wall_utc
                WHERE c.candidate_id IS NULL AND d.source_id IS NULL ORDER BY e.peak_wall_utc, e.id
            """).fetchall()
        return [Event(int(row["id"]), row["source_id"], row["session_id"],
                      from_utc_text(row["peak_wall_utc"]), from_utc_text(row["start_wall_utc"]),
                      from_utc_text(row["end_wall_utc"]), float(row["score"]),
                      tuple(json.loads(row["reason_codes_json"])) if "reason_codes_json" in row.keys() else (),
                      json.loads(row["components_json"]) if "components_json" in row.keys() else {},
                      json.loads(row["ranking_components_json"]) if "ranking_components_json" in row.keys() else {}) for row in rows]

    def record_selection_decision(self, event: Event, reason: str, now: datetime) -> None:
        with self.connect() as connection:
            connection.execute("INSERT OR IGNORE INTO candidate_selection_decisions VALUES (?,?,?,?,?)",
                               (event.source_id, to_utc_text(event.peak), event.score, reason, to_utc_text(now)))

    @staticmethod
    def _snapshot(window: Window) -> str:
        return json.dumps([dict(id=e.id, source_id=e.source_id, session_id=e.session_id,
                                peak=to_utc_text(e.peak), start=to_utc_text(e.start),
                                end=to_utc_text(e.end), score=e.score, reason_codes=e.reason_codes,
                                components=e.components, ranking_components=e.ranking_components) for e in window.events])

    @staticmethod
    def _decode(row: sqlite3.Row) -> Candidate:
        events = tuple(Event(int(e["id"]), e["source_id"], e["session_id"], from_utc_text(e["peak"]),
                             from_utc_text(e["start"]), from_utc_text(e["end"]), float(e["score"]),
                             tuple(e.get("reason_codes", ())), e.get("components", {}), e.get("ranking_components", {}))
                       for e in json.loads(row["event_snapshot"]))
        return Candidate(row["id"], Window(events, from_utc_text(row["start_wall_utc"]),
                         from_utc_text(row["end_wall_utc"]), bool(row["has_gap"])), row["status"],
                         tuple(json.loads(row["segment_ids"])), from_utc_text(row["created_utc"]),
                         row["clip_path"], row["error"])

    def all(self) -> list[Candidate]:
        with self.connect() as connection:
            return [self._decode(row) for row in connection.execute("SELECT * FROM candidates ORDER BY created_utc, id")]

    def create(self, window: Window, records: Sequence[SegmentRecord], now: datetime, status: str = "waiting") -> str:
        key = "|".join(sorted(f"{event.source_id}:{to_utc_text(event.peak)}" for event in window.events))
        candidate_id = hashlib.sha256(key.encode()).hexdigest()[:24]
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute("SELECT id FROM candidates WHERE id=?", (candidate_id,)).fetchone():
                return candidate_id
            # Stable event keys survive Stage 2 replacing its rows and IDs.
            if any(connection.execute("SELECT 1 FROM candidate_events WHERE source_id=? AND peak_wall_utc=?",
                                      (e.source_id, to_utc_text(e.peak))).fetchone() for e in window.events):
                return candidate_id
            connection.execute("""INSERT INTO candidates
                (id,source_id,session_id,event_ids,start_wall_utc,end_wall_utc,score,status,has_gap,
                 segment_ids,created_utc,event_snapshot) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (candidate_id, window.primary.source_id, window.primary.session_id,
                 json.dumps([e.id for e in window.events]), to_utc_text(window.start), to_utc_text(window.end),
                 max(e.score for e in window.events), status, int(window.has_gap), "[]", to_utc_text(now), self._snapshot(window)))
            connection.executemany("INSERT INTO candidate_events VALUES (?,?,?)",
                                   [(e.source_id, to_utc_text(e.peak), candidate_id) for e in window.events])
            if status == "waiting":
                self._reserve(connection, candidate_id, records)
        return candidate_id

    @staticmethod
    def _reserve(connection: sqlite3.Connection, candidate_id: str, records: Sequence[SegmentRecord]) -> None:
        for record in records:
            row = connection.execute("SELECT pinned,status FROM segments WHERE id=?", (record.id,)).fetchone()
            if row is None or row["status"] != "complete":
                continue
            connection.execute("INSERT OR IGNORE INTO segment_reservations VALUES (?,?)", (record.id, row["pinned"]))
            connection.execute("INSERT OR IGNORE INTO candidate_segment_pins VALUES (?,?)", (candidate_id, record.id))
            connection.execute("UPDATE segments SET pinned=1 WHERE id=?", (record.id,))
        ids = [row[0] for row in connection.execute("SELECT segment_id FROM candidate_segment_pins WHERE candidate_id=? ORDER BY segment_id", (candidate_id,))]
        connection.execute("UPDATE candidates SET segment_ids=? WHERE id=?", (json.dumps(ids), candidate_id))

    def reserve(self, candidate_id: str, records: Sequence[SegmentRecord]) -> None:
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._reserve(connection, candidate_id, records)

    def extend(self, candidate_id: str, window: Window, records: Sequence[SegmentRecord]) -> None:
        """Merge later events into a waiting candidate without changing its ID."""
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for event in window.events:
                connection.execute("INSERT OR IGNORE INTO candidate_events VALUES (?,?,?)",
                                   (event.source_id, to_utc_text(event.peak), candidate_id))
            connection.execute("""UPDATE candidates SET event_ids=?, event_snapshot=?,
                start_wall_utc=?,end_wall_utc=?,score=?,has_gap=? WHERE id=? AND status='waiting'""",
                (json.dumps([event.id for event in window.events]), self._snapshot(window),
                 to_utc_text(window.start), to_utc_text(window.end), max(event.score for event in window.events),
                 int(window.has_gap), candidate_id))
            self._reserve(connection, candidate_id, records)

    def update(self, candidate_id: str, status: str, *, window: Window | None = None,
               clip_path: str | None = None, error: str | None = None) -> None:
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("UPDATE candidates SET status=?, clip_path=?, error=? WHERE id=?",
                               (status, clip_path, error, candidate_id))
            if window:
                connection.execute("UPDATE candidates SET start_wall_utc=?,end_wall_utc=?,has_gap=? WHERE id=?",
                                   (to_utc_text(window.start), to_utc_text(window.end), int(window.has_gap), candidate_id))
            if status not in ("waiting", "extracting"):
                ids = [row[0] for row in connection.execute("SELECT segment_id FROM candidate_segment_pins WHERE candidate_id=?", (candidate_id,))]
                connection.execute("DELETE FROM candidate_segment_pins WHERE candidate_id=?", (candidate_id,))
                for segment_id in ids:
                    if not connection.execute("SELECT 1 FROM candidate_segment_pins WHERE segment_id=?", (segment_id,)).fetchone():
                        connection.execute("UPDATE segments SET pinned=COALESCE((SELECT was_pinned FROM segment_reservations WHERE segment_id=?),pinned) WHERE id=?", (segment_id, segment_id))
                        connection.execute("DELETE FROM segment_reservations WHERE segment_id=?", (segment_id,))
