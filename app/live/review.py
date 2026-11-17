"""Human Review, Approval, and Dual-Ranking Persistence for Live Candidates."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import sqlite3
from typing import Any, Iterator

from app.ingestion.rolling_buffer import utc_now

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class LiveReviewItem:
    candidate_id: str
    source_id: str
    session_id: str
    signal_score: float
    signal_rank: int
    ai_rank: int | None
    ai_editorial_rating: int | None
    ai_confidence: float | None
    ai_category: str | None
    ai_approved: str | None  # "y" | "n"
    ai_summary: str | None
    hook_suggestion: str | None
    recommended_layout: str | None
    human_approval: str  # "pending" | "approved" | "rejected"
    rejection_reason: str | None
    notes: str | None
    clip_path: str | None
    created_at: str
    updated_at: str


class LiveReviewStore:
    """Manages human review decisions and editorial status in SQLite."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 10000")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS live_candidate_reviews (
                    candidate_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    signal_score REAL NOT NULL,
                    signal_rank INTEGER NOT NULL,
                    ai_rank INTEGER,
                    ai_editorial_rating INTEGER,
                    ai_confidence REAL,
                    ai_category TEXT,
                    ai_approved TEXT,
                    ai_summary TEXT,
                    hook_suggestion TEXT,
                    recommended_layout TEXT,
                    human_approval TEXT NOT NULL DEFAULT 'pending',
                    rejection_reason TEXT,
                    notes TEXT,
                    clip_path TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def record_candidate(
        self,
        candidate_id: str,
        source_id: str,
        session_id: str,
        signal_score: float,
        signal_rank: int,
        clip_path: str | None = None,
        ai_record: dict[str, Any] | None = None,
        ai_rank: int | None = None,
    ) -> None:
        now_iso = utc_now().isoformat()
        with self.connect() as conn:
            existing = conn.execute(
                "SELECT human_approval, notes, rejection_reason FROM live_candidate_reviews WHERE candidate_id = ?",
                (candidate_id,),
            ).fetchone()

            human_app = existing["human_approval"] if existing else "pending"
            notes = existing["notes"] if existing else None
            rej_reason = existing["rejection_reason"] if existing else None

            ai_rating = None
            ai_conf = None
            ai_cat = None
            ai_app = None
            ai_sum = None
            hook = None
            layout = None

            if ai_record:
                ai_rating = ai_record.get("editorial_rating")
                ai_conf = ai_record.get("confidence")
                ai_cat = ai_record.get("event_category")
                ai_app = "y" if ai_rating is not None and ai_rating >= 3 else "n"
                ai_sum = ai_record.get("summary")
                hook = ai_record.get("hook_suggestion")
                layout = ai_record.get("recommended_layout")
                if not rej_reason and ai_record.get("rejection_reason"):
                    rej_reason = ai_record.get("rejection_reason")

            conn.execute(
                """
                INSERT INTO live_candidate_reviews (
                    candidate_id, source_id, session_id, signal_score, signal_rank,
                    ai_rank, ai_editorial_rating, ai_confidence, ai_category, ai_approved,
                    ai_summary, hook_suggestion, recommended_layout, human_approval,
                    rejection_reason, notes, clip_path, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(candidate_id) DO UPDATE SET
                    signal_score = excluded.signal_score,
                    signal_rank = excluded.signal_rank,
                    ai_rank = COALESCE(excluded.ai_rank, live_candidate_reviews.ai_rank),
                    ai_editorial_rating = COALESCE(excluded.ai_editorial_rating, live_candidate_reviews.ai_editorial_rating),
                    ai_confidence = COALESCE(excluded.ai_confidence, live_candidate_reviews.ai_confidence),
                    ai_category = COALESCE(excluded.ai_category, live_candidate_reviews.ai_category),
                    ai_approved = COALESCE(excluded.ai_approved, live_candidate_reviews.ai_approved),
                    ai_summary = COALESCE(excluded.ai_summary, live_candidate_reviews.ai_summary),
                    hook_suggestion = COALESCE(excluded.hook_suggestion, live_candidate_reviews.hook_suggestion),
                    recommended_layout = COALESCE(excluded.recommended_layout, live_candidate_reviews.recommended_layout),
                    clip_path = COALESCE(excluded.clip_path, live_candidate_reviews.clip_path),
                    updated_at = excluded.updated_at
                """,
                (
                    candidate_id,
                    source_id,
                    session_id,
                    signal_score,
                    signal_rank,
                    ai_rank,
                    ai_rating,
                    ai_conf,
                    ai_cat,
                    ai_app,
                    ai_sum,
                    hook,
                    layout,
                    human_app,
                    rej_reason,
                    notes,
                    clip_path,
                    now_iso,
                    now_iso,
                ),
            )

    def set_approval(
        self,
        candidate_id: str,
        status: str,  # "approved" | "rejected" | "pending"
        notes: str | None = None,
        rejection_reason: str | None = None,
    ) -> bool:
        if status not in ("approved", "rejected", "pending"):
            raise ValueError(f"Invalid approval status: {status}")

        now_iso = utc_now().isoformat()
        with self.connect() as conn:
            cursor = conn.execute(
                """
                UPDATE live_candidate_reviews
                SET human_approval = ?, notes = COALESCE(?, notes),
                    rejection_reason = COALESCE(?, rejection_reason), updated_at = ?
                WHERE candidate_id = ?
                """,
                (status, notes, rejection_reason, now_iso, candidate_id),
            )
            return cursor.rowcount > 0

    def list_candidates(self, limit: int = 50) -> list[LiveReviewItem]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM live_candidate_reviews
                ORDER BY
                    CASE WHEN ai_editorial_rating IS NOT NULL THEN ai_editorial_rating ELSE 0 END DESC,
                    signal_score DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [LiveReviewItem(**dict(r)) for r in rows]

    def get_candidate(self, candidate_id: str) -> LiveReviewItem | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM live_candidate_reviews WHERE candidate_id = ?",
                (candidate_id,),
            ).fetchone()
            return LiveReviewItem(**dict(row)) if row else None
