"""Persistent SQLite Queue for Live Candidate AI Judgments."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
import json
import logging
from pathlib import Path
import sqlite3
import uuid
from typing import Any, Iterator

from app.ingestion.rolling_buffer import utc_now

LOGGER = logging.getLogger(__name__)


class LiveJobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    RETRY_WAIT = "RETRY_WAIT"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass(slots=True)
class LiveJobRecord:
    job_id: str
    source_id: str
    session_id: str
    candidate_id: str
    context_path: str
    model_fingerprint: str
    priority: float
    status: str
    attempts: int
    max_attempts: int
    created_at: str
    updated_at: str
    started_at: str | None = None
    completed_at: str | None = None
    error_details: str | None = None
    judgment_json: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LiveJudgmentQueue:
    """Thread-safe SQLite persistent job queue for AI Highlight Judgments."""

    def __init__(self, db_path: Path, max_queue_depth: int = 50, max_attempts: int = 3) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_queue_depth = max_queue_depth
        self.max_attempts = max_attempts
        self._init_schema()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=15.0)
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
                CREATE TABLE IF NOT EXISTS live_ai_jobs (
                    job_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    candidate_id TEXT NOT NULL UNIQUE,
                    context_path TEXT NOT NULL,
                    model_fingerprint TEXT NOT NULL,
                    priority REAL NOT NULL DEFAULT 0.0,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL DEFAULT 3,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    error_details TEXT,
                    judgment_json TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_live_ai_jobs_status_priority
                ON live_ai_jobs (status, priority DESC, created_at ASC)
                """
            )

    def enqueue(
        self,
        candidate_id: str,
        source_id: str,
        session_id: str,
        context_path: Path | str,
        priority: float = 0.0,
        model_fingerprint: str = "qwen2.5-vl-7b-q4_k_m",
    ) -> tuple[LiveJobRecord, bool]:
        """Idempotently enqueue a candidate for AI judgment.
        
        Returns (job_record, was_inserted).
        """
        now_iso = utc_now().isoformat()
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        str_ctx_path = str(Path(context_path).resolve())

        with self.connect() as conn:
            # Check if already enqueued
            row = conn.execute(
                "SELECT * FROM live_ai_jobs WHERE candidate_id = ?",
                (candidate_id,),
            ).fetchone()
            if row:
                LOGGER.info("Candidate %s already in AI queue (status: %s)", candidate_id, row["status"])
                return LiveJobRecord(**dict(row)), False

            # Check queue depth for backpressure warning
            pending_count = conn.execute(
                "SELECT COUNT(*) FROM live_ai_jobs WHERE status IN ('PENDING', 'RETRY_WAIT')"
            ).fetchone()[0]

            if pending_count >= self.max_queue_depth:
                LOGGER.warning(
                    "AI Judgment Queue backpressure: pending=%d exceeds max_depth=%d. Job enqueued with priority=%.2f",
                    pending_count,
                    self.max_queue_depth,
                    priority,
                )

            conn.execute(
                """
                INSERT INTO live_ai_jobs (
                    job_id, source_id, session_id, candidate_id, context_path,
                    model_fingerprint, priority, status, attempts, max_attempts,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?)
                """,
                (
                    job_id,
                    source_id,
                    session_id,
                    candidate_id,
                    str_ctx_path,
                    model_fingerprint,
                    priority,
                    LiveJobStatus.PENDING.value,
                    self.max_attempts,
                    now_iso,
                    now_iso,
                ),
            )

            record = LiveJobRecord(
                job_id=job_id,
                source_id=source_id,
                session_id=session_id,
                candidate_id=candidate_id,
                context_path=str_ctx_path,
                model_fingerprint=model_fingerprint,
                priority=priority,
                status=LiveJobStatus.PENDING.value,
                attempts=0,
                max_attempts=self.max_attempts,
                created_at=now_iso,
                updated_at=now_iso,
            )
            LOGGER.info("Enqueued candidate %s into AI queue (job_id: %s, priority: %.2f)", candidate_id, job_id, priority)
            return record, True

    def claim_next(self, worker_id: str = "worker_1") -> LiveJobRecord | None:
        """Atomically claim the highest priority pending or retryable job."""
        now_iso = utc_now().isoformat()
        with self.connect() as conn:
            # Find next eligible job
            row = conn.execute(
                """
                SELECT * FROM live_ai_jobs
                WHERE status IN ('PENDING', 'RETRY_WAIT')
                ORDER BY priority DESC, created_at ASC
                LIMIT 1
                """
            ).fetchone()

            if not row:
                return None

            job_id = row["job_id"]
            new_attempts = row["attempts"] + 1

            # Atomically claim
            conn.execute(
                """
                UPDATE live_ai_jobs
                SET status = ?, attempts = ?, started_at = ?, updated_at = ?
                WHERE job_id = ? AND status IN ('PENDING', 'RETRY_WAIT')
                """,
                (
                    LiveJobStatus.RUNNING.value,
                    new_attempts,
                    now_iso,
                    now_iso,
                    job_id,
                ),
            )

            # Re-fetch claimed record
            claimed = conn.execute("SELECT * FROM live_ai_jobs WHERE job_id = ?", (job_id,)).fetchone()
            if claimed and claimed["status"] == LiveJobStatus.RUNNING.value:
                return LiveJobRecord(**dict(claimed))
            return None

    def mark_completed(self, job_id: str, judgment_dict: dict[str, Any]) -> None:
        """Mark a job completed with serialized judgment JSON."""
        now_iso = utc_now().isoformat()
        serialized = json.dumps(judgment_dict, indent=2)
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE live_ai_jobs
                SET status = ?, completed_at = ?, updated_at = ?, judgment_json = ?, error_details = NULL
                WHERE job_id = ?
                """,
                (LiveJobStatus.COMPLETED.value, now_iso, now_iso, serialized, job_id),
            )
        LOGGER.info("Job %s marked COMPLETED in AI queue", job_id)

    def mark_failed(self, job_id: str, error_message: str, retryable: bool = True) -> None:
        """Mark a job failed or scheduled for retry."""
        now_iso = utc_now().isoformat()
        with self.connect() as conn:
            row = conn.execute("SELECT attempts, max_attempts FROM live_ai_jobs WHERE job_id = ?", (job_id,)).fetchone()
            if not row:
                return

            attempts = row["attempts"]
            max_attempts = row["max_attempts"]

            if retryable and attempts < max_attempts:
                next_status = LiveJobStatus.RETRY_WAIT.value
                LOGGER.warning("Job %s attempt %d/%d failed: %s (will retry)", job_id, attempts, max_attempts, error_message)
            else:
                next_status = LiveJobStatus.FAILED.value
                LOGGER.error("Job %s exhausted retries (%d/%d): %s", job_id, attempts, max_attempts, error_message)

            conn.execute(
                """
                UPDATE live_ai_jobs
                SET status = ?, updated_at = ?, error_details = ?
                WHERE job_id = ?
                """,
                (next_status, now_iso, error_message, job_id),
            )

    def recover_stale_jobs(self, timeout_seconds: float = 120.0) -> int:
        """Recover jobs left in RUNNING status after worker crash or timeout."""
        cutoff = (utc_now() - timedelta(seconds=timeout_seconds)).isoformat()
        now_iso = utc_now().isoformat()
        with self.connect() as conn:
            cursor = conn.execute(
                """
                UPDATE live_ai_jobs
                SET status = 'RETRY_WAIT', updated_at = ?, error_details = 'Recovered from stale RUNNING state'
                WHERE status = 'RUNNING' AND updated_at < ? AND attempts < max_attempts
                """,
                (now_iso, cutoff),
            )
            recovered = cursor.rowcount

            # Mark any that exceeded max_attempts as FAILED
            conn.execute(
                """
                UPDATE live_ai_jobs
                SET status = 'FAILED', updated_at = ?, error_details = 'Timed out and exceeded max attempts'
                WHERE status = 'RUNNING' AND updated_at < ? AND attempts >= max_attempts
                """,
                (now_iso, cutoff),
            )

        if recovered > 0:
            LOGGER.info("Recovered %d stale RUNNING jobs in AI queue", recovered)
        return recovered

    def get_stats(self) -> dict[str, int]:
        """Return counts by status."""
        with self.connect() as conn:
            rows = conn.execute("SELECT status, COUNT(*) as cnt FROM live_ai_jobs GROUP BY status").fetchall()
            stats = {s.value: 0 for s in LiveJobStatus}
            for r in rows:
                stats[r["status"]] = r["cnt"]
            return stats

    def get_queue_depth(self) -> int:
        with self.connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM live_ai_jobs WHERE status IN ('PENDING', 'RETRY_WAIT')").fetchone()
            return row[0] if row else 0

    def list_jobs(self, limit: int = 50, status: str | None = None) -> list[LiveJobRecord]:
        with self.connect() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM live_ai_jobs WHERE status = ? ORDER BY priority DESC, created_at DESC LIMIT ?",
                    (status, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM live_ai_jobs ORDER BY priority DESC, created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [LiveJobRecord(**dict(r)) for r in rows]
