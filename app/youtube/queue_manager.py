"""YouTube Shorts Upload Queue Manager with Background Worker."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import threading
import time
from typing import Any

from .uploader import upload_short

LOGGER = logging.getLogger(__name__)

QUEUE_FILE = Path("data/youtube_queue.json")


class YouTubeQueueManager:
    """Thread-safe upload queue manager persisted to disk."""

    def __init__(self, queue_file: Path = QUEUE_FILE) -> None:
        self.queue_file = Path(queue_file).resolve()
        self._lock = threading.Lock()
        self._worker_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._current_upload_id: str | None = None

    def _load_locked(self) -> list[dict[str, Any]]:
        if not self.queue_file.is_file():
            return []
        try:
            return json.loads(self.queue_file.read_text(encoding="utf-8"))
        except Exception as exc:
            LOGGER.warning("Corrupt youtube_queue.json, resetting: %s", exc)
            return []

    def _save_locked(self, queue: list[dict[str, Any]]) -> None:
        self.queue_file.parent.mkdir(parents=True, exist_ok=True)
        self.queue_file.write_text(json.dumps(queue, indent=2), encoding="utf-8")

    def get_queue(self) -> list[dict[str, Any]]:
        """Return snapshot of current upload queue."""
        with self._lock:
            return self._load_locked()

    def add_to_queue(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Add one or more videos to upload queue."""
        added = []
        with self._lock:
            queue = self._load_locked()
            existing_paths = {item["video_path"] for item in queue}

            for raw in items:
                vpath = str(Path(raw["video_path"]).resolve())
                if vpath in existing_paths:
                    continue

                item_id = f"ytq_{int(time.time() * 1000)}_{len(queue) + 1}"
                title = raw.get("title") or raw.get("hook") or Path(vpath).stem
                desc = raw.get("description") or f"Highlight clip from stream: {title}"
                privacy = raw.get("privacy", "public").lower()
                if privacy not in ("public", "unlisted", "private"):
                    privacy = "public"

                entry = {
                    "id": item_id,
                    "video_path": vpath,
                    "file_name": Path(vpath).name,
                    "title": title,
                    "description": desc,
                    "tags": raw.get("tags") or ["Shorts", "gaming", "stream", "highlights"],
                    "privacy": privacy,
                    "status": "pending",  # pending, uploading, completed, failed
                    "progress": 0,
                    "shorts_url": None,
                    "video_id": None,
                    "error": None,
                    "queued_at": datetime.now(timezone.utc).isoformat(),
                    "uploaded_at": None,
                }
                queue.append(entry)
                existing_paths.add(vpath)
                added.append(entry)

            self._save_locked(queue)

        LOGGER.info("Added %d items to YouTube upload queue.", len(added))
        return added

    def remove_item(self, item_id: str) -> bool:
        """Remove item from queue by ID."""
        with self._lock:
            queue = self._load_locked()
            orig_len = len(queue)
            queue = [q for q in queue if q["id"] != item_id]
            if len(queue) != orig_len:
                self._save_locked(queue)
                return True
        return False

    def update_item(self, item_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        """Update metadata (title, description, privacy, tags) for an item."""
        with self._lock:
            queue = self._load_locked()
            target = None
            for q in queue:
                if q["id"] == item_id:
                    for k in ("title", "description", "privacy", "tags"):
                        if k in updates:
                            q[k] = updates[k]
                    target = q
                    break
            if target:
                self._save_locked(queue)
            return target

    def clear_completed(self) -> int:
        """Remove all items marked completed or failed."""
        with self._lock:
            queue = self._load_locked()
            kept = [q for q in queue if q["status"] in ("pending", "uploading")]
            cleared = len(queue) - len(kept)
            self._save_locked(kept)
            return cleared

    @property
    def is_uploading(self) -> bool:
        return self._worker_thread is not None and self._worker_thread.is_alive()

    def start_upload_worker(self) -> dict[str, Any]:
        """Trigger background processing of pending queue items."""
        if self.is_uploading:
            return {"status": "ALREADY_RUNNING", "message": "Upload worker is already active"}

        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._process_queue_loop,
            name="YouTubeUploadWorker",
            daemon=True,
        )
        self._worker_thread.start()
        return {"status": "STARTED", "message": "YouTube upload worker started"}

    def _update_progress(self, item_id: str, pct: int) -> None:
        with self._lock:
            queue = self._load_locked()
            for q in queue:
                if q["id"] == item_id:
                    q["progress"] = pct
                    break
            self._save_locked(queue)

    def _process_queue_loop(self) -> None:
        """Sequential queue processing worker."""
        LOGGER.info("Starting YouTube queue worker...")
        while not self._stop_event.is_set():
            next_item = None
            with self._lock:
                queue = self._load_locked()
                for q in queue:
                    if q["status"] == "pending":
                        next_item = dict(q)
                        q["status"] = "uploading"
                        q["progress"] = 0
                        self._current_upload_id = q["id"]
                        self._save_locked(queue)
                        break

            if not next_item:
                LOGGER.info("No pending items in YouTube upload queue. Worker exiting.")
                break

            item_id = next_item["id"]
            LOGGER.info("Processing queue upload: %s (%s)", next_item["title"], item_id)

            try:
                result = upload_short(
                    video_path=next_item["video_path"],
                    title=next_item["title"],
                    description=next_item["description"],
                    tags=next_item["tags"],
                    privacy=next_item["privacy"],
                    progress_callback=lambda pct: self._update_progress(item_id, pct),
                )
                with self._lock:
                    queue = self._load_locked()
                    for q in queue:
                        if q["id"] == item_id:
                            q["status"] = "completed"
                            q["progress"] = 100
                            q["video_id"] = result.get("video_id")
                            q["shorts_url"] = result.get("shorts_url")
                            q["uploaded_at"] = result.get("uploaded_at")
                            break
                    self._save_locked(queue)
            except Exception as exc:
                LOGGER.error("Failed uploading item %s: %s", item_id, exc)
                with self._lock:
                    queue = self._load_locked()
                    for q in queue:
                        if q["id"] == item_id:
                            q["status"] = "failed"
                            q["error"] = str(exc)
                            break
                    self._save_locked(queue)

            time.sleep(1.0)

        self._current_upload_id = None
        LOGGER.info("YouTube queue worker finished.")


# Global singleton instance
GLOBAL_YOUTUBE_QUEUE = YouTubeQueueManager()
