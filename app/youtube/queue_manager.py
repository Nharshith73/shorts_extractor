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


def delete_uploaded_short_storage(video_path: str | Path) -> dict[str, Any]:
    """Delete rendered short MP4, intermediate work folder, and metadata to free disk storage."""
    import shutil
    vpath = Path(video_path).resolve()
    freed_bytes = 0
    deleted_items = []

    # 1. Delete the rendered Short MP4
    if vpath.is_file():
        try:
            freed_bytes += vpath.stat().st_size
            vpath.unlink(missing_ok=True)
            deleted_items.append(vpath.name)
        except Exception as exc:
            LOGGER.warning("Could not delete short video %s: %s", vpath, exc)

    # 2. Delete subtitles (.ass) if alongside
    ass_file = vpath.with_suffix(".ass")
    if ass_file.is_file():
        try:
            freed_bytes += ass_file.stat().st_size
            ass_file.unlink(missing_ok=True)
            deleted_items.append(ass_file.name)
        except Exception:
            pass

    # 3. Delete work directory (raw clip, audio wav, temporary fonts)
    stem = vpath.stem
    if vpath.parent.name == "shorts":
        work_dir = vpath.parent.parent / "work" / stem
        if work_dir.is_dir():
            try:
                for f in work_dir.rglob("*"):
                    if f.is_file():
                        freed_bytes += f.stat().st_size
                shutil.rmtree(work_dir, ignore_errors=True)
                deleted_items.append(f"work/{stem}")
            except Exception as exc:
                LOGGER.warning("Could not remove work dir %s: %s", work_dir, exc)

        # 4. Delete metadata json so it drops out of the local un-uploaded library
        meta_json = vpath.parent.parent / "metadata" / f"{stem}.json"
        if meta_json.is_file():
            try:
                freed_bytes += meta_json.stat().st_size
                meta_json.unlink(missing_ok=True)
                deleted_items.append(f"metadata/{stem}.json")
            except Exception:
                pass
    elif vpath.with_suffix(".json").is_file():
        try:
            freed_bytes += vpath.with_suffix(".json").stat().st_size
            vpath.with_suffix(".json").unlink(missing_ok=True)
        except Exception:
            pass

    freed_mb = round(freed_bytes / (1024 * 1024), 2)
    LOGGER.info("Freed %.2f MB storage after YouTube upload: %s", freed_mb, stem)
    return {
        "freed_mb": freed_mb,
        "deleted_items": deleted_items,
    }


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
                v_obj = Path(vpath)

                # Look for existing metadata JSON
                meta = {}
                for cand in [v_obj.parent.parent / "metadata" / f"{v_obj.stem}.json", v_obj.with_suffix(".json")]:
                    if cand.is_file():
                        try:
                            meta = json.loads(cand.read_text(encoding="utf-8"))
                            break
                        except Exception:
                            pass

                title = raw.get("title") or meta.get("title") or meta.get("suggested_hook")
                desc = raw.get("description") or meta.get("description")
                tags = raw.get("tags") or meta.get("tags")
                hook = raw.get("suggested_hook") or meta.get("suggested_hook")
                creator = raw.get("channel") or meta.get("channel") or meta.get("creator") or ""

                # If missing or bland, viralize with attention-grabbing system
                if not title or title.startswith(("spike_", "clip_", "rank_", "test_clip_")) or ("#Shorts" not in title and "#shorts" not in title):
                    from app.evaluation.viral_metadata import ViralMetadataGenerator
                    viral = ViralMetadataGenerator.generate(
                        dialogue=meta.get("caption_text") or desc or "",
                        creator=creator,
                        platform=meta.get("platform") or "youtube",
                        category=meta.get("event_category") or "funny",
                        spike_ratio=float(meta.get("spike_ratio") or 0.0),
                        raw_summary=desc or title or "",
                        use_ai=True,
                    )
                    title = viral.title
                    desc = viral.description
                    tags = viral.tags
                    hook = viral.suggested_hook

                privacy = raw.get("privacy", "public").lower()
                if privacy not in ("public", "unlisted", "private"):
                    privacy = "public"

                entry = {
                    "id": item_id,
                    "video_path": vpath,
                    "file_name": Path(vpath).name,
                    "title": title,
                    "suggested_hook": hook,
                    "description": desc,
                    "tags": tags or ["Shorts", "gaming", "stream", "highlights", "viral"],
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

    def enhance_item(self, item_id: str, creator: str = "") -> dict[str, Any] | None:
        """Auto-enhance title, description, and tags with viral attention-grabbing system."""
        from app.evaluation.viral_metadata import ViralMetadataGenerator
        with self._lock:
            queue = self._load_locked()
            target = None
            for q in queue:
                if q["id"] == item_id:
                    target = q
                    break
            if not target:
                return None

            vpath = Path(target["video_path"])
            meta = {}
            for cand in [vpath.parent.parent / "metadata" / f"{vpath.stem}.json", vpath.with_suffix(".json")]:
                if cand.is_file():
                    try:
                        meta = json.loads(cand.read_text(encoding="utf-8"))
                        break
                    except Exception:
                        pass

            c_name = creator or meta.get("channel") or meta.get("creator") or target.get("channel") or ""
            dialogue = meta.get("caption_text") or meta.get("summary") or target.get("description") or ""
            category = meta.get("event_category") or "funny"
            spike = float(meta.get("spike_ratio") or 0.0)

            viral = ViralMetadataGenerator.generate(
                dialogue=dialogue,
                creator=c_name,
                platform=meta.get("platform") or "youtube",
                category=category,
                spike_ratio=spike,
                raw_summary=meta.get("description") or target.get("description") or "",
                use_ai=True,
            )

            target["title"] = viral.title
            target["description"] = viral.description
            target["tags"] = viral.tags
            target["suggested_hook"] = viral.suggested_hook
            self._save_locked(queue)
            return target

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
                # Free disk space: delete uploaded short and intermediate files from storage
                cleanup_info = delete_uploaded_short_storage(next_item["video_path"])
                with self._lock:
                    queue = self._load_locked()
                    for q in queue:
                        if q["id"] == item_id:
                            q["status"] = "completed"
                            q["progress"] = 100
                            q["video_id"] = result.get("video_id")
                            q["shorts_url"] = result.get("shorts_url")
                            q["uploaded_at"] = result.get("uploaded_at")
                            q["deleted_from_storage"] = True
                            q["freed_mb"] = cleanup_info.get("freed_mb", 0.0)
                            break
                    self._save_locked(queue)
            except Exception as exc:
                LOGGER.error("Failed uploading item %s: %s", item_id, exc)
                err_msg = str(exc)
                if "uploadLimitExceeded" in err_msg:
                    err_msg = "⚠️ Daily YouTube upload limit reached (~5-10 videos/day for new channels). Your quota resets in 24h. You can immediately increase this limit by verifying your channel in YouTube Studio (Settings > Channel > Feature Eligibility > Advanced Features)."
                with self._lock:
                    queue = self._load_locked()
                    for q in queue:
                        if q["id"] == item_id:
                            q["status"] = "failed"
                            q["error"] = err_msg
                            break
                    self._save_locked(queue)

            time.sleep(1.0)

        self._current_upload_id = None
        LOGGER.info("YouTube queue worker finished.")


# Global singleton instance
GLOBAL_YOUTUBE_QUEUE = YouTubeQueueManager()
