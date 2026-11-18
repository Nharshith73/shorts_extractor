"""YouTube Shorts Uploader using resumable MediaFileUpload."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any, Callable

from googleapiclient.http import MediaFileUpload

from .client import get_youtube_service

LOGGER = logging.getLogger(__name__)


def upload_short(
    video_path: Path | str,
    title: str,
    description: str = "",
    tags: list[str] | None = None,
    privacy: str = "public",
    category_id: str = "20",  # 20 = Gaming, 24 = Entertainment
    progress_callback: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Upload a vertical Short to YouTube with resumable multi-megabyte chunks."""
    video_path = Path(video_path).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found for upload: {video_path}")

    youtube = get_youtube_service()
    if not youtube:
        raise PermissionError("YouTube account is not connected. Authenticate first.")

    # YouTube Shorts requirements:
    # Must have #Shorts in title or description to be auto-classified as a vertical Short.
    clean_title = title.strip()[:100]
    if "#Shorts" not in clean_title and "#shorts" not in clean_title:
        # If room, add tag to title
        if len(clean_title) <= 90:
            clean_title = f"{clean_title} #Shorts"

    desc_lines = [description.strip()] if description.strip() else []
    if not any("#shorts" in line.lower() for line in desc_lines):
        desc_lines.append("\n#Shorts #YouTubeShorts #Trending")
    full_description = "\n".join(desc_lines).strip()

    if tags is None:
        tags = ["Shorts", "viral", "highlights", "gaming", "clip"]

    if privacy.lower() not in ("public", "unlisted", "private"):
        privacy = "public"

    body = {
        "snippet": {
            "title": clean_title,
            "description": full_description,
            "tags": tags,
            "categoryId": category_id,
        },
        "status": {
            "privacyStatus": privacy.lower(),
            "selfDeclaredMadeForKids": False,
        },
    }

    LOGGER.info("Starting YouTube upload for %s: title=%r, privacy=%s", video_path.name, clean_title, privacy)

    media = MediaFileUpload(
        str(video_path),
        mimetype="video/mp4",
        resumable=True,
        chunksize=1024 * 1024 * 5,  # 5MB chunks
    )

    request = youtube.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media,
    )

    response = None
    last_reported_pct = -1

    while response is None:
        status, response = request.next_chunk()
        if status:
            pct = int(status.progress() * 100)
            if pct != last_reported_pct:
                last_reported_pct = pct
                LOGGER.debug("Upload progress %s: %d%%", video_path.name, pct)
                if progress_callback:
                    try:
                        progress_callback(pct)
                    except Exception:
                        pass

    if progress_callback:
        try:
            progress_callback(100)
        except Exception:
            pass

    video_id = response.get("id")
    shorts_url = f"https://www.youtube.com/shorts/{video_id}"
    watch_url = f"https://www.youtube.com/watch?v={video_id}"

    # Auto-upload custom designed thumbnail if present and channel verified
    thumb_candidates = [
        video_path.with_name(f"{video_path.stem}_thumb.jpg"),
        video_path.with_name(f"{video_path.stem}_thumb.png"),
        video_path.with_suffix(".jpg"),
    ]
    for t_cand in thumb_candidates:
        if t_cand.is_file():
            try:
                youtube.thumbnails().set(
                    videoId=video_id,
                    media_body=MediaFileUpload(str(t_cand), mimetype="image/jpeg"),
                ).execute()
                LOGGER.info("Uploaded custom Short thumbnail for %s", video_id)
                break
            except Exception as exc:
                LOGGER.debug("Custom thumbnail set skipped (channel may require advanced verification): %s", exc)
                break

    LOGGER.info("Successfully uploaded Short: id=%s url=%s", video_id, shorts_url)

    return {
        "status": "COMPLETED",
        "video_id": video_id,
        "title": clean_title,
        "description": full_description,
        "privacy": privacy,
        "shorts_url": shorts_url,
        "watch_url": watch_url,
        "file_name": video_path.name,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "youtube_response": response,
    }
