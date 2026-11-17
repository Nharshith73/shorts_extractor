"""Official YouTube Data API Integration for Live Broadcast and Metadata Inspection."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import re
from typing import Any
import urllib.parse

import httpx

LOGGER = logging.getLogger(__name__)

YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"


@dataclass(slots=True)
class YouTubeBroadcastMetadata:
    video_id: str
    is_live: bool
    live_broadcast_content: str  # "live" | "upcoming" | "none"
    title: str | None = None
    channel_id: str | None = None
    channel_title: str | None = None
    active_live_chat_id: str | None = None
    scheduled_start_time: str | None = None
    actual_start_time: str | None = None
    chat_enabled: bool = False
    api_available: bool = True
    error_message: str | None = None


def extract_video_id(url: str) -> str | None:
    """Extract YouTube video ID from watch or share URL."""
    if not url:
        return None
    # Check v= parameter
    parsed = urllib.parse.urlparse(url)
    if "youtube.com" in parsed.netloc:
        query = urllib.parse.parse_qs(parsed.query)
        if "v" in query and query["v"]:
            return query["v"][0]
    elif "youtu.be" in parsed.netloc:
        return parsed.path.lstrip("/")

    # Check bare 11-char ID regex
    match = re.search(r"^[a-zA-Z0-9_-]{11}$", url.strip())
    if match:
        return match.group(0)
    return None


def inspect_youtube_source(
    watch_url: str,
    api_key: str | None = None,
    timeout_seconds: float = 10.0,
) -> YouTubeBroadcastMetadata:
    """Inspect YouTube livestream status and active chat via official YouTube Data API v3."""
    video_id = extract_video_id(watch_url)
    if not video_id:
        return YouTubeBroadcastMetadata(
            video_id=watch_url,
            is_live=False,
            live_broadcast_content="none",
            api_available=False,
            error_message=f"Could not parse YouTube video ID from '{watch_url}'",
        )

    resolved_api_key = api_key or os.environ.get("YOUTUBE_API_KEY", "").strip()
    if not resolved_api_key:
        LOGGER.info("No YOUTUBE_API_KEY configured; returning offline metadata record")
        return YouTubeBroadcastMetadata(
            video_id=video_id,
            is_live=False,
            live_broadcast_content="unknown",
            api_available=False,
            error_message="YOUTUBE_API_KEY environment variable is not configured",
        )

    url = f"{YOUTUBE_API_BASE}/videos"
    params = {
        "part": "snippet,liveStreamingDetails,status",
        "id": video_id,
        "key": resolved_api_key,
    }

    try:
        with httpx.Client(timeout=timeout_seconds) as client:
            resp = client.get(url, params=params)
            if resp.status_code != 200:
                return YouTubeBroadcastMetadata(
                    video_id=video_id,
                    is_live=False,
                    live_broadcast_content="unknown",
                    api_available=True,
                    error_message=f"YouTube API returned HTTP {resp.status_code}: {resp.text[:200]}",
                )

            data = resp.json()
            items = data.get("items", [])
            if not items:
                return YouTubeBroadcastMetadata(
                    video_id=video_id,
                    is_live=False,
                    live_broadcast_content="none",
                    api_available=True,
                    error_message=f"Video ID '{video_id}' not found on YouTube",
                )

            item = items[0]
            snippet = item.get("snippet", {})
            streaming = item.get("liveStreamingDetails", {})

            content_type = snippet.get("liveBroadcastContent", "none")
            is_live = content_type == "live"
            active_chat_id = streaming.get("activeLiveChatId")

            return YouTubeBroadcastMetadata(
                video_id=video_id,
                is_live=is_live,
                live_broadcast_content=content_type,
                title=snippet.get("title"),
                channel_id=snippet.get("channelId"),
                channel_title=snippet.get("channelTitle"),
                active_live_chat_id=active_chat_id,
                scheduled_start_time=streaming.get("scheduledStartTime"),
                actual_start_time=streaming.get("actualStartTime"),
                chat_enabled=bool(active_chat_id),
                api_available=True,
            )

    except Exception as exc:
        LOGGER.warning("Failed to query YouTube API for video %s: %s", video_id, exc)
        return YouTubeBroadcastMetadata(
            video_id=video_id,
            is_live=False,
            live_broadcast_content="unknown",
            api_available=False,
            error_message=str(exc),
        )
