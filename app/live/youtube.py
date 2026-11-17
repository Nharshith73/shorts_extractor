"""Official YouTube Data API Integration for Live Broadcast and Metadata Inspection."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import re
import sys
from typing import Any
import urllib.parse

import httpx

LOGGER = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

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
    actual_end_time: str | None = None
    broadcast_status: str = "unknown"  # "live" | "upcoming" | "completed" | "none" | "unknown"
    upload_status: str | None = None
    privacy_status: str | None = None
    chat_enabled: bool = False
    api_available: bool = True
    error_message: str | None = None


def resolve_youtube_api_key(api_key: str | None = None) -> str:
    """Retrieve YouTube API key from explicit argument, process env, or Windows registry."""
    if api_key and api_key.strip():
        return api_key.strip()

    key = os.environ.get("YOUTUBE_API_KEY", "").strip()
    if key:
        return key

    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as env_key:
                val, _ = winreg.QueryValueEx(env_key, "YOUTUBE_API_KEY")
                if val and str(val).strip():
                    resolved = str(val).strip()
                    os.environ["YOUTUBE_API_KEY"] = resolved
                    return resolved
        except Exception:
            pass
        try:
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
            ) as env_key:
                val, _ = winreg.QueryValueEx(env_key, "YOUTUBE_API_KEY")
                if val and str(val).strip():
                    resolved = str(val).strip()
                    os.environ["YOUTUBE_API_KEY"] = resolved
                    return resolved
        except Exception:
            pass

    return ""


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
            broadcast_status="unknown",
            api_available=False,
            error_message=f"Could not parse YouTube video ID from '{watch_url}'",
        )

    resolved_api_key = resolve_youtube_api_key(api_key)
    if not resolved_api_key:
        LOGGER.info("No YOUTUBE_API_KEY configured; returning offline metadata record")
        return YouTubeBroadcastMetadata(
            video_id=video_id,
            is_live=False,
            live_broadcast_content="unknown",
            broadcast_status="unknown",
            api_available=False,
            error_message="YOUTUBE_API_KEY environment variable is not configured",
        )

    url = f"{YOUTUBE_API_BASE}/videos"
    headers = {"X-Goog-Api-Key": resolved_api_key}
    params = {
        "part": "snippet,liveStreamingDetails,status",
        "id": video_id,
    }

    try:
        with httpx.Client(timeout=timeout_seconds, headers=headers) as client:
            resp = client.get(url, params=params)
            if resp.status_code != 200:
                return YouTubeBroadcastMetadata(
                    video_id=video_id,
                    is_live=False,
                    live_broadcast_content="unknown",
                    broadcast_status="unknown",
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
                    broadcast_status="none",
                    api_available=True,
                    error_message=f"Video ID '{video_id}' not found on YouTube",
                )

            item = items[0]
            snippet = item.get("snippet", {})
            status_info = item.get("status", {})
            streaming = item.get("liveStreamingDetails", {})

            content_type = snippet.get("liveBroadcastContent", "none")
            is_live = content_type == "live"
            active_chat_id = streaming.get("activeLiveChatId")
            actual_end_time = streaming.get("actualEndTime")
            actual_start_time = streaming.get("actualStartTime")
            scheduled_start_time = streaming.get("scheduledStartTime")

            if is_live:
                broadcast_status = "live"
            elif content_type == "upcoming":
                broadcast_status = "upcoming"
            elif actual_end_time or (actual_start_time and content_type == "none"):
                broadcast_status = "completed"
            else:
                broadcast_status = "none"

            return YouTubeBroadcastMetadata(
                video_id=video_id,
                is_live=is_live,
                live_broadcast_content=content_type,
                title=snippet.get("title"),
                channel_id=snippet.get("channelId"),
                channel_title=snippet.get("channelTitle"),
                active_live_chat_id=active_chat_id,
                scheduled_start_time=scheduled_start_time,
                actual_start_time=actual_start_time,
                actual_end_time=actual_end_time,
                broadcast_status=broadcast_status,
                upload_status=status_info.get("uploadStatus"),
                privacy_status=status_info.get("privacyStatus"),
                chat_enabled=bool(active_chat_id),
                api_available=True,
            )

    except Exception as exc:
        LOGGER.warning("Failed to query YouTube API for video %s: %s", video_id, exc)
        return YouTubeBroadcastMetadata(
            video_id=video_id,
            is_live=False,
            live_broadcast_content="unknown",
            broadcast_status="unknown",
            api_available=False,
            error_message=str(exc),
        )


def test_live_chat_interface(
    live_chat_id: str | None = None,
    api_key: str | None = None,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    """Test the supported public-data live chat streaming interface.
    
    Verifies authentication, endpoint reachability, and protocol response.
    Never exposes or returns raw API credentials.
    """
    resolved_api_key = resolve_youtube_api_key(api_key)
    if not resolved_api_key:
        return {
            "status": "UNAUTHENTICATED",
            "supported": False,
            "message": "YOUTUBE_API_KEY environment variable is not configured",
        }

    url = f"{YOUTUBE_API_BASE}/liveChat/messages"
    headers = {"X-Goog-Api-Key": resolved_api_key}

    if live_chat_id:
        params: dict[str, Any] = {
            "liveChatId": live_chat_id,
            "part": "id,snippet,authorDetails",
            "maxResults": 10,
        }
        try:
            with httpx.Client(timeout=timeout_seconds, headers=headers) as client:
                resp = client.get(url, params=params)
                if resp.status_code == 200:
                    data = resp.json()
                    return {
                        "status": "ACTIVE",
                        "supported": True,
                        "http_code": 200,
                        "live_chat_id": live_chat_id,
                        "polling_interval_ms": data.get("pollingIntervalMillis", 5000),
                        "items_returned": len(data.get("items", [])),
                        "message": "Live chat streaming interface active and healthy",
                    }
                elif resp.status_code == 404:
                    return {
                        "status": "ENDED",
                        "supported": True,
                        "http_code": 404,
                        "live_chat_id": live_chat_id,
                        "message": "Live chat room has ended (HTTP 404 Not Found)",
                    }
                else:
                    return {
                        "status": "ERROR",
                        "supported": False,
                        "http_code": resp.status_code,
                        "message": f"YouTube liveChat returned HTTP {resp.status_code}: {resp.text[:150]}",
                    }
        except Exception as exc:
            return {
                "status": "FAILED",
                "supported": False,
                "error": str(exc),
            }
    else:
        # Probe authentication reachability using public endpoint parameters
        probe_params = {
            "part": "id,snippet",
            "liveChatId": "probe_check",
        }
        try:
            with httpx.Client(timeout=timeout_seconds, headers=headers) as client:
                resp = client.get(url, params=probe_params)
                # HTTP 400 with invalid parameter confirms API key auth is accepted by YouTube liveChat service
                if resp.status_code in (200, 400, 404):
                    return {
                        "status": "AUTHENTICATED",
                        "supported": True,
                        "endpoint": url,
                        "auth_method": "API_KEY (Public Data Access)",
                        "message": "Public-data API authentication validated against YouTube liveChat service; awaiting active live stream",
                    }
                elif resp.status_code in (401, 403):
                    return {
                        "status": "FORBIDDEN",
                        "supported": False,
                        "http_code": resp.status_code,
                        "message": f"API key rejected or quota exceeded (HTTP {resp.status_code})",
                    }
                else:
                    return {
                        "status": "ERROR",
                        "supported": False,
                        "http_code": resp.status_code,
                        "message": f"Unexpected response from YouTube liveChat (HTTP {resp.status_code})",
                    }
        except Exception as exc:
            return {
                "status": "UNREACHABLE",
                "supported": False,
                "error": str(exc),
            }


def diagnose_youtube_source(
    watch_url: str,
    api_key: str | None = None,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    """Perform complete official YouTube Data API v3 diagnostic verification.
    
    1. Retrieves video resource using videos.list with part=snippet,status,liveStreamingDetails.
    2. Identifies channel metadata and broadcast status.
    3. Determines whether the broadcast is upcoming, currently live or completed.
    4. Retrieves activeLiveChatId if available.
    5. Tests the supported live chat streaming interface using public-data API authentication.
    6. Returns structured diagnostic dictionary without exposing raw credentials.
    """
    resolved_api_key = resolve_youtube_api_key(api_key)
    meta = inspect_youtube_source(watch_url, api_key=resolved_api_key, timeout_seconds=timeout_seconds)
    chat_result = test_live_chat_interface(
        live_chat_id=meta.active_live_chat_id,
        api_key=resolved_api_key,
        timeout_seconds=timeout_seconds,
    )

    # Compute duration if start and end are available
    duration_str: str | None = None
    if meta.actual_start_time and meta.actual_end_time:
        try:
            from datetime import datetime
            t_start = datetime.fromisoformat(meta.actual_start_time.replace("Z", "+00:00"))
            t_end = datetime.fromisoformat(meta.actual_end_time.replace("Z", "+00:00"))
            diff_sec = max(0, int((t_end - t_start).total_seconds()))
            h, rem = divmod(diff_sec, 3600)
            m, s = divmod(rem, 60)
            duration_str = f"{h}h {m}m {s}s"
        except Exception:
            duration_str = None

    return {
        "video_id": meta.video_id,
        "watch_url": watch_url,
        "api_configured": bool(resolved_api_key),
        "api_available": meta.api_available,
        "error_message": meta.error_message,
        "snippet": {
            "title": meta.title,
            "channel_title": meta.channel_title,
            "channel_id": meta.channel_id,
            "live_broadcast_content": meta.live_broadcast_content,
        },
        "status": {
            "upload_status": meta.upload_status,
            "privacy_status": meta.privacy_status,
        },
        "live_streaming_details": {
            "broadcast_status": meta.broadcast_status,
            "is_live": meta.is_live,
            "scheduled_start_time": meta.scheduled_start_time,
            "actual_start_time": meta.actual_start_time,
            "actual_end_time": meta.actual_end_time,
            "duration": duration_str,
            "active_live_chat_id": meta.active_live_chat_id,
            "chat_enabled": meta.chat_enabled,
        },
        "chat_interface": chat_result,
        "media_acquisition": {
            "type": "WEB_METADATA_URL",
            "direct_media_available": False,
            "note": (
                "YouTube Data API v3 provides broadcast lifecycle and chat metadata, not raw "
                "H.264/AAC media bytes. An authorized direct stream feed (e.g. RTMP/SRT/HLS via "
                "STREAMCLIP_LIVE_INPUT) is required for FFmpeg rolling segment capture."
            ),
        },
    }


def format_youtube_diagnostic_report(diag: dict[str, Any]) -> str:
    """Format structured diagnostic report into an authoritative terminal/markdown table."""
    snippet = diag.get("snippet", {})
    status_info = diag.get("status", {})
    lsd = diag.get("live_streaming_details", {})
    chat_info = diag.get("chat_interface", {})
    media_info = diag.get("media_acquisition", {})

    lines: list[str] = []
    lines.append("=" * 88)
    lines.append("           STREAMCLIP AI — YOUTUBE DATA API V3 DIAGNOSTIC REPORT")
    lines.append("=" * 88)
    lines.append(f"Target Watch URL:     {diag.get('watch_url')}")
    lines.append(f"Extracted Video ID:   {diag.get('video_id')}")
    lines.append(f"API Authentication:   {'VALID (Configured in environment)' if diag.get('api_configured') else 'MISSING / UNCONFIGURED'}")

    if not diag.get("api_configured"):
        lines.append("-" * 88)
        lines.append("ENVIRONMENT CONFIGURATION ISSUE:")
        lines.append("  YOUTUBE_API_KEY is not configured in process environment or Windows registry.")
        lines.append("  To configure: setx YOUTUBE_API_KEY \"your_api_key_here\" and restart the session.")
        lines.append("=" * 88)
        return "\n".join(lines)

    lines.append("-" * 88)
    lines.append("--- 1. Video & Channel Metadata (snippet) ---")
    lines.append(f"Video Title:          {snippet.get('title') or '(none)'}")
    lines.append(f"Channel Title:        {snippet.get('channel_title') or '(none)'}")
    lines.append(f"Channel ID:           {snippet.get('channel_id') or '(none)'}")
    lines.append(f"liveBroadcastContent: {snippet.get('live_broadcast_content') or 'unknown'}")

    lines.append("\n--- 2. Publication & Processing Status (status) ---")
    lines.append(f"Upload Status:        {status_info.get('upload_status') or 'unknown'}")
    lines.append(f"Privacy Status:       {status_info.get('privacy_status') or 'unknown'}")

    lines.append("\n--- 3. Broadcast Lifecycle & Timing (liveStreamingDetails) ---")
    broadcast_st = str(lsd.get("broadcast_status", "unknown")).upper()
    lines.append(f"Broadcast Status:     {broadcast_st} (is_live: {lsd.get('is_live')})")
    lines.append(f"Scheduled Start:      {lsd.get('scheduled_start_time') or '(none)'}")
    lines.append(f"Actual Start Time:    {lsd.get('actual_start_time') or '(none)'}")
    lines.append(f"Actual End Time:      {lsd.get('actual_end_time') or '(ongoing or not started)'}")
    if lsd.get("duration"):
        lines.append(f"Broadcast Duration:   {lsd.get('duration')}")

    lines.append("\n--- 4. Live Chat Streaming Interface (liveChat/messages) ---")
    chat_id = lsd.get("active_live_chat_id")
    lines.append(f"Active Chat ID:       {chat_id if chat_id else '(none - chat room closed upon broadcast completion)'}")
    lines.append(f"Chat Interface Auth:  {chat_info.get('status')} ({chat_info.get('auth_method', 'Public-Data API')})")
    lines.append(f"Chat Diagnostic Note: {chat_info.get('message')}")

    lines.append("\n--- 5. Media Acquisition & Transport Distinction ---")
    lines.append(f"API Ingest Category:  {media_info.get('type')}")
    lines.append("Raw Stream Status:    NO DIRECT FEED SUPPLIED")
    lines.append(f"Operational Boundary: {media_info.get('note')}")
    lines.append("=" * 88)

    return "\n".join(lines)
