"""Polling and Ingestion for Live Chat Messages via Official YouTube API."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any

import httpx

from .youtube import YOUTUBE_API_BASE

LOGGER = logging.getLogger(__name__)


class LiveChatPoller:
    """Non-blocking background poller for YouTube live chat messages."""

    def __init__(
        self,
        live_chat_id: str | None,
        output_file: Path,
        api_key: str | None = None,
        poll_interval_seconds: float = 5.0,
    ) -> None:
        self.live_chat_id = live_chat_id
        self.output_file = output_file
        self.api_key = api_key or os.environ.get("YOUTUBE_API_KEY", "").strip()
        self.poll_interval_seconds = poll_interval_seconds

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._next_page_token: str | None = None
        self._message_count = 0
        self._messages: list[dict[str, Any]] = []
        self._lock = threading.Lock()

        self.output_file.parent.mkdir(parents=True, exist_ok=True)

    @property
    def is_available(self) -> bool:
        return bool(self.live_chat_id and self.api_key)

    @property
    def message_count(self) -> int:
        with self._lock:
            return self._message_count

    def start(self) -> None:
        if not self.is_available:
            LOGGER.info("Live chat poller not started: activeLiveChatId or YOUTUBE_API_KEY not configured")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="live-chat-poller",
            daemon=True,
        )
        self._thread.start()
        LOGGER.info("Live chat poller started for chat_id: %s", self.live_chat_id)

    def _poll_loop(self) -> None:
        url = f"{YOUTUBE_API_BASE}/liveChat/messages"

        while not self._stop_event.is_set():
            params: dict[str, Any] = {
                "liveChatId": self.live_chat_id,
                "part": "id,snippet,authorDetails",
                "key": self.api_key,
                "maxResults": 200,
            }
            if self._next_page_token:
                params["pageToken"] = self._next_page_token

            sleep_duration = self.poll_interval_seconds
            try:
                with httpx.Client(timeout=10.0) as client:
                    resp = client.get(url, params=params)
                    if resp.status_code == 200:
                        data = resp.json()
                        self._next_page_token = data.get("nextPageToken")
                        poll_ms = data.get("pollingIntervalMillis", int(self.poll_interval_seconds * 1000))
                        sleep_duration = max(2.0, poll_ms / 1000.0)

                        new_items = data.get("items", [])
                        if new_items:
                            self._append_messages(new_items)
                    elif resp.status_code == 403:
                        LOGGER.warning("YouTube chat quota exceeded or forbidden: %s", resp.text[:200])
                        sleep_duration = 60.0
                    elif resp.status_code == 404:
                        LOGGER.info("Live chat stream ended (HTTP 404)")
                        break
                    else:
                        LOGGER.warning("YouTube chat poll error HTTP %d: %s", resp.status_code, resp.text[:200])

            except Exception as exc:
                LOGGER.warning("Error during chat poll: %s", exc)
                sleep_duration = 10.0

            # Interruptible sleep
            self._stop_event.wait(timeout=sleep_duration)

    def _append_messages(self, items: list[dict[str, Any]]) -> None:
        parsed_batch = []
        for item in items:
            snippet = item.get("snippet", {})
            author = item.get("authorDetails", {})
            msg = {
                "id": item.get("id"),
                "published_at": snippet.get("publishedAt"),
                "display_message": snippet.get("displayMessage"),
                "author_channel_id": author.get("channelId"),
                "author_display_name": author.get("displayName"),
                "is_moderator": author.get("isChatModerator", False),
                "is_sponsor": author.get("isChatSponsor", False),
            }
            parsed_batch.append(msg)

        with self._lock:
            self._messages.extend(parsed_batch)
            self._message_count = len(self._messages)
            # Atomically write snapshot to output_file
            tmp_file = self.output_file.with_suffix(".tmp.json")
            tmp_file.write_text(json.dumps(self._messages, indent=2), encoding="utf-8")
            tmp_file.replace(self.output_file)

        LOGGER.info("Captured %d new live chat messages (total: %d)", len(parsed_batch), self._message_count)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        LOGGER.info("Live chat poller stopped")
