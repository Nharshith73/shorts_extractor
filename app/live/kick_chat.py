"""Anonymous Kick chat client using official public guest realtime connection."""

from __future__ import annotations

from collections import deque
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from queue import Queue
import sys
import threading
import time
from typing import Any, Callable
from urllib.request import Request, urlopen
from uuid import uuid4

import websocket

from app.live.twitch_irc import ChatMessage

LOGGER = logging.getLogger(__name__)

REALTIME_BASE = "https://web.kick.com/api/v1/realtime"


def _request_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 10.0) -> dict[str, Any]:
    """Helper to perform HTTP JSON requests to Kick APIs."""
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = Request(
        url,
        data=body,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Origin": "https://kick.com",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) clip-watcher/1.0",
        },
    )
    with urlopen(req, timeout=timeout) as response:
        return json.load(response)


def _get_guest_token(client_id: str) -> str:
    """Fetch anonymous guest token from Kick realtime auth."""
    data = _request_json(f"{REALTIME_BASE}/auth/connection", {"client_id": client_id})
    return str(data["data"]["token"])


class KickChatClient:
    """Thread-safe anonymous Kick chat client over Centrifugo WebSocket."""

    def __init__(
        self,
        channel: str,
        max_buffer_size: int = 10000,
        on_message_callback: Callable[[ChatMessage], None] | None = None,
    ) -> None:
        self.channel = channel.lower().strip().replace("https://kick.com/", "").strip("/")
        self.max_buffer_size = max_buffer_size
        self.on_message_callback = on_message_callback

        self._messages: deque[ChatMessage] = deque(maxlen=max_buffer_size)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._connected = False
        self._total_messages_received = 0

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def total_messages_received(self) -> int:
        return self._total_messages_received

    def start(self) -> None:
        """Start listening for chat messages in a background daemon thread."""
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"kick-chat-{self.channel}",
            daemon=True,
        )
        self._thread.start()
        LOGGER.info("Kick chat client started for channel %s", self.channel)

    def stop(self) -> None:
        """Stop chat client and terminate thread."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
            self._thread = None

        self._connected = False
        LOGGER.info("Kick chat client stopped for channel %s", self.channel)

    def get_messages(self, limit: int = 100) -> list[ChatMessage]:
        """Get the most recent messages."""
        with self._lock:
            return list(self._messages)[-limit:]

    def get_messages_since(self, timestamp: float) -> list[ChatMessage]:
        """Get all messages arriving on or after timestamp."""
        with self._lock:
            return [m for m in self._messages if m.timestamp >= timestamp]

    def count_messages_in_window(self, start_ts: float, end_ts: float) -> int:
        """Count messages between start_ts and end_ts."""
        with self._lock:
            return sum(1 for m in self._messages if start_ts <= m.timestamp < end_ts)

    def _run_loop(self) -> None:
        """Main loop that resolves channel info and listens via Centrifugo WS."""
        while not self._stop_event.is_set():
            try:
                # 1. Fetch channel metadata to obtain numeric channel_id and chatroom_id
                info = _request_json(f"https://kick.com/api/v2/channels/{self.channel}")
                chatroom_id = int(info["chatroom"]["id"])
                channel_id = int(info["id"])
                room = f"chatrooms.{chatroom_id}.v2"
                client_id = str(uuid4())

                # 2. Negotiate realtime connection with accepted centrifugo provider
                options = _request_json(
                    f"{REALTIME_BASE}/channels/{channel_id}/chat/connection",
                    {
                        "client": {"id": client_id, "type": "web"},
                        "capabilities": {"accepted_providers": [{"provider": "centrifugo"}]},
                    },
                )
                connections = options.get("data", {}).get("connections", [])
                ws_url = next(
                    (c["credentials"]["url"] for c in connections if c.get("provider") == "centrifugo"),
                    "",
                )
                if not ws_url.startswith("wss://"):
                    raise ValueError(f"No valid Centrifugo wss URL returned for Kick channel {self.channel}")

                LOGGER.info("Connecting to Kick Centrifugo WebSocket for %s (room %s)", self.channel, room)

                with closing(websocket.create_connection(ws_url, origin="https://kick.com", timeout=10)) as sock:
                    sock.settimeout(1.0)
                    self._listen_socket(sock, room, client_id)

            except Exception as exc:
                self._connected = False
                if not self._stop_event.is_set():
                    LOGGER.warning("Kick chat error for %s (%s). Retrying in 5s...", self.channel, exc)
                    self._stop_event.wait(5.0)

    def _listen_socket(self, sock: websocket.WebSocket, room: str, client_id: str) -> None:
        """Handle Centrifugo handshake, subscription, heartbeat, and incoming frames."""
        token = _get_guest_token(client_id)
        # Centrifugo connect command
        sock.send(json.dumps({"id": 1, "connect": {"token": token, "name": "clip-watcher"}}))

        refresh_at = float("inf")
        deadline = time.monotonic() + 25.0
        heartbeat = 60.0
        command_id = 2

        while not self._stop_event.is_set():
            now = time.monotonic()
            if now > deadline:
                raise ConnectionError("Kick chat timed out")

            if now >= refresh_at:
                command_id += 1
                new_token = _get_guest_token(client_id)
                sock.send(json.dumps({"id": command_id, "refresh": {"token": new_token}}))
                refresh_at = float("inf")

            try:
                frame = sock.recv()
            except websocket.WebSocketTimeoutException:
                continue

            if not frame:
                raise ConnectionError("Kick closed the WebSocket connection")

            deadline = time.monotonic() + heartbeat

            for line in frame.splitlines():
                if not line.strip():
                    continue
                try:
                    reply = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if "error" in reply:
                    raise ConnectionError(f"Kick chat returned error {reply['error'].get('code')}")

                if not reply:
                    # Empty reply keeps connection active
                    sock.send("{}")

                if "connect" in reply:
                    self._connected = True
                    ping_val = reply["connect"].get("ping", 25)
                    heartbeat = max(30.0, float(ping_val) * 2.0 + 5.0)
                    # Subscribe to the room
                    sock.send(json.dumps({"id": 2, "subscribe": {"channel": room}}))
                    LOGGER.info("Connected and subscribing to Kick channel %s (%s)", self.channel, room)

                if "subscribe" in reply:
                    LOGGER.info("Successfully subscribed to Kick chat for %s", self.channel)

                info = reply.get("connect", reply.get("refresh", {}))
                if info.get("expires"):
                    ttl = float(info.get("ttl", 300))
                    refresh_at = time.monotonic() + max(1.0, ttl - 60.0)

                push = reply.get("push", {})
                if "disconnect" in push or "unsubscribe" in push:
                    raise ConnectionError("Kick chat subscription ended")

                if push.get("channel") == room:
                    event = push.get("pub", {}).get("data", {})
                    if event.get("event") == r"App\Events\ChatMessageEvent":
                        chat_data = json.loads(event.get("data", "{}")) if isinstance(event.get("data"), str) else event.get("data", {})
                        username = chat_data.get("sender", {}).get("username", "viewer")
                        content = chat_data.get("content", "")

                        msg = ChatMessage(
                            timestamp=time.time(),
                            channel=self.channel,
                            username=username,
                            text=content,
                        )
                        with self._lock:
                            self._messages.append(msg)
                            self._total_messages_received += 1

                        if self.on_message_callback:
                            try:
                                self.on_message_callback(msg)
                            except Exception as cb_exc:
                                LOGGER.debug("Error in on_message callback: %s", cb_exc)
