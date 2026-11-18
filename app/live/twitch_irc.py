"""Anonymous read-only Twitch IRC chat client (zero tokens required)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
import random
import re
import socket
import threading
import time
from typing import Callable

LOGGER = logging.getLogger(__name__)

TWITCH_IRC_HOST = "irc.chat.twitch.tv"
TWITCH_IRC_PORT = 6667
PRIVMSG_REGEX = re.compile(r"^:([^!]+)![^@]+@[^\s]+\s+PRIVMSG\s+#[^\s]+\s+:(.*)$")


@dataclass(slots=True)
class ChatMessage:
    """Parsed Twitch chat message."""

    timestamp: float
    channel: str
    username: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "channel": self.channel,
            "username": self.username,
            "text": self.text,
            "datetime": datetime.fromtimestamp(self.timestamp, timezone.utc).isoformat(),
        }


class TwitchIRCClient:
    """Thread-safe anonymous Twitch IRC client for reading live chat."""

    def __init__(
        self,
        channel: str,
        host: str = TWITCH_IRC_HOST,
        port: int = TWITCH_IRC_PORT,
        max_buffer_size: int = 10000,
        on_message_callback: Callable[[ChatMessage], None] | None = None,
    ) -> None:
        self.channel = channel.lower().lstrip("#").strip()
        self.host = host
        self.port = port
        self.max_buffer_size = max_buffer_size
        self.on_message_callback = on_message_callback

        self._messages: deque[ChatMessage] = deque(maxlen=max_buffer_size)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._connected = False
        self._socket: socket.socket | None = None
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
            name=f"twitch-irc-{self.channel}",
            daemon=True,
        )
        self._thread.start()
        LOGGER.info("Twitch IRC client started for channel #%s", self.channel)

    def stop(self) -> None:
        """Stop IRC client and disconnect socket."""
        self._stop_event.set()
        if self._socket:
            try:
                self._socket.shutdown(socket.SHUT_RDWR)
                self._socket.close()
            except Exception:
                pass
            self._socket = None

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            self._thread = None

        self._connected = False
        LOGGER.info("Twitch IRC client stopped for channel #%s", self.channel)

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
        """Main connection and message reading loop with auto-reconnect."""
        nick = f"justinfan{random.randint(10000, 99999)}"

        while not self._stop_event.is_set():
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(5.0)
                s.connect((self.host, self.port))
                self._socket = s

                # Anonymous IRC handshake
                s.sendall(b"PASS SCHMOOPIE\r\n")
                s.sendall(f"NICK {nick}\r\n".encode("utf-8"))
                s.sendall(f"JOIN #{self.channel}\r\n".encode("utf-8"))
                self._connected = True
                LOGGER.info("Connected anonymously to Twitch IRC as %s on #%s", nick, self.channel)

                buffer = ""
                while not self._stop_event.is_set():
                    try:
                        chunk = s.recv(4096).decode("utf-8", errors="ignore")
                        if not chunk:
                            LOGGER.warning("IRC connection closed by remote server")
                            break
                        buffer += chunk
                        while "\r\n" in buffer:
                            line, buffer = buffer.split("\r\n", 1)
                            self._handle_line(s, line)
                    except socket.timeout:
                        continue
                    except Exception as exc:
                        if not self._stop_event.is_set():
                            LOGGER.warning("Error reading from IRC socket: %s", exc)
                        break

            except Exception as exc:
                if not self._stop_event.is_set():
                    LOGGER.warning("Could not connect to Twitch IRC (%s). Retrying in 3s...", exc)
            finally:
                self._connected = False
                if self._socket:
                    try:
                        self._socket.close()
                    except Exception:
                        pass
                    self._socket = None

            if not self._stop_event.is_set():
                time.sleep(3.0)

    def _handle_line(self, sock: socket.socket, line: str) -> None:
        """Parse raw IRC line, reply to PING, and dispatch PRIVMSG."""
        if not line:
            return

        # PING / PONG heartbeat
        if line.startswith("PING"):
            pong_resp = line.replace("PING", "PONG", 1) + "\r\n"
            try:
                sock.sendall(pong_resp.encode("utf-8"))
            except Exception:
                pass
            return

        # PRIVMSG chat message
        match = PRIVMSG_REGEX.match(line)
        if match:
            username = match.group(1).strip()
            text = match.group(2).strip()
            msg = ChatMessage(
                timestamp=time.time(),
                channel=self.channel,
                username=username,
                text=text,
            )
            with self._lock:
                self._messages.append(msg)
                self._total_messages_received += 1

            if self.on_message_callback:
                try:
                    self.on_message_callback(msg)
                except Exception as exc:
                    LOGGER.debug("Error in on_message callback: %s", exc)
