"""Defensive TwitchDownloader JSON adapter; no live chat or API access."""

from bisect import bisect_right
from dataclasses import dataclass, field
import json
import logging
import math
from pathlib import Path
import re
from typing import Sequence

import yaml

from app.detection.models import FeatureBin
from app.detection.scoring import AuxiliaryFeatures
from .timeline import media_seconds

LOGGER = logging.getLogger(__name__)
CHAT_NAMES = ("chat_message_rate", "chat_unique_chatters", "chat_emote_density", "chat_caps_ratio")
FLOORS = dict(zip(CHAT_NAMES, (1.0, 1.0, 0.05, 0.05)))


@dataclass(frozen=True, slots=True)
class ChatConfig:
    time_offset_seconds: float = 0.0
    weights: dict[str, float] = field(default_factory=lambda: {name: 1.0 for name in CHAT_NAMES})
    scale_floors: dict[str, float] = field(default_factory=lambda: dict(FLOORS))


def load_chat_config(path: Path) -> ChatConfig:
    with path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    section = raw.get("detection", {}).get("chat", {})
    if not isinstance(section, dict) or set(section) - {"time_offset_seconds", "weights", "scale_floors"}:
        raise ValueError("invalid detection.chat configuration")
    weights, floors = {name: 1.0 for name in CHAT_NAMES}, dict(FLOORS)
    for key, target in (("weights", weights), ("scale_floors", floors)):
        values = section.get(key, {})
        if not isinstance(values, dict) or set(values) - set(CHAT_NAMES):
            raise ValueError(f"invalid detection.chat.{key}")
        target.update({name: float(value) for name, value in values.items()})
    offset = float(section.get("time_offset_seconds", 0))
    if not math.isfinite(offset) or any(not math.isfinite(value) or value < 0 for value in weights.values()):
        raise ValueError("chat weights and offset must be finite; weights must be non-negative")
    if any(not math.isfinite(value) or value <= 0 for value in floors.values()):
        raise ValueError("chat scale floors must be finite and positive")
    return ChatConfig(offset, weights, floors)


def _key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", key.lower())


def _mapping(value: object) -> dict[str, object]:
    return {_key(str(key)): item for key, item in value.items()} if isinstance(value, dict) else {}


@dataclass(frozen=True, slots=True)
class ChatMessage:
    seconds: float
    chatter: str | None
    text: str
    emotes: int


def parse_chat(data: object, time_offset_seconds: float = 0) -> list[ChatMessage]:
    """Accept Comments/comments and snake/camel field casing; skip invalid offsets."""
    root = _mapping(data)
    comments = root.get("comments", data if isinstance(data, list) else None)
    if not isinstance(comments, list):
        raise ValueError("chat JSON must have a Comments/comments array")
    emote_names: set[str] = set()
    def collect_names(value: object) -> None:
        if isinstance(value, list):
            for entry in value:
                collect_names(entry)
        elif isinstance(value, dict):
            normal = _mapping(value)
            for name in (normal.get("name"), normal.get("code")):
                if isinstance(name, str):
                    emote_names.add(name)
            for entry in value.values():
                if isinstance(entry, (list, dict)):
                    collect_names(entry)
    collect_names(root.get("emotes", {}))
    embedded = _mapping(root.get("embeddeddata"))
    for field in ("firstparty", "thirdparty"):
        collect_names(embedded.get(field, []))
    messages: list[ChatMessage] = []
    invalid = 0
    for raw in comments:
        comment = _mapping(raw)
        try:
            offset = float(comment.get("contentoffsetseconds")) + time_offset_seconds
            if not math.isfinite(offset) or offset < 0:
                raise ValueError("invalid offset")
        except (TypeError, ValueError, OverflowError):
            invalid += 1
            continue
        message = _mapping(comment.get("message"))
        fragments = message.get("fragments")
        fragments = fragments if isinstance(fragments, list) else []
        body = message.get("body")
        text = body if isinstance(body, str) else "".join(str(_mapping(fragment).get("text", "") or "") for fragment in fragments)
        commenter = _mapping(comment.get("commenter"))
        identity = next((commenter.get(field) for field in ("id", "displayname", "name") if commenter.get(field)), None)
        chatter = str(identity).casefold() if identity is not None and str(identity) else None
        annotated = sum(1 for fragment in fragments if _mapping(fragment).get("emoticon") or _mapping(fragment).get("emote"))
        ranges = message.get("emoticons")
        range_count = len(ranges) if isinstance(ranges, list) else 0
        named = sum(token in emote_names for token in text.split())
        messages.append(ChatMessage(offset, chatter, text, max(annotated, range_count, named)))
    if invalid:
        LOGGER.warning("Skipped %d chat records with invalid/out-of-range timestamps", invalid)
    return sorted(messages, key=lambda message: message.seconds)


def aggregate_chat(messages: Sequence[ChatMessage], features: Sequence[FeatureBin], config: ChatConfig) -> AuxiliaryFeatures:
    """Bin rates, distinct known authors, emote tokens, and uppercase/letter ratio."""
    starts = [media_seconds(feature.wall_utc) for feature in features]
    buckets: list[list[ChatMessage]] = [[] for _ in features]
    for message in messages:
        index = bisect_right(starts, message.seconds) - 1
        if index >= 0 and message.seconds < starts[index] + features[index].bin_seconds:
            buckets[index].append(message)
    values = {}
    for feature, bucket in zip(features, buckets):
        tokens = sum(len(message.text.split()) for message in bucket)
        letters = [char for message in bucket for char in message.text if char.isalpha()]
        values[feature.wall_utc] = dict(zip(CHAT_NAMES, (
            len(bucket) / feature.bin_seconds,
            float(len({message.chatter for message in bucket if message.chatter is not None})),
            min(1.0, sum(message.emotes for message in bucket) / tokens) if tokens else 0.0,
            sum(char.isupper() for char in letters) / len(letters) if letters else 0.0,
        )))
    return AuxiliaryFeatures(values, config.weights, config.scale_floors)


def load_chat(path: Path, features: Sequence[FeatureBin], config: ChatConfig) -> AuxiliaryFeatures:
    with path.open(encoding="utf-8-sig") as stream:
        messages = parse_chat(json.load(stream), config.time_offset_seconds)
    return aggregate_chat(messages, features, config)
