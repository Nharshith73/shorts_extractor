"""Streamer identity from explicit attribution metadata, never chat commenters."""

import json
from pathlib import Path
from urllib.parse import urlparse


def streamer_name(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = " ".join(value.split()).strip()
    if "://" in text or text.startswith(("twitch.tv/", "www.twitch.tv/", "youtube.com/")):
        parsed = urlparse(text if "://" in text else "https://"+text)
        if parsed.netloc.lower().removeprefix("www.") not in ("twitch.tv", "youtube.com"):
            return None
        text = parsed.path.strip("/").split("/")[-1]
    return text.lstrip("@")[:80] or None


def metadata_streamer(data: object) -> str | None:
    if not isinstance(data, dict):
        return None
    fields = {str(key).replace("_", "").lower(): value for key, value in data.items()}
    for key in ("streamer", "uploader", "channel", "creator", "author"):
        value = fields.get(key)
        if isinstance(value, dict):
            nested = {str(key).replace("_", "").lower(): item for key, item in value.items()}
            value = next((nested[name] for name in ("login", "name", "displayname") if nested.get(name)), None)
        name = streamer_name(value)
        if name is not None:
            return name
    for key in ("metadata", "vod", "video"):
        name = metadata_streamer(fields.get(key))
        if name:
            return name
    return None


def source_streamer(video: Path) -> str | None:
    path = video.with_suffix(".info.json")
    if not path.is_file():
        return None
    try:
        return metadata_streamer(json.loads(path.read_text(encoding="utf-8-sig")))
    except (OSError, ValueError):
        return None
