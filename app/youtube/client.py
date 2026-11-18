"""YouTube Data API v3 OAuth and Channel Client."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

LOGGER = logging.getLogger(__name__)

CLIENT_SECRET_FILE = Path("data/youtube_client_secret.json")
TOKEN_FILE = Path("data/youtube_token.json")

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/userinfo.profile",
]


def has_client_secret() -> bool:
    """Check if Google OAuth client secrets file is installed."""
    return CLIENT_SECRET_FILE.is_file() and CLIENT_SECRET_FILE.stat().st_size > 20


def is_connected() -> bool:
    """Check if active credentials token file exists."""
    return TOKEN_FILE.is_file() and TOKEN_FILE.stat().st_size > 20


def get_auth_url(redirect_uri: str) -> tuple[str, str]:
    """Generate authorization URL for Google OAuth consent screen."""
    if not has_client_secret():
        raise FileNotFoundError(f"YouTube client secrets file not found: {CLIENT_SECRET_FILE}")

    flow = InstalledAppFlow.from_client_secrets_file(
        str(CLIENT_SECRET_FILE),
        scopes=SCOPES,
        redirect_uri=redirect_uri,
    )
    auth_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )
    return auth_url, state


def exchange_code(code: str, redirect_uri: str) -> dict[str, Any]:
    """Exchange authorization code for OAuth tokens and persist."""
    if not has_client_secret():
        raise FileNotFoundError(f"YouTube client secrets file not found: {CLIENT_SECRET_FILE}")

    flow = InstalledAppFlow.from_client_secrets_file(
        str(CLIENT_SECRET_FILE),
        scopes=SCOPES,
        redirect_uri=redirect_uri,
    )
    flow.fetch_token(code=code)
    credentials = flow.credentials
    save_credentials(credentials)
    return get_channel_info()


def save_credentials(credentials: Credentials) -> None:
    """Save credentials to JSON file."""
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    token_data = {
        "token": credentials.token,
        "refresh_token": credentials.refresh_token,
        "token_uri": credentials.token_uri,
        "client_id": credentials.client_id,
        "client_secret": credentials.client_secret,
        "scopes": credentials.scopes,
        "expiry": credentials.expiry.isoformat() if credentials.expiry else None,
    }
    TOKEN_FILE.write_text(json.dumps(token_data, indent=2), encoding="utf-8")
    LOGGER.info("Saved YouTube credentials to %s", TOKEN_FILE)


def get_credentials() -> Credentials | None:
    """Load and auto-refresh stored Google credentials."""
    if not is_connected():
        return None

    try:
        data = json.loads(TOKEN_FILE.read_text(encoding="utf-8"))
        credentials = Credentials(
            token=data.get("token"),
            refresh_token=data.get("refresh_token"),
            token_uri=data.get("token_uri"),
            client_id=data.get("client_id"),
            client_secret=data.get("client_secret"),
            scopes=data.get("scopes", SCOPES),
        )

        if credentials.expired and credentials.refresh_token:
            LOGGER.info("Refreshing expired YouTube access token...")
            credentials.refresh(Request())
            save_credentials(credentials)

        return credentials
    except Exception as exc:
        LOGGER.warning("Failed to load/refresh YouTube credentials: %s", exc)
        return None


def disconnect() -> bool:
    """Remove stored credentials to disconnect account."""
    if TOKEN_FILE.is_file():
        TOKEN_FILE.unlink()
        LOGGER.info("Removed YouTube credentials file.")
        return True
    return False


def get_youtube_service():
    """Build and return an authorized YouTube Data API service object."""
    credentials = get_credentials()
    if not credentials:
        return None
    return build("youtube", "v3", credentials=credentials)


def get_channel_info() -> dict[str, Any]:
    """Fetch profile and channel stats for the connected YouTube account."""
    if not is_connected():
        return {
            "connected": False,
            "has_client_secret": has_client_secret(),
            "message": "YouTube account not connected",
        }

    youtube = get_youtube_service()
    if not youtube:
        return {
            "connected": False,
            "has_client_secret": has_client_secret(),
            "message": "Failed to authenticate YouTube service",
        }

    try:
        response = youtube.channels().list(mine=True, part="snippet,statistics").execute()
        items = response.get("items", [])
        if not items:
            return {
                "connected": True,
                "has_channel": False,
                "message": "Connected Google account has no public YouTube channel",
            }

        ch = items[0]
        snippet = ch.get("snippet", {})
        stats = ch.get("statistics", {})

        avatar = ""
        thumbnails = snippet.get("thumbnails", {})
        if "medium" in thumbnails:
            avatar = thumbnails["medium"]["url"]
        elif "default" in thumbnails:
            avatar = thumbnails["default"]["url"]

        return {
            "connected": True,
            "has_channel": True,
            "has_client_secret": True,
            "channel_id": ch.get("id"),
            "title": snippet.get("title", "YouTube Creator"),
            "description": snippet.get("description", ""),
            "custom_url": snippet.get("customUrl", ""),
            "avatar_url": avatar,
            "subscriber_count": int(stats.get("subscriberCount", 0)),
            "video_count": int(stats.get("videoCount", 0)),
            "view_count": int(stats.get("viewCount", 0)),
        }
    except Exception as exc:
        LOGGER.error("Failed to query YouTube channels: %s", exc)
        return {
            "connected": True,
            "error": str(exc),
            "has_client_secret": True,
        }
