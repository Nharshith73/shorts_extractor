"""Configuration and validation for Single-Stream Live Operations."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
from typing import Any
import urllib.parse
import yaml

LOGGER = logging.getLogger(__name__)


class AuthorizationError(RuntimeError):
    """Raised when source authorization is missing or incomplete."""


class MissingMediaInputError(RuntimeError):
    """Raised when an authorized media feed cannot be resolved from configuration."""


@dataclass(slots=True)
class AuthorizationConfig:
    """Legal and operational authorization record for creator livestream."""

    permitted: bool = False
    capture: bool = False
    clipping: bool = False
    editing: bool = False
    publishing: bool = False
    permission_reference: str = ""

    def validate(self) -> None:
        """Verify all mandatory permission gates are explicitly true."""
        if not self.permitted:
            raise AuthorizationError("Live capture not permitted: authorization.permitted is false")
        if not self.capture:
            raise AuthorizationError("Stream capture not authorized: authorization.capture is false")
        if not self.clipping:
            raise AuthorizationError("Candidate clipping not authorized: authorization.clipping is false")
        if not self.editing:
            raise AuthorizationError("Video editing not authorized: authorization.editing is false")
        if not self.publishing:
            raise AuthorizationError("Shorts publishing not authorized: authorization.publishing is false")
        if not self.permission_reference.strip():
            raise AuthorizationError("Missing permission_reference identifier in authorization config")


@dataclass(slots=True)
class AcquisitionConfig:
    """Stream acquisition parameters and media feed resolution."""

    method: str = "direct_media_input"
    media_url_env: str = "STREAMCLIP_LIVE_INPUT"
    media_url: str = ""
    reconnect_enabled: bool = True
    reconnect_max_attempts: int = 0  # 0 = infinite

    def resolve_media_url(self, watch_url: str | None = None) -> str:
        """Resolve an FFmpeg-compatible media URL without circumvention or undocumented scraping.
        
        A YouTube watch-page URL (e.g. https://www.youtube.com/watch?v=...) is a metadata reference,
        not a direct media stream. An authorized direct media feed must be supplied via environment
        variable (e.g. STREAMCLIP_LIVE_INPUT) or explicit creator RTMP/SRT/HLS media_url.
        """
        # 1. Environment variable override
        if self.media_url_env:
            env_val = os.environ.get(self.media_url_env, "").strip()
            if env_val:
                LOGGER.info("Resolved live media feed from env var '%s'", self.media_url_env)
                return env_val

        # 2. Explicit direct media URL in config
        if self.media_url.strip():
            return self.media_url.strip()

        # 3. Check if watch_url is a YouTube page
        if watch_url:
            parsed = urllib.parse.urlparse(watch_url)
            if "youtube.com" in parsed.netloc or "youtu.be" in parsed.netloc:
                raise MissingMediaInputError(
                    f"The configured watch URL '{watch_url}' is a web metadata reference and is not "
                    "a direct FFmpeg-compatible media stream. StreamClip AI does not implement undocumented "
                    f"scraping or access-control bypass. Please provide an authorized live stream feed via "
                    f"environment variable '{self.media_url_env}' or direct RTMP/SRT/HLS media feed in source.yaml."
                )

        raise MissingMediaInputError(
            f"No live media input stream found. Please configure '{self.media_url_env}' environment variable "
            "or set 'acquisition.media_url' in source.yaml to an authorized direct RTMP/SRT/HLS feed."
        )


@dataclass(slots=True)
class LiveCaptureConfig:
    segment_seconds: float = 6.0
    retention_seconds: float = 900.0
    segments_dir: Path = Path("data/live_segments")

    def validate(self) -> None:
        if self.segment_seconds <= 0:
            raise ValueError("segment_seconds must be positive")
        if self.retention_seconds <= 0:
            raise ValueError("retention_seconds must be positive")


@dataclass(slots=True)
class LiveProcessingConfig:
    detection_enabled: bool = True
    transcription_enabled: bool = True
    ai_judging_enabled: bool = True
    rendering_enabled: bool = True
    human_approval_required: bool = True


@dataclass(slots=True)
class LiveQueueConfig:
    max_queue_depth: int = 50
    max_attempts: int = 3
    stale_job_timeout_seconds: float = 120.0


@dataclass(slots=True)
class LiveGpuConfig:
    serialize_vlm_and_render: bool = True
    max_vram_mb: float = 7800.0


@dataclass(slots=True)
class LiveSourceConfig:
    """Complete root configuration for a single authorized live stream."""

    id: str = "primary_creator"
    creator_name: str = "authorized_creator"
    platform: str = "youtube"
    watch_url: str = ""
    authorization: AuthorizationConfig = field(default_factory=AuthorizationConfig)
    acquisition: AcquisitionConfig = field(default_factory=AcquisitionConfig)
    capture: LiveCaptureConfig = field(default_factory=LiveCaptureConfig)
    processing: LiveProcessingConfig = field(default_factory=LiveProcessingConfig)
    queue: LiveQueueConfig = field(default_factory=LiveQueueConfig)
    gpu: LiveGpuConfig = field(default_factory=LiveGpuConfig)

    def validate(self) -> None:
        """Validate entire source configuration."""
        if not self.id.strip():
            raise ValueError("source id must not be empty")
        self.authorization.validate()
        self.capture.validate()


def load_source_config(config_path: Path, media_override: str | None = None) -> LiveSourceConfig:
    """Load and validate source.yaml."""
    path = config_path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Source configuration file not found at {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    src_data = data.get("source", {})
    auth_data = src_data.get("authorization", {})
    acq_data = src_data.get("acquisition", {})
    cap_data = data.get("capture", {})
    proc_data = data.get("processing", {})
    queue_data = data.get("queue", {})
    gpu_data = data.get("gpu", {})

    auth_cfg = AuthorizationConfig(
        permitted=bool(auth_data.get("permitted", False)),
        capture=bool(auth_data.get("capture", False)),
        clipping=bool(auth_data.get("clipping", False)),
        editing=bool(auth_data.get("editing", False)),
        publishing=bool(auth_data.get("publishing", False)),
        permission_reference=str(auth_data.get("permission_reference", "")),
    )

    acq_cfg = AcquisitionConfig(
        method=str(acq_data.get("method", "direct_media_input")),
        media_url_env=str(acq_data.get("media_url_env", "STREAMCLIP_LIVE_INPUT")),
        media_url=str(media_override or acq_data.get("media_url", "")),
        reconnect_enabled=bool(acq_data.get("reconnect_enabled", True)),
        reconnect_max_attempts=int(acq_data.get("reconnect_max_attempts", 0)),
    )

    segments_dir = Path(cap_data.get("segments_dir", "data/live_segments"))
    if not segments_dir.is_absolute():
        segments_dir = (path.parent / segments_dir).resolve()

    cap_cfg = LiveCaptureConfig(
        segment_seconds=float(cap_data.get("segment_seconds", 6.0)),
        retention_seconds=float(cap_data.get("retention_seconds", 900.0)),
        segments_dir=segments_dir,
    )

    proc_cfg = LiveProcessingConfig(
        detection_enabled=bool(proc_data.get("detection_enabled", True)),
        transcription_enabled=bool(proc_data.get("transcription_enabled", True)),
        ai_judging_enabled=bool(proc_data.get("ai_judging_enabled", True)),
        rendering_enabled=bool(proc_data.get("rendering_enabled", True)),
        human_approval_required=bool(proc_data.get("human_approval_required", True)),
    )

    queue_cfg = LiveQueueConfig(
        max_queue_depth=int(queue_data.get("max_queue_depth", 50)),
        max_attempts=int(queue_data.get("max_attempts", 3)),
        stale_job_timeout_seconds=float(queue_data.get("stale_job_timeout_seconds", 120.0)),
    )

    gpu_cfg = LiveGpuConfig(
        serialize_vlm_and_render=bool(gpu_data.get("serialize_vlm_and_render", True)),
        max_vram_mb=float(gpu_data.get("max_vram_mb", 7800.0)),
    )

    config = LiveSourceConfig(
        id=str(src_data.get("id", "primary_creator")),
        creator_name=str(src_data.get("creator_name", "authorized_creator")),
        platform=str(src_data.get("platform", "youtube")),
        watch_url=str(src_data.get("watch_url", "")),
        authorization=auth_cfg,
        acquisition=acq_cfg,
        capture=cap_cfg,
        processing=proc_cfg,
        queue=queue_cfg,
        gpu=gpu_cfg,
    )
    config.validate()
    return config
