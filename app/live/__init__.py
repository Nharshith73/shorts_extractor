"""StreamClip AI — Single-Stream Live Intelligence Subsystem."""

from __future__ import annotations

from .config import (
    AcquisitionConfig,
    AuthorizationConfig,
    AuthorizationError,
    LiveCaptureConfig,
    LiveGpuConfig,
    LiveProcessingConfig,
    LiveQueueConfig,
    LiveSourceConfig,
    MissingMediaInputError,
    load_source_config,
)
from .gpu import GpuResourceManager, VramSnapshot, query_gpu_vram
from .pipeline import SingleStreamLivePipeline
from .queue import LiveJobRecord, LiveJobStatus, LiveJudgmentQueue
from .review import LiveReviewItem, LiveReviewStore
from .supervisor import LiveState, LiveSupervisor, SupervisorStateStore
from .youtube import YouTubeBroadcastMetadata, inspect_youtube_source

__all__ = [
    "AcquisitionConfig",
    "AuthorizationConfig",
    "AuthorizationError",
    "GpuResourceManager",
    "LiveCaptureConfig",
    "LiveGpuConfig",
    "LiveJobRecord",
    "LiveJobStatus",
    "LiveJudgmentQueue",
    "LiveProcessingConfig",
    "LiveQueueConfig",
    "LiveReviewItem",
    "LiveReviewStore",
    "LiveSourceConfig",
    "LiveState",
    "LiveSupervisor",
    "MissingMediaInputError",
    "SingleStreamLivePipeline",
    "SupervisorStateStore",
    "VramSnapshot",
    "YouTubeBroadcastMetadata",
    "inspect_youtube_source",
    "load_source_config",
    "query_gpu_vram",
]
