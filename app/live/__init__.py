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
from .preflight import (
    CheckStatus,
    PreflightCheckResult,
    PreflightReport,
    run_preflight,
)
from .queue import LiveJobRecord, LiveJobStatus, LiveJudgmentQueue
from .review import LiveReviewItem, LiveReviewStore
from .supervisor import LiveState, LiveSupervisor, SupervisorStateStore
from .youtube import (
    YouTubeBroadcastMetadata,
    diagnose_youtube_source,
    format_youtube_diagnostic_report,
    inspect_youtube_source,
    resolve_youtube_api_key,
    test_live_chat_interface,
)

__all__ = [
    "AcquisitionConfig",
    "AuthorizationConfig",
    "AuthorizationError",
    "CheckStatus",
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
    "PreflightCheckResult",
    "PreflightReport",
    "SingleStreamLivePipeline",
    "SupervisorStateStore",
    "VramSnapshot",
    "YouTubeBroadcastMetadata",
    "diagnose_youtube_source",
    "format_youtube_diagnostic_report",
    "inspect_youtube_source",
    "load_source_config",
    "query_gpu_vram",
    "resolve_youtube_api_key",
    "run_preflight",
    "test_live_chat_interface",
]
