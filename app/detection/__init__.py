"""Stage 2 lightweight, local excitement detection."""

from .config import DetectionConfig, load_detection_config, load_segments_dir
from .models import DetectedEvent, FeatureBin
from .runner import DetectionRunner

__all__ = [
    "DetectedEvent",
    "DetectionConfig",
    "DetectionRunner",
    "FeatureBin",
    "load_detection_config",
    "load_segments_dir",
]
