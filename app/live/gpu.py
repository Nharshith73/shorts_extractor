"""Conservative GPU Resource Manager for AI Inference and Hardware Rendering."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import logging
import subprocess
import threading
import time
from typing import Iterator

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class VramSnapshot:
    used_mb: float
    free_mb: float
    total_mb: float


def query_gpu_vram() -> VramSnapshot | None:
    """Query current GPU VRAM utilization via nvidia-smi."""
    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3.0,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            parts = [float(x.strip()) for x in res.stdout.strip().split(",")]
            if len(parts) >= 3:
                return VramSnapshot(used_mb=parts[0], free_mb=parts[1], total_mb=parts[2])
    except Exception:
        pass
    return None


class GpuResourceManager:
    """Enforces conservative serialization between Multimodal AI Inference and NVENC Rendering."""

    def __init__(self, serialize_vlm_and_render: bool = True, min_free_vram_mb: float = 500.0) -> None:
        self.serialize_vlm_and_render = serialize_vlm_and_render
        self.min_free_vram_mb = min_free_vram_mb

        # Internal concurrency locks
        self._global_gpu_lock = threading.Lock()
        self._inference_active = False
        self._render_active = False
        self._lock = threading.Lock()

    @property
    def is_inference_active(self) -> bool:
        with self._lock:
            return self._inference_active

    @property
    def is_render_active(self) -> bool:
        with self._lock:
            return self._render_active

    @contextmanager
    def acquire_inference_slot(self, timeout_seconds: float = 60.0) -> Iterator[None]:
        """Acquire permission to run local VLM inference on GPU."""
        acquired = False
        deadline = time.time() + timeout_seconds

        while time.time() < deadline:
            if not self.serialize_vlm_and_render:
                acquired = True
                break
            # Try to acquire lock
            if self._global_gpu_lock.acquire(timeout=1.0):
                acquired = True
                break

        if not acquired:
            raise TimeoutError(f"Could not acquire GPU inference slot within {timeout_seconds}s (render active)")

        with self._lock:
            self._inference_active = True

        vram = query_gpu_vram()
        if vram:
            LOGGER.debug("GPU inference slot claimed (VRAM: %.1f MB used, %.1f MB free)", vram.used_mb, vram.free_mb)

        try:
            yield
        finally:
            with self._lock:
                self._inference_active = False
            if self.serialize_vlm_and_render:
                self._global_gpu_lock.release()
            LOGGER.debug("GPU inference slot released")

    @contextmanager
    def acquire_render_slot(self, timeout_seconds: float = 60.0, allow_cpu_fallback: bool = True) -> Iterator[bool]:
        """Acquire permission to run NVENC GPU rendering.
        
        Yields True if GPU slot was granted, or False if CPU fallback should be used.
        """
        acquired = False
        deadline = time.time() + timeout_seconds

        # Check VRAM headroom before attempting GPU render
        vram = query_gpu_vram()
        if vram and vram.free_mb < self.min_free_vram_mb:
            LOGGER.warning(
                "Low VRAM headroom (%.1f MB free < %.1f MB threshold); switching render to CPU fallback",
                vram.free_mb,
                self.min_free_vram_mb,
            )
            if allow_cpu_fallback:
                yield False
                return
            raise RuntimeError(f"Insufficient VRAM for GPU render (free: {vram.free_mb} MB)")

        while time.time() < deadline:
            if not self.serialize_vlm_and_render:
                acquired = True
                break
            if self._global_gpu_lock.acquire(timeout=1.0):
                acquired = True
                break

        if not acquired:
            if allow_cpu_fallback:
                LOGGER.info("GPU busy with AI inference; falling back to CPU render without waiting")
                yield False
                return
            raise TimeoutError(f"Could not acquire GPU render slot within {timeout_seconds}s")

        with self._lock:
            self._render_active = True

        try:
            yield True
        finally:
            with self._lock:
                self._render_active = False
            if self.serialize_vlm_and_render:
                self._global_gpu_lock.release()
            LOGGER.debug("GPU render slot released")
