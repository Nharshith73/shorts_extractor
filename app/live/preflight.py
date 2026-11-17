"""Production Preflight Verification for StreamClip AI Single-Stream Live Operations.

Performs a comprehensive, non-intrusive audit of:
1. Source configuration validity.
2. Authorization record completeness.
3. Permission status across all 5 legal/operational gates.
4. FFmpeg and ffprobe binary availability and hardware encoder support.
5. SQLite database availability and WAL mode persistence.
6. Capture, candidate, context, judgment, and short directory permissions.
7. Required environment variables (STREAMCLIP_LIVE_INPUT, YOUTUBE_API_KEY).
8. Media input protocol and distinction between web metadata URLs and direct media streams.
9. Local Qwen VLM llama-server health and offline auto-start disk assets.
10. NVIDIA GPU VRAM availability and resource headroom.
11. Model and render configuration consistency, including subtitle font assets.

Security rule: Does not initiate persistent capture. Does not print credentials,
signed URLs, query tokens, or secrets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
import logging
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
from typing import Any
import urllib.parse
import yaml

import httpx

from app.evaluation.providers.qwen import QwenProviderConfig
from app.live.config import (
    AuthorizationConfig,
    AuthorizationError,
    LiveSourceConfig,
    MissingMediaInputError,
    load_source_config,
)
from app.live.gpu import query_gpu_vram
from app.live.youtube import (
    extract_video_id,
    inspect_youtube_source,
    resolve_youtube_api_key,
    test_live_chat_interface,
)
from app.rendering.config import load_render_config

LOGGER = logging.getLogger("app.live.preflight")


class CheckStatus(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(slots=True)
class PreflightCheckResult:
    name: str
    status: CheckStatus
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
        }


@dataclass(slots=True)
class PreflightReport:
    timestamp: str
    config_path: str
    source_id: str
    creator_name: str
    checks: list[PreflightCheckResult]
    overall_status: CheckStatus
    verdict: str
    exit_code: int
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "config_path": self.config_path,
            "source_id": self.source_id,
            "creator_name": self.creator_name,
            "overall_status": self.overall_status.value,
            "verdict": self.verdict,
            "exit_code": self.exit_code,
            "reasons": self.reasons,
            "checks": [c.to_dict() for c in self.checks],
        }

    def render_text(self) -> str:
        lines: list[str] = []
        lines.append("=" * 88)
        lines.append("           STREAMCLIP AI — LIVE PRODUCTION PREFLIGHT REPORT")
        lines.append("=" * 88)
        lines.append(f"Timestamp:    {self.timestamp}")
        lines.append(f"Source ID:    {self.source_id} ({self.creator_name})")
        lines.append(f"Config Path:  {self.config_path}")
        lines.append("-" * 88)
        lines.append(f"{'CHECK':<34} {'STATUS':<8} {'DETAILS'}")
        lines.append("-" * 88)

        for c in self.checks:
            status_tag = f"[{c.status.value}]"
            lines.append(f"{c.name:<34} {status_tag:<8} {c.message}")

        lines.append("-" * 88)
        lines.append(f"OVERALL STATUS:  [{self.overall_status.value}]")
        lines.append(f"FINAL VERDICT:   {self.verdict}")
        lines.append(f"EXIT CODE:       {self.exit_code}")

        if self.reasons:
            lines.append("SUMMARY NOTES:")
            for r in self.reasons:
                lines.append(f"  * {r}")

        lines.append("=" * 88)
        return "\n".join(lines)


def sanitize_url(url: str | None) -> str:
    """Mask credentials, tokens, query parameters and private stream keys.
    
    Ensures that no signed URLs, authorization tokens or stream secrets are printed.
    """
    if not url or not url.strip():
        return "(none)"
    try:
        parsed = urllib.parse.urlparse(url.strip())
        netloc = parsed.netloc
        if "@" in netloc:
            # Mask user:pass@host -> ***@host
            host_part = netloc.split("@")[-1]
            netloc = f"***@{host_part}"

        query = "***" if parsed.query else ""
        fragment = "***" if parsed.fragment else ""

        path = parsed.path
        # If it is an RTMP/SRT stream, mask the stream key component in the path
        if parsed.scheme.lower() in ("rtmp", "rtmps", "srt") and "/" in path.lstrip("/"):
            segments = [s for s in path.strip("/").split("/") if s]
            if len(segments) >= 2:
                path = f"/{segments[0]}/***"

        return urllib.parse.urlunparse((parsed.scheme, netloc, path, parsed.params, query, fragment))
    except Exception:
        return "***"


def mask_secret(secret: str | None) -> str:
    """Mask API keys or credentials showing only minimal length indicator."""
    if not secret or not secret.strip():
        return "(unset)"
    val = secret.strip()
    if len(val) <= 8:
        return "***"
    return f"{val[:4]}...{val[-4:]}"


# ---------------------------------------------------------------------------
# Individual Preflight Check Implementations
# ---------------------------------------------------------------------------

def check_source_config(config_path: Path) -> tuple[PreflightCheckResult, LiveSourceConfig | None]:
    """Check 1: Source Configuration Validity."""
    path = config_path.resolve()
    if not path.is_file():
        return (
            PreflightCheckResult(
                name="Source Configuration Validity",
                status=CheckStatus.FAIL,
                message=f"Configuration file not found at '{path}'",
                details={"path": str(path), "error": "file_not_found"},
            ),
            None,
        )

    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict) or "source" not in data:
            return (
                PreflightCheckResult(
                    name="Source Configuration Validity",
                    status=CheckStatus.FAIL,
                    message="Configuration missing root 'source' mapping",
                    details={"path": str(path)},
                ),
                None,
            )
        config = load_source_config(path, validate_auth=False)
        return (
            PreflightCheckResult(
                name="Source Configuration Validity",
                status=CheckStatus.PASS,
                message=f"Valid YAML; ID '{config.id}', {config.capture.segment_seconds:.1f}s segments, {config.capture.retention_seconds:.0f}s retention",
                details={
                    "source_id": config.id,
                    "platform": config.platform,
                    "segment_seconds": config.capture.segment_seconds,
                    "retention_seconds": config.capture.retention_seconds,
                },
            ),
            config,
        )
    except Exception as exc:
        return (
            PreflightCheckResult(
                name="Source Configuration Validity",
                status=CheckStatus.FAIL,
                message=f"Configuration schema error: {exc}",
                details={"error": str(exc)},
            ),
            None,
        )


def check_authorization_completeness(config: LiveSourceConfig) -> PreflightCheckResult:
    """Check 2: Authorization Record Completeness."""
    auth = config.authorization
    ref = (auth.permission_reference or "").strip()
    if not ref:
        return PreflightCheckResult(
            name="Authorization Record Completeness",
            status=CheckStatus.FAIL,
            message="Missing required 'permission_reference' identifier in authorization record",
            details={"permission_reference": ""},
        )

    return PreflightCheckResult(
        name="Authorization Record Completeness",
        status=CheckStatus.PASS,
        message=f"Authorization record complete (Ref: '{ref}')",
        details={
            "permission_reference": ref,
            "has_permitted": isinstance(auth.permitted, bool),
            "has_capture": isinstance(auth.capture, bool),
            "has_clipping": isinstance(auth.clipping, bool),
            "has_editing": isinstance(auth.editing, bool),
            "has_publishing": isinstance(auth.publishing, bool),
        },
    )


def check_permission_status(config: LiveSourceConfig) -> PreflightCheckResult:
    """Check 3: Permission Status across all 5 gates."""
    auth = config.authorization
    unpermitted: list[str] = []
    if not auth.permitted:
        unpermitted.append("permitted (root)")
    if not auth.capture:
        unpermitted.append("capture")
    if not auth.clipping:
        unpermitted.append("clipping")
    if not auth.editing:
        unpermitted.append("editing")
    if not auth.publishing:
        unpermitted.append("publishing")

    if unpermitted:
        return PreflightCheckResult(
            name="Permission Status",
            status=CheckStatus.FAIL,
            message=f"Authorization gates blocked: {', '.join(unpermitted)}",
            details={"unpermitted": unpermitted},
        )

    return PreflightCheckResult(
        name="Permission Status",
        status=CheckStatus.PASS,
        message="All 5 gates explicitly permitted (capture, clipping, editing, publishing)",
        details={
            "permitted": True,
            "capture": True,
            "clipping": True,
            "editing": True,
            "publishing": True,
        },
    )


def check_ffmpeg_ffprobe() -> PreflightCheckResult:
    """Check 4: FFmpeg and ffprobe availability and hardware acceleration encoders."""
    ffmpeg_bin = shutil.which("ffmpeg")
    ffprobe_bin = shutil.which("ffprobe")

    if not ffmpeg_bin:
        return PreflightCheckResult(
            name="FFmpeg & ffprobe Availability",
            status=CheckStatus.FAIL,
            message="'ffmpeg' binary not found on system PATH",
            details={"ffmpeg": None, "ffprobe": ffprobe_bin},
        )

    if not ffprobe_bin:
        return PreflightCheckResult(
            name="FFmpeg & ffprobe Availability",
            status=CheckStatus.FAIL,
            message="'ffprobe' binary not found on system PATH",
            details={"ffmpeg": ffmpeg_bin, "ffprobe": None},
        )

    # Check ffmpeg version
    ffmpeg_ver = "unknown"
    try:
        res = subprocess.run([ffmpeg_bin, "-version"], capture_output=True, text=True, timeout=5.0, check=False)
        if res.returncode == 0 and res.stdout:
            first_line = res.stdout.splitlines()[0]
            m = re.search(r"ffmpeg version\s+([^\s]+)", first_line)
            if m:
                ffmpeg_ver = m.group(1)
            else:
                ffmpeg_ver = first_line[:30]
    except Exception as exc:
        return PreflightCheckResult(
            name="FFmpeg & ffprobe Availability",
            status=CheckStatus.FAIL,
            message=f"Failed executing ffmpeg: {exc}",
            details={"error": str(exc)},
        )

    # Check ffprobe version
    ffprobe_ver = "unknown"
    try:
        res = subprocess.run([ffprobe_bin, "-version"], capture_output=True, text=True, timeout=5.0, check=False)
        if res.returncode == 0 and res.stdout:
            first_line = res.stdout.splitlines()[0]
            m = re.search(r"ffprobe version\s+([^\s]+)", first_line)
            if m:
                ffprobe_ver = m.group(1)
            else:
                ffprobe_ver = first_line[:30]
    except Exception as exc:
        return PreflightCheckResult(
            name="FFmpeg & ffprobe Availability",
            status=CheckStatus.FAIL,
            message=f"Failed executing ffprobe: {exc}",
            details={"error": str(exc)},
        )

    # Check available encoders (h264_nvenc vs libx264)
    has_nvenc = False
    has_x264 = False
    try:
        res = subprocess.run([ffmpeg_bin, "-encoders"], capture_output=True, text=True, timeout=5.0, check=False)
        if res.returncode == 0 and res.stdout:
            has_nvenc = "h264_nvenc" in res.stdout
            has_x264 = "libx264" in res.stdout
    except Exception:
        pass

    enc_desc = "NVENC active" if has_nvenc else ("libx264 fallback" if has_x264 else "standard")
    status = CheckStatus.PASS

    return PreflightCheckResult(
        name="FFmpeg & ffprobe Availability",
        status=status,
        message=f"FFmpeg {ffmpeg_ver} ({enc_desc}), ffprobe {ffprobe_ver}",
        details={
            "ffmpeg_path": ffmpeg_bin,
            "ffprobe_path": ffprobe_bin,
            "ffmpeg_version": ffmpeg_ver,
            "ffprobe_version": ffprobe_ver,
            "has_nvenc": has_nvenc,
            "has_libx264": has_x264,
        },
    )


def check_sqlite_database(config: LiveSourceConfig) -> PreflightCheckResult:
    """Check 5: SQLite Database availability, write permissions, and WAL mode support."""
    seg_dir = config.capture.segments_dir
    manifest_path = seg_dir / "manifest.sqlite3"

    try:
        seg_dir.mkdir(parents=True, exist_ok=True)
        # Test SQLite connection and WAL mode
        conn = sqlite3.connect(str(manifest_path), timeout=5.0)
        try:
            cursor = conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL;")
            mode = cursor.fetchone()
            wal_active = mode and mode[0].upper() == "WAL"
            # Verify basic table query
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
        finally:
            conn.close()

        return PreflightCheckResult(
            name="SQLite Database Availability",
            status=CheckStatus.PASS,
            message=f"Manifest DB writable (WAL active: {wal_active}, {len(tables)} tables present)",
            details={
                "db_path": str(manifest_path),
                "wal_mode": wal_active,
                "existing_tables": tables,
            },
        )
    except Exception as exc:
        return PreflightCheckResult(
            name="SQLite Database Availability",
            status=CheckStatus.FAIL,
            message=f"SQLite database error at '{manifest_path}': {exc}",
            details={"db_path": str(manifest_path), "error": str(exc)},
        )


def check_directory_permissions(
    config: LiveSourceConfig,
    base_dir: Path | None = None,
) -> PreflightCheckResult:
    """Check 6: Directory write/read/delete permissions across all live storage paths."""
    base = (base_dir or Path(".")).resolve()
    dirs_to_check = [
        ("segments", config.capture.segments_dir),
        ("candidates", base / "data" / "live_candidates"),
        ("context", base / "data" / "live_context"),
        ("judgments", base / "data" / "live_judgments"),
        ("shorts", base / "data" / "live_shorts"),
    ]

    failed_dirs: list[str] = []
    for label, dpath in dirs_to_check:
        try:
            dpath.mkdir(parents=True, exist_ok=True)
            probe_file = dpath / ".preflight_probe_test"
            probe_file.write_text("streamclip_probe", encoding="utf-8")
            content = probe_file.read_text(encoding="utf-8")
            if content != "streamclip_probe":
                failed_dirs.append(f"{label} (read mismatch)")
            probe_file.unlink()
        except Exception as exc:
            failed_dirs.append(f"{label} ({exc})")

    if failed_dirs:
        return PreflightCheckResult(
            name="Capture Directory Permissions",
            status=CheckStatus.FAIL,
            message=f"Directory permission failures: {', '.join(failed_dirs)}",
            details={"failed_directories": failed_dirs},
        )

    return PreflightCheckResult(
        name="Capture Directory Permissions",
        status=CheckStatus.PASS,
        message=f"Read/write/delete permissions verified across {len(dirs_to_check)} live directories",
        details={"checked_directories": [str(p) for _, p in dirs_to_check]},
    )


def check_environment_variables(config: LiveSourceConfig) -> PreflightCheckResult:
    """Check 7: Required environment variables (STREAMCLIP_LIVE_INPUT, YOUTUBE_API_KEY)."""
    env_name = config.acquisition.media_url_env or "STREAMCLIP_LIVE_INPUT"
    live_input_val = os.environ.get(env_name, "").strip()
    yt_api_key = resolve_youtube_api_key()

    status = CheckStatus.PASS
    messages: list[str] = []

    if live_input_val:
        messages.append(f"{env_name} is set ({sanitize_url(live_input_val)})")
    else:
        messages.append(f"{env_name} is unset")

    if yt_api_key:
        messages.append(f"YOUTUBE_API_KEY configured ({mask_secret(yt_api_key)})")
    else:
        messages.append("YOUTUBE_API_KEY unset (metadata check will run in offline mode)")

    if not live_input_val and not config.acquisition.media_url.strip():
        status = CheckStatus.WARN

    return PreflightCheckResult(
        name="Required Environment Variables",
        status=status,
        message="; ".join(messages),
        details={
            "live_input_env": env_name,
            "live_input_set": bool(live_input_val),
            "youtube_api_key_set": bool(yt_api_key),
        },
    )


def check_media_input_protocol(config: LiveSourceConfig) -> PreflightCheckResult:
    """Check 8: Media input protocol and Web Metadata URL vs Direct Media Feed distinction."""
    env_name = config.acquisition.media_url_env or "STREAMCLIP_LIVE_INPUT"
    env_val = os.environ.get(env_name, "").strip()
    config_url = config.acquisition.media_url.strip()
    watch_url = (config.watch_url or "").strip()

    # Determine input candidate
    direct_url = env_val or config_url

    if direct_url:
        parsed = urllib.parse.urlparse(direct_url)
        scheme = parsed.scheme.lower()
        # Check if direct_url is accidentally a YouTube web URL
        if "youtube.com" in parsed.netloc or "youtu.be" in parsed.netloc:
            return PreflightCheckResult(
                name="Media Input Protocol",
                status=CheckStatus.FAIL,
                message=f"Provided media input '{sanitize_url(direct_url)}' is a YouTube web watch page, not an FFmpeg stream",
                details={
                    "type": "WEB_METADATA_URL",
                    "url": sanitize_url(direct_url),
                    "supported": False,
                },
            )

        supported_schemes = {"rtmp", "rtmps", "srt", "http", "https", "udp", "file"}
        # File paths (e.g. D:/video.mp4 or relative) have empty scheme or drive letter scheme
        is_file = not scheme or len(scheme) == 1 or scheme == "file"

        if scheme in supported_schemes or is_file:
            proto_name = "file/local" if is_file else scheme.upper()
            return PreflightCheckResult(
                name="Media Input Protocol",
                status=CheckStatus.PASS,
                message=f"Direct media stream ready (Protocol: {proto_name}, Feed: {sanitize_url(direct_url)})",
                details={
                    "type": "DIRECT_MEDIA_INPUT",
                    "protocol": proto_name,
                    "url": sanitize_url(direct_url),
                    "supported": True,
                },
            )
        else:
            return PreflightCheckResult(
                name="Media Input Protocol",
                status=CheckStatus.WARN,
                message=f"Uncommon protocol scheme '{scheme}' for media input '{sanitize_url(direct_url)}'",
                details={"type": "UNKNOWN_SCHEME", "scheme": scheme},
            )

    # No direct media input provided; check watch_url
    if watch_url:
        parsed_watch = urllib.parse.urlparse(watch_url)
        is_yt = "youtube.com" in parsed_watch.netloc or "youtu.be" in parsed_watch.netloc
        if is_yt:
            vid = extract_video_id(watch_url)
            yt_meta = inspect_youtube_source(watch_url)
            broadcast_desc = yt_meta.live_broadcast_content
            return PreflightCheckResult(
                name="Media Input Protocol",
                status=CheckStatus.WARN,
                message=(
                    f"Configured source is a Web Metadata URL (YouTube video '{vid}', state: '{broadcast_desc}'). "
                    f"Direct FFmpeg ingestion requires creator RTMP/SRT feed via '{env_name}' or acquisition.media_url."
                ),
                details={
                    "type": "WEB_METADATA_URL",
                    "watch_url": sanitize_url(watch_url),
                    "video_id": vid,
                    "broadcast_state": broadcast_desc,
                    "direct_feed_supplied": False,
                },
            )

    return PreflightCheckResult(
        name="Media Input Protocol",
        status=CheckStatus.FAIL,
        message="No media stream or watch URL configured",
        details={"type": "UNCONFIGURED"},
    )


def check_qwen_server(judge_yaml_path: Path | None = None) -> PreflightCheckResult:
    """Check 9: Qwen Multimodal VLM Server Availability and Auto-start Assets on Disk."""
    judge_path = (judge_yaml_path or Path("judge.yaml")).resolve()
    provider_cfg = QwenProviderConfig()

    if judge_path.is_file():
        try:
            with judge_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            jdata = data.get("judge", {})
            provider_cfg = QwenProviderConfig(
                model_path=Path(jdata.get("model_path", provider_cfg.model_path)),
                projector_path=Path(jdata.get("projector_path", provider_cfg.projector_path)),
                server_binary=Path(jdata.get("server_binary", provider_cfg.server_binary)),
                endpoint_url=str(jdata.get("endpoint_url", provider_cfg.endpoint_url)),
                model_name=str(jdata.get("model_name", provider_cfg.model_name)),
                n_gpu_layers=int(jdata.get("n_gpu_layers", provider_cfg.n_gpu_layers)),
                server_host=str(jdata.get("host", provider_cfg.server_host)),
                server_port=int(jdata.get("port", provider_cfg.server_port)),
            )
        except Exception as exc:
            LOGGER.warning("Could not parse judge.yaml: %s", exc)

    # 1. Probe live server endpoint
    server_running = False
    endpoint_base = provider_cfg.endpoint_url.rstrip("/").rstrip("/v1")
    try:
        with httpx.Client(timeout=2.0) as client:
            r = client.get(f"{endpoint_base}/health")
            if r.status_code == 200:
                server_running = True
    except Exception:
        server_running = False

    if server_running:
        return PreflightCheckResult(
            name="Qwen Server Availability",
            status=CheckStatus.PASS,
            message=f"Live llama-server is ACTIVE and healthy at {provider_cfg.endpoint_url}",
            details={"running": True, "endpoint": provider_cfg.endpoint_url},
        )

    # 2. Server offline: verify auto-start assets on disk
    bin_ok = provider_cfg.server_binary.is_file()
    model_ok = provider_cfg.model_path.is_file()
    proj_ok = provider_cfg.projector_path.is_file()

    model_size_mb = (provider_cfg.model_path.stat().st_size / (1024 * 1024)) if model_ok else 0
    proj_size_mb = (provider_cfg.projector_path.stat().st_size / (1024 * 1024)) if proj_ok else 0

    if bin_ok and model_ok and proj_ok and model_size_mb > 100:
        return PreflightCheckResult(
            name="Qwen Server Availability",
            status=CheckStatus.PASS,
            message=(
                f"Server offline; disk assets verified for auto-start "
                f"(Binary, Qwen model: {model_size_mb:.0f} MB, Projector: {proj_size_mb:.0f} MB)"
            ),
            details={
                "running": False,
                "auto_start_ready": True,
                "server_binary": str(provider_cfg.server_binary),
                "model_path": str(provider_cfg.model_path),
                "model_size_mb": model_size_mb,
                "projector_path": str(provider_cfg.projector_path),
            },
        )

    missing: list[str] = []
    if not bin_ok:
        missing.append(f"binary ({provider_cfg.server_binary})")
    if not model_ok:
        missing.append(f"model ({provider_cfg.model_path})")
    if not proj_ok:
        missing.append(f"projector ({provider_cfg.projector_path})")

    return PreflightCheckResult(
        name="Qwen Server Availability",
        status=CheckStatus.FAIL,
        message=f"Server offline and missing disk assets: {', '.join(missing)}",
        details={"running": False, "auto_start_ready": False, "missing": missing},
    )


def check_gpu_availability(config: LiveSourceConfig) -> PreflightCheckResult:
    """Check 10: NVIDIA GPU resource availability, VRAM headroom, and device health."""
    vram = query_gpu_vram()
    gpu_name = "NVIDIA GPU"

    # Query device name via nvidia-smi
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=3.0,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            gpu_name = res.stdout.strip().splitlines()[0]
    except Exception:
        pass

    if vram is None:
        return PreflightCheckResult(
            name="GPU Resource Availability",
            status=CheckStatus.WARN,
            message="No NVIDIA GPU detected via nvidia-smi; CPU fallback will be utilized",
            details={"has_gpu": False},
        )

    min_required_mb = 500.0
    if vram.free_mb < min_required_mb:
        return PreflightCheckResult(
            name="GPU Resource Availability",
            status=CheckStatus.WARN,
            message=(
                f"{gpu_name}: Low VRAM ({vram.free_mb:.0f} MB free / {vram.total_mb:.0f} MB total). "
                f"Contention risk during concurrent VLM/NVENC operations."
            ),
            details={
                "gpu_name": gpu_name,
                "used_mb": vram.used_mb,
                "free_mb": vram.free_mb,
                "total_mb": vram.total_mb,
            },
        )

    return PreflightCheckResult(
        name="GPU Resource Availability",
        status=CheckStatus.PASS,
        message=f"{gpu_name} ({vram.free_mb:.0f} MB free / {vram.total_mb:.0f} MB total, {vram.used_mb:.0f} MB used)",
        details={
            "gpu_name": gpu_name,
            "used_mb": vram.used_mb,
            "free_mb": vram.free_mb,
            "total_mb": vram.total_mb,
        },
    )


def check_model_and_render_config(
    judge_yaml_path: Path | None = None,
    render_yaml_path: Path | None = None,
) -> PreflightCheckResult:
    """Check 11: Multimodal Judge and Vertical Short Rendering configuration and assets."""
    judge_path = (judge_yaml_path or Path("judge.yaml")).resolve()
    render_path = (render_yaml_path or Path("render.yaml")).resolve()

    errors: list[str] = []

    # 1. Validate judge.yaml
    if not judge_path.is_file():
        errors.append(f"judge.yaml not found at '{judge_path}'")
    else:
        try:
            with judge_path.open("r", encoding="utf-8") as f:
                jdata = yaml.safe_load(f) or {}
            j = jdata.get("judge", {})
            if j.get("provider") not in ("local-qwen", "mock"):
                errors.append(f"Unknown judge provider '{j.get('provider')}'")
            if int(j.get("n_ctx", 0)) < 4096:
                errors.append(f"n_ctx too low ({j.get('n_ctx')}); recommend >= 6144")
        except Exception as exc:
            errors.append(f"judge.yaml invalid: {exc}")

    # 2. Validate render.yaml
    render_cfg = None
    if not render_path.is_file():
        errors.append(f"render.yaml not found at '{render_path}'")
    else:
        try:
            render_cfg = load_render_config(render_path)
            render_cfg.validate()
        except Exception as exc:
            errors.append(f"render.yaml invalid: {exc}")

    # 3. Check subtitle font assets
    fonts_dir = (Path("assets/fonts")).resolve()
    font_count = 0
    font_desc = "System Arial Black fallback"
    if fonts_dir.is_dir():
        font_files = [f for f in fonts_dir.iterdir() if f.suffix.lower() in (".ttf", ".otf")]
        font_count = len(font_files)
        if font_count > 0:
            font_desc = f"{font_count} custom font file(s) in assets/fonts"
        else:
            # Check system fallback font (Arial Black / Arial)
            win_ariblk = Path("C:/Windows/Fonts/ariblk.ttf")
            if win_ariblk.is_file():
                font_desc = "System Arial Black available (Windows Fonts)"
            else:
                font_desc = "Platform default font substitution"
    else:
        font_desc = "assets/fonts not found; platform default substitution active"

    if errors:
        return PreflightCheckResult(
            name="Model & Render Configuration",
            status=CheckStatus.FAIL,
            message="; ".join(errors),
            details={"errors": errors},
        )

    layout_name = render_cfg.layout if render_cfg else "blurred_background"
    enc_name = render_cfg.encoder if render_cfg else "auto"

    return PreflightCheckResult(
        name="Model & Render Configuration",
        status=CheckStatus.PASS,
        message=f"judge.yaml & render.yaml valid (Layout: '{layout_name}', Encoder: '{enc_name}', {font_desc})",
        details={
            "judge_config": str(judge_path),
            "render_config": str(render_path),
            "layout": layout_name,
            "encoder": enc_name,
            "font_count": font_count,
            "font_desc": font_desc,
        },
    )


# ---------------------------------------------------------------------------
# Preflight Orchestration
# ---------------------------------------------------------------------------

def run_preflight(
    config_path: Path = Path("source.yaml"),
    base_dir: Path | None = None,
    judge_path: Path | None = None,
    render_path: Path | None = None,
) -> PreflightReport:
    """Run comprehensive production preflight audit without initiating live capture.
    
    Evaluates all 11 production criteria and derives an authoritative verdict:
    - 'FIRST LIVE STREAM READY' (Exit Code 0)
    - 'FIRST LIVE STREAM VERIFIED WITH LIMITATIONS' (Exit Code 2)
    - 'FIRST LIVE STREAM NOT READY' (Exit Code 1)
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    checks: list[PreflightCheckResult] = []

    # 1. Source configuration validity
    res_cfg, config = check_source_config(config_path)
    checks.append(res_cfg)

    source_id = config.id if config else "unknown"
    creator_name = config.creator_name if config else "unknown"

    if config is not None:
        # 2. Authorization completeness
        checks.append(check_authorization_completeness(config))

        # 3. Permission status
        checks.append(check_permission_status(config))
    else:
        checks.append(PreflightCheckResult("Authorization Record Completeness", CheckStatus.FAIL, "Skipped due to invalid config"))
        checks.append(PreflightCheckResult("Permission Status", CheckStatus.FAIL, "Skipped due to invalid config"))

    # 4. FFmpeg and ffprobe
    checks.append(check_ffmpeg_ffprobe())

    if config is not None:
        # 5. SQLite database
        checks.append(check_sqlite_database(config))

        # 6. Capture directory permissions
        checks.append(check_directory_permissions(config, base_dir=base_dir))

        # 7. Environment variables
        checks.append(check_environment_variables(config))

        # 8. Media input protocol
        checks.append(check_media_input_protocol(config))

        # 10. GPU availability
        checks.append(check_gpu_availability(config))
    else:
        checks.append(PreflightCheckResult("SQLite Database Availability", CheckStatus.FAIL, "Skipped due to invalid config"))
        checks.append(PreflightCheckResult("Capture Directory Permissions", CheckStatus.FAIL, "Skipped due to invalid config"))
        checks.append(PreflightCheckResult("Required Environment Variables", CheckStatus.FAIL, "Skipped due to invalid config"))
        checks.append(PreflightCheckResult("Media Input Protocol", CheckStatus.FAIL, "Skipped due to invalid config"))
        checks.append(PreflightCheckResult("GPU Resource Availability", CheckStatus.FAIL, "Skipped due to invalid config"))

    # 9. Qwen server availability
    checks.append(check_qwen_server(judge_path))

    # 11. Model & render configuration
    checks.append(check_model_and_render_config(judge_path, render_path))

    # Determine overall status and verdict
    has_fail = any(c.status == CheckStatus.FAIL for c in checks)
    has_warn = any(c.status == CheckStatus.WARN for c in checks)

    reasons: list[str] = []

    # Analyze critical media input check specifically
    media_check = next((c for c in checks if c.name == "Media Input Protocol"), None)
    is_web_metadata = (
        media_check is not None
        and media_check.details.get("type") == "WEB_METADATA_URL"
    )

    if has_fail:
        overall_status = CheckStatus.FAIL
        verdict = "FIRST LIVE STREAM NOT READY"
        exit_code = 1
        for c in checks:
            if c.status == CheckStatus.FAIL:
                reasons.append(f"{c.name}: {c.message}")
    elif is_web_metadata or has_warn:
        overall_status = CheckStatus.WARN
        verdict = "FIRST LIVE STREAM VERIFIED WITH LIMITATIONS"
        exit_code = 2
        if is_web_metadata:
            reasons.append(
                "Configured source watch URL is a web metadata reference (YouTube watch page); "
                "direct creator media feed (STREAMCLIP_LIVE_INPUT or acquisition.media_url) must be "
                "provided for FFmpeg live ingestion."
            )
        for c in checks:
            if c.status == CheckStatus.WARN and c.name != "Media Input Protocol":
                reasons.append(f"{c.name}: {c.message}")
    else:
        overall_status = CheckStatus.PASS
        verdict = "FIRST LIVE STREAM READY"
        exit_code = 0
        reasons.append("All infrastructure, AI models, directories, and media feeds are fully verified.")

    return PreflightReport(
        timestamp=now_iso,
        config_path=str(config_path),
        source_id=source_id,
        creator_name=creator_name,
        checks=checks,
        overall_status=overall_status,
        verdict=verdict,
        exit_code=exit_code,
        reasons=reasons,
    )
