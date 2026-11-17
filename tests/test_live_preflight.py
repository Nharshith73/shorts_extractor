"""Unit and Regression Tests for StreamClip AI Production Preflight (Stage 4C.2).

Validates:
1. Preflight on default source.yaml (verified with limitations).
2. Preflight with direct media stream supplied (stream ready).
3. Missing or invalid source configuration handling.
4. Authorization record completeness and unpermitted gate rejection.
5. FFmpeg and ffprobe binary detection and hardware acceleration probe.
6. SQLite database WAL mode persistence and directory permission verification.
7. Secret and stream key masking / sanitization.
8. Low GPU VRAM warning behavior.
9. Local Qwen VLM offline auto-start asset validation.
10. CLI invocation and JSON report structure.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import MagicMock, patch
import pytest
import yaml

from app.live.config import LiveSourceConfig, load_source_config
from app.live.gpu import VramSnapshot
from app.live.preflight import (
    CheckStatus,
    PreflightCheckResult,
    PreflightReport,
    check_authorization_completeness,
    check_directory_permissions,
    check_environment_variables,
    check_ffmpeg_ffprobe,
    check_gpu_availability,
    check_media_input_protocol,
    check_model_and_render_config,
    check_permission_status,
    check_qwen_server,
    check_source_config,
    check_sqlite_database,
    mask_secret,
    run_preflight,
    sanitize_url,
)


# ---------------------------------------------------------------------------
# Test 1: Preflight on Default source.yaml (No direct feed provided)
# ---------------------------------------------------------------------------
def test_preflight_default_source_yaml() -> None:
    # Ensure STREAMCLIP_LIVE_INPUT is not set
    env_backup = os.environ.pop("STREAMCLIP_LIVE_INPUT", None)
    try:
        report = run_preflight(Path("source.yaml"))
        assert len(report.checks) == 11
        assert report.source_id == "primary_creator"
        assert report.overall_status == CheckStatus.WARN
        assert report.verdict == "FIRST LIVE STREAM VERIFIED WITH LIMITATIONS"
        assert report.exit_code == 2

        # Verify critical system checks all passed
        check_dict = {c.name: c for c in report.checks}
        assert check_dict["Source Configuration Validity"].status == CheckStatus.PASS
        assert check_dict["Authorization Record Completeness"].status == CheckStatus.PASS
        assert check_dict["Permission Status"].status == CheckStatus.PASS
        assert check_dict["FFmpeg & ffprobe Availability"].status == CheckStatus.PASS
        assert check_dict["SQLite Database Availability"].status == CheckStatus.PASS
        assert check_dict["Capture Directory Permissions"].status == CheckStatus.PASS
        assert check_dict["GPU Resource Availability"].status == CheckStatus.PASS
        assert check_dict["Qwen Server Availability"].status == CheckStatus.PASS
        assert check_dict["Model & Render Configuration"].status == CheckStatus.PASS

        # Media Input Protocol should be WARN because watch_url is a web metadata URL
        assert check_dict["Media Input Protocol"].status == CheckStatus.WARN
        assert check_dict["Media Input Protocol"].details["type"] == "WEB_METADATA_URL"
    finally:
        if env_backup is not None:
            os.environ["STREAMCLIP_LIVE_INPUT"] = env_backup


# ---------------------------------------------------------------------------
# Test 2: Preflight with Authorized Direct Media Stream
# ---------------------------------------------------------------------------
def test_preflight_with_authorized_direct_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STREAMCLIP_LIVE_INPUT", "rtmp://127.0.0.1:1935/live/stream_test")
    report = run_preflight(Path("source.yaml"))

    check_dict = {c.name: c for c in report.checks}
    assert check_dict["Media Input Protocol"].status == CheckStatus.PASS
    assert check_dict["Media Input Protocol"].details["type"] == "DIRECT_MEDIA_INPUT"
    assert check_dict["Required Environment Variables"].status == CheckStatus.PASS

    assert report.overall_status == CheckStatus.PASS
    assert report.verdict == "FIRST LIVE STREAM READY"
    assert report.exit_code == 0


# ---------------------------------------------------------------------------
# Test 3: Missing Configuration File
# ---------------------------------------------------------------------------
def test_preflight_missing_config_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "non_existent_source.yaml"
    report = run_preflight(missing_path)

    assert report.overall_status == CheckStatus.FAIL
    assert report.verdict == "FIRST LIVE STREAM NOT READY"
    assert report.exit_code == 1

    cfg_check = next(c for c in report.checks if c.name == "Source Configuration Validity")
    assert cfg_check.status == CheckStatus.FAIL
    assert "not found" in cfg_check.message.lower()


# ---------------------------------------------------------------------------
# Test 4: Invalid Configuration Schema / Broken YAML
# ---------------------------------------------------------------------------
def test_preflight_invalid_yaml(tmp_path: Path) -> None:
    bad_cfg = tmp_path / "bad_source.yaml"
    bad_cfg.write_text("invalid_root_key: [1, 2, 3}", encoding="utf-8")  # YAML syntax error
    report = run_preflight(bad_cfg)

    assert report.overall_status == CheckStatus.FAIL
    assert report.exit_code == 1
    cfg_check = next(c for c in report.checks if c.name == "Source Configuration Validity")
    assert cfg_check.status == CheckStatus.FAIL


# ---------------------------------------------------------------------------
# Test 5: Incomplete Authorization Record (Missing Reference)
# ---------------------------------------------------------------------------
def test_preflight_missing_permission_reference(tmp_path: Path) -> None:
    cfg_data = {
        "source": {
            "id": "creator_test",
            "authorization": {
                "permitted": True,
                "capture": True,
                "clipping": True,
                "editing": True,
                "publishing": True,
                "permission_reference": "",  # Missing reference
            },
        },
    }
    cfg_file = tmp_path / "source.yaml"
    with cfg_file.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_data, f)

    report = run_preflight(cfg_file)
    assert report.overall_status == CheckStatus.FAIL
    assert report.exit_code == 1

    auth_comp = next(c for c in report.checks if c.name == "Authorization Record Completeness")
    assert auth_comp.status == CheckStatus.FAIL
    assert "permission_reference" in auth_comp.message


# ---------------------------------------------------------------------------
# Test 6: Unpermitted Permission Gates
# ---------------------------------------------------------------------------
def test_preflight_unpermitted_gates(tmp_path: Path) -> None:
    cfg_data = {
        "source": {
            "id": "creator_test",
            "authorization": {
                "permitted": True,
                "capture": True,
                "clipping": False,  # Blocked
                "editing": True,
                "publishing": False,  # Blocked
                "permission_reference": "ref-123",
            },
        },
    }
    cfg_file = tmp_path / "source.yaml"
    with cfg_file.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_data, f)

    report = run_preflight(cfg_file)
    assert report.overall_status == CheckStatus.FAIL
    assert report.exit_code == 1

    perm_check = next(c for c in report.checks if c.name == "Permission Status")
    assert perm_check.status == CheckStatus.FAIL
    assert "clipping" in perm_check.message
    assert "publishing" in perm_check.message


# ---------------------------------------------------------------------------
# Test 7: Missing FFmpeg / ffprobe Binaries
# ---------------------------------------------------------------------------
def test_preflight_missing_ffmpeg() -> None:
    with patch("shutil.which", return_value=None):
        res = check_ffmpeg_ffprobe()
        assert res.status == CheckStatus.FAIL
        assert "not found" in res.message.lower()


# ---------------------------------------------------------------------------
# Test 8: Secret Sanitization & Masking
# ---------------------------------------------------------------------------
def test_preflight_secret_sanitization() -> None:
    # 1. RTMP stream key masking
    url = "rtmp://live.stream.server:1935/live/live_secret_auth_token_999"
    sanitized = sanitize_url(url)
    assert "live_secret_auth_token_999" not in sanitized
    assert "rtmp://live.stream.server:1935/live/***" == sanitized

    # 2. Signed URL query parameter masking
    signed_url = "https://ingest.cdn.net/hls/master.m3u8?token=SECRET_JWT_SIGNATURE&expires=1234567"
    sanitized_signed = sanitize_url(signed_url)
    assert "SECRET_JWT_SIGNATURE" not in sanitized_signed
    assert "token=" not in sanitized_signed
    assert sanitized_signed.endswith("?***")

    # 3. HTTP basic auth user:pass masking
    auth_url = "https://admin:super_secret_pass@stream.provider.com/feed.mp4"
    sanitized_auth = sanitize_url(auth_url)
    assert "super_secret_pass" not in sanitized_auth
    assert "***@stream.provider.com" in sanitized_auth

    # 4. API key masking
    assert mask_secret("") == "(unset)"
    assert mask_secret("short") == "***"
    assert mask_secret("AIzaSyAbCdEfGhIjKlMnOpQrStUvWxYz") == "AIza...WxYz"


# ---------------------------------------------------------------------------
# Test 9: Low GPU VRAM Warning
# ---------------------------------------------------------------------------
def test_preflight_low_vram_warning() -> None:
    fake_vram = VramSnapshot(used_mb=7900.0, free_mb=288.0, total_mb=8188.0)
    with patch("app.live.preflight.query_gpu_vram", return_value=fake_vram):
        config = load_source_config(Path("source.yaml"))
        res = check_gpu_availability(config)
        assert res.status == CheckStatus.WARN
        assert "Low VRAM" in res.message


# ---------------------------------------------------------------------------
# Test 10: Capture Directory Permissions Probe
# ---------------------------------------------------------------------------
def test_preflight_directory_permissions(tmp_path: Path) -> None:
    config = load_source_config(Path("source.yaml"))
    # Point capture to tmp_path
    config.capture.segments_dir = tmp_path / "segments"
    res = check_directory_permissions(config, base_dir=tmp_path)
    assert res.status == CheckStatus.PASS
    assert "verified across 5 live directories" in res.message


# ---------------------------------------------------------------------------
# Test 11: Text and JSON Report Formats
# ---------------------------------------------------------------------------
def test_preflight_report_serialization() -> None:
    report = run_preflight(Path("source.yaml"))
    text_rep = report.render_text()
    assert "STREAMCLIP AI — LIVE PRODUCTION PREFLIGHT REPORT" in text_rep
    assert "FINAL VERDICT:" in text_rep
    assert "OVERALL STATUS:" in text_rep

    dict_rep = report.to_dict()
    assert "timestamp" in dict_rep
    assert "verdict" in dict_rep
    assert "checks" in dict_rep
    assert len(dict_rep["checks"]) == 11

    # Verify JSON serializeable
    json_str = json.dumps(dict_rep)
    assert json_str.startswith("{")


# ---------------------------------------------------------------------------
# Test 12: CLI Execution via Subprocess
# ---------------------------------------------------------------------------
def test_preflight_cli_execution() -> None:
    res = subprocess.run(
        [sys.executable, "-m", "app.live", "preflight", "--config", "source.yaml", "--json"],
        capture_output=True,
        text=True,
        timeout=10.0,
        check=False,
    )
    # With default source.yaml, exit code should be 2 (VERIFIED WITH LIMITATIONS)
    assert res.returncode == 2
    # Verify stdout contains valid JSON
    # Note: logging could appear on stderr or stdout
    assert "FIRST LIVE STREAM VERIFIED WITH LIMITATIONS" in res.stdout
    data = json.loads(res.stdout[res.stdout.find("{"):res.stdout.rfind("}") + 1])
    assert data["verdict"] == "FIRST LIVE STREAM VERIFIED WITH LIMITATIONS"
    assert data["exit_code"] == 2
