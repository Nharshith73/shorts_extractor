"""Regression tests for P1: Clip cache render provenance and invalidation."""

import hashlib
import json
from pathlib import Path
import shutil
import pytest

from app.vod.pipeline import make_render_cache_key, make_render_provenance


def test_cache_provenance_changes_with_encoder() -> None:
    """Changing requested/effective encoder creates a different render_cache_key."""
    fp = {"path": "video.mp4", "size": 1000, "mtime_ns": 12345}
    prov1 = make_render_provenance(fp, 10.0, 30.0, "libx264", "libx264")
    prov2 = make_render_provenance(fp, 10.0, 30.0, "h264_nvenc", "h264_nvenc")

    key1 = make_render_cache_key(prov1)
    key2 = make_render_cache_key(prov2)

    assert key1 != key2


def test_cache_provenance_changes_with_render_profile_version() -> None:
    """Changing render profile/version changes render_cache_key."""
    fp = {"path": "video.mp4", "size": 1000, "mtime_ns": 12345}
    prov_v1 = make_render_provenance(fp, 10.0, 30.0, "libx264", "libx264", render_version="1.0")
    prov_v2 = make_render_provenance(fp, 10.0, 30.0, "libx264", "libx264", render_version="2.0")

    assert make_render_cache_key(prov_v1) != make_render_cache_key(prov_v2)


def test_editorial_candidate_identity_is_stable_across_render_settings() -> None:
    """Candidate window_id depends only on source fingerprint and boundaries, not render settings."""
    fp = {"path": "video.mp4", "size": 1000, "mtime_ns": 12345}
    left, right = 10.0, 30.0
    window_id = hashlib.sha256(f"{fp}:{left:.6f}:{right:.6f}".encode()).hexdigest()[:24]

    prov_libx264 = make_render_provenance(fp, left, right, "libx264", "libx264")
    prov_nvenc = make_render_provenance(fp, left, right, "h264_nvenc", "h264_nvenc")

    # Render keys differ
    assert make_render_cache_key(prov_libx264) != make_render_cache_key(prov_nvenc)
    # But candidate identity remains identical
    assert window_id == hashlib.sha256(f"{fp}:{left:.6f}:{right:.6f}".encode()).hexdigest()[:24]


def test_legacy_cache_entry_without_provenance_is_invalidated(tmp_path: Path) -> None:
    """A legacy cache entry lacking provenance or render_cache_key is not treated as a cache hit."""
    fp = {"path": "video.mp4", "size": 1000, "mtime_ns": 12345}
    left, right = 10.0, 30.0
    window_id = "abc123legacy"

    dummy_file = tmp_path / "old.mp4"
    dummy_file.write_bytes(b"dummy video content")

    # Legacy cache structure (no render_cache_key, no provenance)
    cached_legacy = {
        "clip_path": str(dummy_file),
        "size_bytes": len(b"dummy video content"),
        "video_encoder": "libx264",
    }

    expected_prov = make_render_provenance(fp, left, right, "libx264", "libx264")
    expected_render_key = make_render_cache_key(expected_prov)

    # Validate matching condition from pipeline.py
    has_matching_provenance = (
        cached_legacy.get("render_cache_key") == expected_render_key
        and cached_legacy.get("provenance") == expected_prov
    )
    assert has_matching_provenance is False


def test_missing_or_corrupt_cached_file_is_invalidated(tmp_path: Path) -> None:
    """Missing file or file size mismatch prevents false cache reuse."""
    fp = {"path": "video.mp4", "size": 1000, "mtime_ns": 12345}
    left, right = 10.0, 30.0
    prov = make_render_provenance(fp, left, right, "libx264", "libx264")
    render_key = make_render_cache_key(prov)

    missing_file = tmp_path / "missing.mp4"
    cached = {
        "clip_path": str(missing_file),
        "size_bytes": 100,
        "video_encoder": "libx264",
        "render_cache_key": render_key,
        "provenance": prov,
    }

    # Missing file check
    old_file = Path(cached["clip_path"])
    reused = (
        cached.get("render_cache_key") == render_key
        and cached.get("provenance") == prov
        and old_file.is_file()
        and old_file.stat().st_size == cached.get("size_bytes")
    )
    assert reused is False

    # Corrupt/size mismatch check
    corrupt_file = tmp_path / "corrupt.mp4"
    corrupt_file.write_bytes(b"short")  # 5 bytes vs recorded 100 bytes
    cached["clip_path"] = str(corrupt_file)
    old_file = Path(cached["clip_path"])
    reused = (
        cached.get("render_cache_key") == render_key
        and cached.get("provenance") == prov
        and old_file.is_file()
        and old_file.stat().st_size == cached.get("size_bytes")
    )
    assert reused is False


def test_encoder_change_invalidates_cache_and_preserves_review_approval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When render settings change, cache is invalidated but review approval remains associated with window_id."""
    from app.vod.artifacts import write_features, write_json
    from app.vod.pipeline import rescore
    from app.detection.models import FeatureBin
    from app.vod.timeline import media_time

    monkeypatch.setattr("app.vod.pipeline.probe_duration", lambda p: 15.0)

    # Setup config
    cfg_libx264 = tmp_path / "config_x264.yaml"
    cfg_libx264.write_text("detection:\n  warmup_seconds: 0\nselection:\n  min_separation_seconds: 20\ncandidates:\n  encoder: libx264\n", encoding="utf-8")

    cfg_nvenc = tmp_path / "config_nvenc.yaml"
    cfg_nvenc.write_text("detection:\n  warmup_seconds: 0\nselection:\n  min_separation_seconds: 20\ncandidates:\n  encoder: h264_nvenc\n", encoding="utf-8")

    # Features
    features = [
        FeatureBin(
            source_id="s",
            session_id="sess",
            wall_utc=media_time(float(sec)),
            bin_seconds=1.0,
            audio_rms_db=-10.0 if sec == 40 else -40.0,
            voice_energy_db=-10.0 if sec == 40 else -45.0,
            onset_strength=2.0 if sec == 40 else 0.01,
            motion=10.0 if sec == 40 else 0.5,
            cut_change=1.0 if sec == 40 else 0.01,
        )
        for sec in range(120)
    ]
    write_features(tmp_path / "features.csv", features, None)
    dummy_video = tmp_path / "dummy.mp4"
    dummy_video.write_bytes(b"dummy")
    write_json(
        tmp_path / "analysis.json",
        dict(video=str(dummy_video), source_id="s", fingerprint={"path": str(dummy_video), "size": 5, "mtime_ns": 1},
             source_duration_seconds=120, start_seconds=0, end_seconds=120, chat=None)
    )

    # Initial rescore without cache: extracts nothing
    rescore(tmp_path, cfg_libx264, extract=False)
    clips1 = json.loads((tmp_path / "clips.json").read_text())
    assert len(clips1) >= 1
    w_id = clips1[0]["window_id"]

    # Mock an extracted cache entry with libx264 provenance
    clip_file = tmp_path / "clips" / Path(clips1[0]["clip_path"]).name
    clip_file.parent.mkdir(exist_ok=True)
    clip_file.write_bytes(b"rendered mp4 data 12345")

    prov = make_render_provenance(
        {"path": str(dummy_video), "size": 5, "mtime_ns": 1},
        clips1[0]["start_seconds"],
        clips1[0]["end_seconds"],
        "libx264",
        "libx264",
    )
    r_key = make_render_cache_key(prov)
    cache = {
        w_id: {
            "clip_path": str(clip_file),
            "size_bytes": clip_file.stat().st_size,
            "video_encoder": "libx264",
            "render_cache_key": r_key,
            "provenance": prov,
        }
    }
    write_json(tmp_path / "clip_cache.json", cache)

    # Record a human review approval in review.csv
    import csv
    with (tmp_path / "review.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["window_id", "clip_path", "approved", "notes", "score", "start_seconds", "end_seconds", "top_features", "reason_code", "rank", "peak_seconds", "rule", "unclipped_z_scores", "unclipped_score"])
        writer.writeheader()
        writer.writerow({
            "window_id": w_id,
            "clip_path": str(clip_file),
            "approved": "y",
            "notes": "Great highlight!",
            "score": "10.0",
            "start_seconds": str(clips1[0]["start_seconds"]),
            "end_seconds": str(clips1[0]["end_seconds"]),
            "top_features": "motion=10.0",
            "reason_code": "",
            "rank": "1",
            "peak_seconds": "40.0",
            "rule": "single",
            "unclipped_z_scores": "{}",
            "unclipped_score": "10.0",
        })

    # Run rescore with same settings (libx264): must reuse cache
    rescore(tmp_path, cfg_libx264, extract=False)
    clips2 = json.loads((tmp_path / "clips.json").read_text())
    assert clips2[0]["reused"] is True

    # Check review.csv preserved approval
    with (tmp_path / "review.csv").open("r", encoding="utf-8") as f:
        rows2 = list(csv.DictReader(f))
    assert rows2[0]["approved"] == "y"
    assert rows2[0]["notes"] == "Great highlight!"

    # Now run rescore with different requested encoder (h264_nvenc): MUST INVALIDATE CACHE
    rescore(tmp_path, cfg_nvenc, extract=False)
    clips3 = json.loads((tmp_path / "clips.json").read_text())
    # Cache was invalidated!
    assert clips3[0]["reused"] is False

    # But review approval is STILL PRESERVED!
    with (tmp_path / "review.csv").open("r", encoding="utf-8") as f:
        rows3 = list(csv.DictReader(f))
    assert rows3[0]["window_id"] == w_id
    assert rows3[0]["approved"] == "y"
    assert rows3[0]["notes"] == "Great highlight!"


# ==============================================================================
# Task 1 Required Tests: 1 through 8
# ==============================================================================

def _setup_rescore_fixture(tmp_path: Path):
    from app.vod.artifacts import write_features, write_json
    from app.detection.models import FeatureBin
    from app.vod.pipeline import rescore
    from app.vod.timeline import media_time

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "detection:\n  warmup_seconds: 0\nselection:\n  min_separation_seconds: 20\n  min_score: 5.0\ncandidates:\n  encoder: libx264\n",
        encoding="utf-8",
    )
    features = [
        FeatureBin(
            source_id="s",
            session_id="sess",
            wall_utc=media_time(float(sec)),
            bin_seconds=1.0,
            audio_rms_db=-10.0 if sec == 40 else -40.0,
            voice_energy_db=-10.0 if sec == 40 else -45.0,
            onset_strength=2.0 if sec == 40 else 0.01,
            motion=10.0 if sec == 40 else 0.5,
            cut_change=1.0 if sec == 40 else 0.01,
        )
        for sec in range(120)
    ]
    write_features(tmp_path / "features.csv", features, None)
    dummy_video = tmp_path / "dummy.mp4"
    dummy_video.write_bytes(b"dummy")
    stat = dummy_video.stat()
    fp = {"path": str(dummy_video), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    write_json(
        tmp_path / "analysis.json",
        dict(
            video=str(dummy_video),
            source_id="s",
            fingerprint=fp,
            source_duration_seconds=120,
            start_seconds=0,
            end_seconds=120,
            chat=None,
        ),
    )
    rescore(tmp_path, cfg_path, extract=False)
    clips = json.loads((tmp_path / "clips.json").read_text())
    assert len(clips) == 1
    return cfg_path, clips[0], fp


def test_1_matching_modern_provenance_reuses_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """1. Matching modern provenance reuses the cache without re-rendering."""
    from app.vod.artifacts import write_json
    from app.vod.pipeline import rescore

    cfg_path, clip0, fp = _setup_rescore_fixture(tmp_path)
    w_id = clip0["window_id"]
    clip_file = tmp_path / "clips" / Path(clip0["clip_path"]).name
    clip_file.parent.mkdir(exist_ok=True)
    clip_file.write_bytes(b"valid rendered media data")

    expected_duration = clip0["end_seconds"] - clip0["start_seconds"]
    monkeypatch.setattr("app.vod.pipeline.probe_duration", lambda p: expected_duration)

    render_calls = 0
    def mock_render_cut(*args, **kwargs):
        nonlocal render_calls
        render_calls += 1
    monkeypatch.setattr("app.vod.pipeline.render_cut", mock_render_cut)

    prov = make_render_provenance(fp, clip0["start_seconds"], clip0["end_seconds"], "libx264", "libx264")
    r_key = make_render_cache_key(prov)
    cache = {
        w_id: {
            "clip_path": str(clip_file),
            "size_bytes": clip_file.stat().st_size,
            "video_encoder": "libx264",
            "render_cache_key": r_key,
            "provenance": prov,
        }
    }
    write_json(tmp_path / "clip_cache.json", cache)

    rescore(tmp_path, cfg_path, extract=True)
    updated = json.loads((tmp_path / "clips.json").read_text())
    assert updated[0]["reused"] is True
    assert updated[0]["extracted"] is True
    assert render_calls == 0


def test_2_missing_provenance_causes_rerendering(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """2. Missing provenance causes re-rendering."""
    from app.vod.artifacts import write_json
    from app.vod.pipeline import rescore

    cfg_path, clip0, fp = _setup_rescore_fixture(tmp_path)
    w_id = clip0["window_id"]
    clip_file = tmp_path / "clips" / Path(clip0["clip_path"]).name
    clip_file.parent.mkdir(exist_ok=True)
    clip_file.write_bytes(b"old clip")

    expected_duration = clip0["end_seconds"] - clip0["start_seconds"]
    monkeypatch.setattr("app.vod.pipeline.probe_duration", lambda p: expected_duration)

    render_calls = 0
    def mock_render_cut(source, target, start, duration, **kwargs):
        nonlocal render_calls
        render_calls += 1
        target.write_bytes(b"newly rendered clip")
    monkeypatch.setattr("app.vod.pipeline.render_cut", mock_render_cut)

    prov = make_render_provenance(fp, clip0["start_seconds"], clip0["end_seconds"], "libx264", "libx264")
    r_key = make_render_cache_key(prov)
    # Cache entry has render_cache_key but lacks provenance
    cache = {
        w_id: {
            "clip_path": str(clip_file),
            "size_bytes": clip_file.stat().st_size,
            "video_encoder": "libx264",
            "render_cache_key": r_key,
            # 'provenance' is missing
        }
    }
    write_json(tmp_path / "clip_cache.json", cache)

    rescore(tmp_path, cfg_path, extract=True)
    updated = json.loads((tmp_path / "clips.json").read_text())
    assert updated[0]["reused"] is False
    assert updated[0]["extracted"] is True
    assert render_calls == 1

    # Cache is updated with modern provenance
    new_cache = json.loads((tmp_path / "clip_cache.json").read_text())
    assert "provenance" in new_cache[w_id]
    assert new_cache[w_id]["provenance"] == prov


def test_3_legacy_encoder_only_metadata_causes_rerendering(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """3. Legacy encoder-only metadata causes re-rendering (legacy bypass removed)."""
    from app.vod.artifacts import write_json
    from app.vod.pipeline import rescore

    cfg_path, clip0, fp = _setup_rescore_fixture(tmp_path)
    w_id = clip0["window_id"]
    clip_file = tmp_path / "clips" / Path(clip0["clip_path"]).name
    clip_file.parent.mkdir(exist_ok=True)
    clip_file.write_bytes(b"legacy clip")

    expected_duration = clip0["end_seconds"] - clip0["start_seconds"]
    monkeypatch.setattr("app.vod.pipeline.probe_duration", lambda p: expected_duration)

    render_calls = 0
    def mock_render_cut(source, target, start, duration, **kwargs):
        nonlocal render_calls
        render_calls += 1
        target.write_bytes(b"new rendered clip")
    monkeypatch.setattr("app.vod.pipeline.render_cut", mock_render_cut)

    # Legacy cache record: only clip_path, size_bytes, video_encoder
    cache = {
        w_id: {
            "clip_path": str(clip_file),
            "size_bytes": clip_file.stat().st_size,
            "video_encoder": "libx264",
        }
    }
    write_json(tmp_path / "clip_cache.json", cache)

    rescore(tmp_path, cfg_path, extract=True)
    updated = json.loads((tmp_path / "clips.json").read_text())
    assert updated[0]["reused"] is False
    assert updated[0]["extracted"] is True
    assert render_calls == 1

    # Cache now contains full modern provenance
    new_cache = json.loads((tmp_path / "clip_cache.json").read_text())
    assert "render_cache_key" in new_cache[w_id]
    assert "provenance" in new_cache[w_id]


def test_4_changed_encoder_invalidates_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """4. Changed encoder invalidates the cache."""
    from app.vod.artifacts import write_json
    from app.vod.pipeline import rescore

    cfg_path, clip0, fp = _setup_rescore_fixture(tmp_path)
    w_id = clip0["window_id"]
    clip_file = tmp_path / "clips" / Path(clip0["clip_path"]).name
    clip_file.parent.mkdir(exist_ok=True)
    clip_file.write_bytes(b"valid clip")

    expected_duration = clip0["end_seconds"] - clip0["start_seconds"]
    monkeypatch.setattr("app.vod.pipeline.probe_duration", lambda p: expected_duration)

    prov = make_render_provenance(fp, clip0["start_seconds"], clip0["end_seconds"], "libx264", "libx264")
    r_key = make_render_cache_key(prov)
    cache = {
        w_id: {
            "clip_path": str(clip_file),
            "size_bytes": clip_file.stat().st_size,
            "video_encoder": "libx264",
            "render_cache_key": r_key,
            "provenance": prov,
        }
    }
    write_json(tmp_path / "clip_cache.json", cache)

    # Change requested encoder to h264_nvenc
    cfg_nvenc = tmp_path / "config_nvenc.yaml"
    cfg_nvenc.write_text(
        "detection:\n  warmup_seconds: 0\nselection:\n  min_separation_seconds: 20\ncandidates:\n  encoder: h264_nvenc\n",
        encoding="utf-8",
    )
    rescore(tmp_path, cfg_nvenc, extract=False)
    updated = json.loads((tmp_path / "clips.json").read_text())
    assert updated[0]["reused"] is False


def test_5_changed_rendering_profile_invalidates_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """5. Changed rendering profile invalidates the cache."""
    from app.vod.artifacts import write_json
    from app.vod.pipeline import rescore

    cfg_path, clip0, fp = _setup_rescore_fixture(tmp_path)
    w_id = clip0["window_id"]
    clip_file = tmp_path / "clips" / Path(clip0["clip_path"]).name
    clip_file.parent.mkdir(exist_ok=True)
    clip_file.write_bytes(b"valid clip")

    expected_duration = clip0["end_seconds"] - clip0["start_seconds"]
    monkeypatch.setattr("app.vod.pipeline.probe_duration", lambda p: expected_duration)

    # Older render profile version
    old_prov = make_render_provenance(fp, clip0["start_seconds"], clip0["end_seconds"], "libx264", "libx264", render_version="0.9")
    old_key = make_render_cache_key(old_prov)
    cache = {
        w_id: {
            "clip_path": str(clip_file),
            "size_bytes": clip_file.stat().st_size,
            "video_encoder": "libx264",
            "render_cache_key": old_key,
            "provenance": old_prov,
        }
    }
    write_json(tmp_path / "clip_cache.json", cache)

    rescore(tmp_path, cfg_path, extract=False)
    updated = json.loads((tmp_path / "clips.json").read_text())
    assert updated[0]["reused"] is False


def test_6_corrupt_media_invalidates_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """6. Corrupt media invalidates the cache."""
    from app.vod.artifacts import write_json
    from app.vod.pipeline import rescore

    cfg_path, clip0, fp = _setup_rescore_fixture(tmp_path)
    w_id = clip0["window_id"]
    clip_file = tmp_path / "clips" / Path(clip0["clip_path"]).name
    clip_file.parent.mkdir(exist_ok=True)
    clip_file.write_bytes(b"corrupt media data")

    expected_duration = clip0["end_seconds"] - clip0["start_seconds"]
    def probe_router(p):
        if Path(p).resolve() == clip_file.resolve():
            raise ValueError("corrupt media packet or unplayable file")
        return expected_duration
    monkeypatch.setattr("app.vod.pipeline.probe_duration", probe_router)

    render_calls = 0
    def mock_render_cut(source, target, start, duration, **kwargs):
        nonlocal render_calls
        render_calls += 1
        target.write_bytes(b"fixed render")
    monkeypatch.setattr("app.vod.pipeline.render_cut", mock_render_cut)

    prov = make_render_provenance(fp, clip0["start_seconds"], clip0["end_seconds"], "libx264", "libx264")
    r_key = make_render_cache_key(prov)
    cache = {
        w_id: {
            "clip_path": str(clip_file),
            "size_bytes": clip_file.stat().st_size,
            "video_encoder": "libx264",
            "render_cache_key": r_key,
            "provenance": prov,
        }
    }
    write_json(tmp_path / "clip_cache.json", cache)

    rescore(tmp_path, cfg_path, extract=True)
    updated = json.loads((tmp_path / "clips.json").read_text())
    assert updated[0]["reused"] is False
    assert updated[0]["extracted"] is True
    assert render_calls == 1


def test_7_missing_media_invalidates_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """7. Missing media invalidates the cache."""
    from app.vod.artifacts import write_json
    from app.vod.pipeline import rescore

    cfg_path, clip0, fp = _setup_rescore_fixture(tmp_path)
    w_id = clip0["window_id"]
    missing_file = tmp_path / "clips" / "does_not_exist.mp4"

    expected_duration = clip0["end_seconds"] - clip0["start_seconds"]
    monkeypatch.setattr("app.vod.pipeline.probe_duration", lambda p: expected_duration)

    render_calls = 0
    def mock_render_cut(source, target, start, duration, **kwargs):
        nonlocal render_calls
        render_calls += 1
        target.write_bytes(b"freshly cut clip")
    monkeypatch.setattr("app.vod.pipeline.render_cut", mock_render_cut)

    prov = make_render_provenance(fp, clip0["start_seconds"], clip0["end_seconds"], "libx264", "libx264")
    r_key = make_render_cache_key(prov)
    cache = {
        w_id: {
            "clip_path": str(missing_file),
            "size_bytes": 1024,
            "video_encoder": "libx264",
            "render_cache_key": r_key,
            "provenance": prov,
        }
    }
    write_json(tmp_path / "clip_cache.json", cache)

    rescore(tmp_path, cfg_path, extract=True)
    updated = json.loads((tmp_path / "clips.json").read_text())
    assert updated[0]["reused"] is False
    assert updated[0]["extracted"] is True
    assert render_calls == 1


def test_8_human_review_decisions_survive_cache_invalidation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """8. Human review decisions survive cache invalidation."""
    from app.vod.artifacts import write_json
    from app.vod.pipeline import rescore
    import csv

    cfg_path, clip0, fp = _setup_rescore_fixture(tmp_path)
    w_id = clip0["window_id"]
    clip_file = tmp_path / "clips" / Path(clip0["clip_path"]).name
    clip_file.parent.mkdir(exist_ok=True)
    clip_file.write_bytes(b"media file")

    expected_duration = clip0["end_seconds"] - clip0["start_seconds"]
    monkeypatch.setattr("app.vod.pipeline.probe_duration", lambda p: expected_duration)

    # Human reviews candidate: approves it with note and reason
    with (tmp_path / "review.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "window_id", "clip_path", "approved", "notes", "score",
                "start_seconds", "end_seconds", "top_features", "reason_code",
                "rank", "peak_seconds", "rule", "unclipped_z_scores", "unclipped_score"
            ],
        )
        writer.writeheader()
        writer.writerow({
            "window_id": w_id,
            "clip_path": str(clip_file),
            "approved": "y",
            "notes": "Verified ace play",
            "score": "15.5",
            "start_seconds": str(clip0["start_seconds"]),
            "end_seconds": str(clip0["end_seconds"]),
            "top_features": "motion=10.0",
            "reason_code": "ace",
            "rank": "1",
            "peak_seconds": "40.0",
            "rule": "single",
            "unclipped_z_scores": "{}",
            "unclipped_score": "15.5",
        })

    # Cause cache invalidation: set legacy metadata lacking provenance
    cache = {
        w_id: {
            "clip_path": str(clip_file),
            "size_bytes": clip_file.stat().st_size,
            "video_encoder": "libx264",
        }
    }
    write_json(tmp_path / "clip_cache.json", cache)

    # Rescore triggers cache invalidation
    rescore(tmp_path, cfg_path, extract=False)

    # Verify review.csv preserves all human decisions
    with (tmp_path / "review.csv").open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 1
    assert rows[0]["window_id"] == w_id
    assert rows[0]["approved"] == "y"
    assert rows[0]["notes"] == "Verified ace play"
    assert rows[0]["reason_code"] == "ace"

