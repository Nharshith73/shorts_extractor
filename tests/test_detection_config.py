from __future__ import annotations

from pathlib import Path

from app.detection.config import load_detection_config, load_segments_dir


def test_detection_config_reads_shared_segments_dir_without_source(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
source: ""
segments_dir: data/segments
detection:
  warmup_seconds: 15
  weights:
    motion: 2.5
""",
        encoding="utf-8",
    )
    config = load_detection_config(config_path)
    assert config.warmup_seconds == 15
    assert config.weights["motion"] == 2.5
    assert config.scale_floors["audio_rms_db"] == 0.5
    assert config.relative_scale_floor == 0.01
    assert load_segments_dir(config_path) == tmp_path / "data" / "segments"
