"""Regression tests for Task 2: Render configuration portability across platforms."""

from pathlib import Path
import pytest

from app.rendering.config import RenderConfig, load_render_config


def test_committed_render_yaml_loads_portable_model_name() -> None:
    """The default render.yaml committed in the repo root uses whisper_model: small."""
    root_render_yaml = Path(__file__).resolve().parents[1] / "render.yaml"
    assert root_render_yaml.is_file(), f"Missing render.yaml at {root_render_yaml}"
    config = load_render_config(root_render_yaml)
    assert config.whisper_model == "small"
    assert config.compute_type == "int8"
    assert config.device == "cpu"
    assert config.language == "auto"


def test_model_name_loading(tmp_path: Path) -> None:
    """Standard model names (small, base, tiny) load without filesystem path resolution."""
    for model_name in ("small", "base", "tiny", "medium", "large-v3"):
        cfg_file = tmp_path / f"render_{model_name}.yaml"
        cfg_file.write_text(f"render:\n  whisper_model: {model_name}\n", encoding="utf-8")
        loaded = load_render_config(cfg_file)
        assert loaded.whisper_model == model_name


def test_local_model_directory_loading_with_spaces(tmp_path: Path) -> None:
    """Existing local model directories (with spaces) are resolved cleanly."""
    model_dir = tmp_path / "My Models" / "custom-whisper-small"
    model_dir.mkdir(parents=True)

    # 1. Absolute path with spaces
    cfg_file = tmp_path / "render_abs.yaml"
    cfg_file.write_text(f'render:\n  whisper_model: "{model_dir.as_posix()}"\n', encoding="utf-8")
    loaded_abs = load_render_config(cfg_file)
    assert loaded_abs.whisper_model == str(model_dir.resolve())

    # 2. Relative path with spaces
    cfg_file_rel = tmp_path / "render_rel.yaml"
    cfg_file_rel.write_text('render:\n  whisper_model: "./My Models/custom-whisper-small"\n', encoding="utf-8")
    loaded_rel = load_render_config(cfg_file_rel)
    assert loaded_rel.whisper_model == str(model_dir.resolve())


def test_relative_font_directory_resolution_with_spaces(tmp_path: Path) -> None:
    """Relative fonts_dir (including spaces) resolves relative to the config file directory."""
    fonts_dir = tmp_path / "assets" / "custom typography"
    fonts_dir.mkdir(parents=True)

    cfg_file = tmp_path / "render.yaml"
    cfg_file.write_text('render:\n  fonts_dir: "assets/custom typography"\n', encoding="utf-8")
    loaded = load_render_config(cfg_file)
    assert loaded.fonts_dir == fonts_dir.resolve()


def test_unavailable_model_directory_raises_clear_error(tmp_path: Path) -> None:
    """Configuring a nonexistent local directory raises FileNotFoundError with the path."""
    # Nonexistent absolute path
    missing_abs = tmp_path / "does_not_exist" / "model"
    cfg_file_abs = tmp_path / "render_missing_abs.yaml"
    cfg_file_abs.write_text(f'render:\n  whisper_model: "{missing_abs.as_posix()}"\n', encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="configured whisper_model directory does not exist"):
        load_render_config(cfg_file_abs)

    # Nonexistent relative path
    cfg_file_rel = tmp_path / "render_missing_rel.yaml"
    cfg_file_rel.write_text('render:\n  whisper_model: "./nonexistent_dir/model"\n', encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="configured whisper_model directory does not exist"):
        load_render_config(cfg_file_rel)


def test_render_config_validate_checks_explicit_missing_paths(tmp_path: Path) -> None:
    """Direct instantiation of RenderConfig validates that explicit paths exist."""
    missing = tmp_path / "missing_dir"
    with pytest.raises(FileNotFoundError, match="configured whisper_model directory does not exist"):
        RenderConfig(whisper_model=str(missing.resolve())).validate()

    # Standard model names do not raise
    RenderConfig(whisper_model="small").validate()
