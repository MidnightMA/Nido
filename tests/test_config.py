"""Tests for configuration loading and validation."""

import tempfile
from pathlib import Path

from nido.config import Config, load_config


def test_default_config() -> None:
    config = Config()
    assert config.assistant.name == "Nido"
    assert config.assistant.language == "en"
    assert config.hotkey.key == "KEY_F9"
    assert config.hotkey.mode_threshold_ms == 300
    assert config.audio.sample_rate == 16000
    assert config.audio.channels == 1
    assert config.audio.max_seconds == 30
    assert config.audio.prebuffer_ms == 400
    assert config.ui.position == "top-right"
    assert config.realtime.enabled is True
    assert config.realtime.max_pending_commands == 8
    assert config.laya.enabled is True
    assert config.laya.max_candidates == 24
    assert config.desktop.max_steps == 24
    assert "browser" in config.apps
    assert config.system.allow_shutdown is False


def test_load_custom_toml() -> None:
    content = """
    [assistant]
    name = "CustomNido"
    language = "en"

    [hotkey]
    key = "KEY_F10"
    mode_threshold_ms = 250

    [audio]
    sample_rate = 48000
    max_seconds = 15
    prebuffer_ms = 500

    [ui]
    position = "bottom-left"

    [realtime]
    enabled = true
    max_pending_commands = 12

    [laya]
    enabled = true
    batch_size = 32
    max_candidates = 18
    shortlist_size = 12

    [desktop]
    max_steps = 16

    [apps]
    custom_editor = "nvim"

    [system]
    allow_shutdown = true

    # Obsolete section should be gracefully ignored
    [translation]
    enabled = true
    """
    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
        f.write(content)
        temp_path = f.name

    try:
        config = load_config(temp_path)
        assert config.assistant.name == "CustomNido"
        assert config.assistant.language == "en"
        assert config.hotkey.key == "KEY_F10"
        assert config.hotkey.mode_threshold_ms == 250
        assert config.audio.sample_rate == 48000
        assert config.audio.max_seconds == 15
        assert config.audio.prebuffer_ms == 500
        assert config.ui.position == "bottom-left"
        assert config.realtime.max_pending_commands == 12
        assert config.laya.batch_size == 32
        assert config.laya.max_candidates == 18
        assert config.laya.shortlist_size == 12
        assert config.desktop.max_steps == 16
        assert config.apps.get("custom_editor") == "nvim"
        assert config.system.allow_shutdown is True
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_load_nonexistent_file_returns_defaults() -> None:
    config = load_config("/non/existent/path/config.toml")
    assert config.assistant.name == "Nido"
    assert config.hotkey.key == "KEY_F9"
    assert "whisper" in config.stt.model_dir


def test_legacy_model_dir_auto_migrates() -> None:
    content = """
    [stt]
    model_dir = "~/.local/share/nido/models/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17"
    """
    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
        f.write(content)
        temp_path = f.name

    try:
        config = load_config(temp_path)
        assert "whisper" in config.stt.model_dir
    finally:
        Path(temp_path).unlink(missing_ok=True)

