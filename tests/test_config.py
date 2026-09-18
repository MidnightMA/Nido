"""Tests for configuration loading and validation."""

import tempfile
from pathlib import Path

from nido.config import Config, load_config


def test_default_config() -> None:
    config = Config()
    assert config.assistant.name == "Nido"
    assert config.assistant.language == "fa"
    assert config.hotkey.key == "KEY_F9"
    assert config.audio.sample_rate == 16000
    assert config.audio.channels == 1
    assert config.audio.max_seconds == 30
    assert config.ui.position == "top-right"
    assert "browser" in config.apps
    assert config.system.allow_shutdown is False


def test_load_custom_toml() -> None:
    content = """
    [assistant]
    name = "CustomNido"

    [hotkey]
    key = "KEY_F10"

    [audio]
    sample_rate = 48000
    max_seconds = 15

    [ui]
    position = "bottom-left"

    [apps]
    custom_editor = "nvim"

    [system]
    allow_shutdown = true
    """
    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
        f.write(content)
        temp_path = f.name

    try:
        config = load_config(temp_path)
        assert config.assistant.name == "CustomNido"
        assert config.hotkey.key == "KEY_F10"
        assert config.audio.sample_rate == 48000
        assert config.audio.max_seconds == 15
        assert config.ui.position == "bottom-left"
        assert config.apps.get("custom_editor") == "nvim"
        assert config.system.allow_shutdown is True
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_load_nonexistent_file_returns_defaults() -> None:
    config = load_config("/non/existent/path/config.toml")
    assert config.assistant.name == "Nido"
    assert config.hotkey.key == "KEY_F9"
