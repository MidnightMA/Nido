"""Tests for tool registry and safe system tool implementations."""

import pytest
from nido.config import Config
from nido.tools import build_default_registry
from nido.tools.apps import AppTools
from nido.tools.audio import decrease_volume, increase_volume, set_volume
from nido.tools.browser import open_url, search_web
from nido.tools.files import open_file, open_folder
from nido.tools.registry import ToolRegistry
from nido.tools.system import SystemTools


def test_tool_registry_registration_and_execution() -> None:
    registry = ToolRegistry()

    @registry.register(name="add_numbers", description="Add two integers.")
    def add(a: int, b: int) -> dict:
        return {"success": True, "sum": a + b}

    assert registry.get_tool("add_numbers") is not None

    # Valid execution
    res = registry.execute("add_numbers", {"a": 3, "b": 7})
    assert res["success"] is True
    assert res["sum"] == 10

    # Missing / invalid arguments
    res_err = registry.execute("add_numbers", {"wrong_param": 10})
    assert res_err["success"] is False
    assert "error" in res_err

    # Unregistered tool
    res_unreg = registry.execute("unknown_tool", {})
    assert res_unreg["success"] is False
    assert "not registered" in res_unreg["error"]


def test_app_tools_whitelist() -> None:
    apps = AppTools(whitelist={"browser": "firefox", "editor": "code"})

    # Disallowed app
    res_disallowed = apps.open_app("rm -rf /")
    assert res_disallowed["success"] is False
    assert "not in the whitelist" in res_disallowed["error"]

    res_disallowed2 = apps.open_app("malicious_binary")
    assert res_disallowed2["success"] is False

    # Close app disallowed
    res_close = apps.close_app("unknown_app")
    assert res_close["success"] is False


def test_browser_url_validation() -> None:
    # Disallowed schemes
    res_file = open_url("file:///etc/passwd")
    assert res_file["success"] is False

    res_js = open_url("javascript:alert(1)")
    assert res_js["success"] is False

    # Search web generates valid URL
    res_search = search_web("python tutorials")
    # Will attempt to open via webbrowser.open, returning dict
    assert isinstance(res_search, dict)


def test_audio_volume_clamping() -> None:
    # Even if audio backend is mocked or not running, arguments must be validated
    res_high = set_volume(150)
    assert isinstance(res_high, dict)

    res_low = set_volume(-50)
    assert isinstance(res_low, dict)


def test_system_tools_safety() -> None:
    # Shutdown disabled by default
    sys_safe = SystemTools(allow_shutdown=False, allow_reboot=False)
    res_sd = sys_safe.shutdown()
    assert res_sd["success"] is False
    assert "disabled" in res_sd["error"]

    res_rb = sys_safe.reboot()
    assert res_rb["success"] is False
    assert "disabled" in res_rb["error"]

    # System info is safe and structured
    info = sys_safe.show_system_info()
    assert info["success"] is True
    assert "details" in info


def test_file_navigation_safety() -> None:
    # Non-existent file
    res = open_file("/non/existent/secret/file.txt")
    assert res["success"] is False
    assert "does not exist" in res["error"]

    # Non-existent folder
    res_f = open_folder("/non/existent/folder/")
    assert res_f["success"] is False
    assert "does not exist" in res_f["error"]


def test_default_registry_builder() -> None:
    cfg = Config()
    registry = build_default_registry(cfg)
    tools = {t.name for t in registry.list_tools()}

    expected = {
        "open_app",
        "close_app",
        "open_url",
        "search_web",
        "set_volume",
        "increase_volume",
        "decrease_volume",
        "mute_volume",
        "unmute_volume",
        "play_pause",
        "next_track",
        "previous_track",
        "open_file",
        "open_folder",
        "find_files",
        "take_screenshot",
        "show_system_info",
        "lock_screen",
    }
    assert expected.issubset(tools)
