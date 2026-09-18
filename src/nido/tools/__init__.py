"""Tool registration and factory module for Nido."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nido.tools.apps import AppTools
from nido.tools.audio import (
    decrease_volume,
    increase_volume,
    mute_volume,
    set_volume,
    unmute_volume,
)
from nido.tools.browser import open_url, search_web
from nido.tools.files import find_files, open_file, open_folder
from nido.tools.media import next_track, play_pause, previous_track
from nido.tools.registry import ToolDefinition, ToolRegistry
from nido.tools.system import SystemTools

if TYPE_CHECKING:
    from nido.config import Config


def build_default_registry(config: "Config") -> ToolRegistry:
    """Construct and populate a ToolRegistry with all enabled tools."""
    registry = ToolRegistry()

    # Apps
    app_tools = AppTools(whitelist=config.apps)
    registry.register(
        name="open_app",
        description="Launch a desktop application from the whitelist (e.g. browser, terminal, code).",
    )(app_tools.open_app)
    registry.register(
        name="close_app",
        description="Terminate a running whitelisted application.",
    )(app_tools.close_app)

    # Browser
    registry.register(
        name="open_url",
        description="Open a web URL (http:// or https://) in the default browser.",
    )(open_url)
    registry.register(
        name="search_web",
        description="Search the web with Google, DuckDuckGo, or Bing.",
    )(search_web)

    # Audio
    registry.register(
        name="set_volume",
        description="Set system output volume percentage (0 to 100).",
    )(set_volume)
    registry.register(
        name="increase_volume",
        description="Increase system volume by step percentage (default 5%).",
    )(increase_volume)
    registry.register(
        name="decrease_volume",
        description="Decrease system volume by step percentage (default 5%).",
    )(decrease_volume)
    registry.register(
        name="mute_volume",
        description="Mute system audio output.",
    )(mute_volume)
    registry.register(
        name="unmute_volume",
        description="Unmute system audio output.",
    )(unmute_volume)

    # Media
    registry.register(
        name="play_pause",
        description="Toggle play/pause on the active media player.",
    )(play_pause)
    registry.register(
        name="next_track",
        description="Skip to next media track.",
    )(next_track)
    registry.register(
        name="previous_track",
        description="Skip to previous media track.",
    )(previous_track)

    # Files
    registry.register(
        name="open_file",
        description="Open a file with the default desktop application.",
    )(open_file)
    registry.register(
        name="open_folder",
        description="Open a folder in the desktop file manager.",
    )(open_folder)
    registry.register(
        name="find_files",
        description="Find files matching a query inside a directory.",
    )(find_files)

    # System
    sys_tools = SystemTools(
        allow_shutdown=config.system.allow_shutdown,
        allow_reboot=config.system.allow_reboot,
    )
    registry.register(
        name="take_screenshot",
        description="Take a desktop screenshot using Spectacle or Grim.",
    )(sys_tools.take_screenshot)
    registry.register(
        name="show_system_info",
        description="Display system operating system, kernel, and memory information.",
    )(sys_tools.show_system_info)
    registry.register(
        name="lock_screen",
        description="Lock the desktop session.",
    )(sys_tools.lock_screen)

    return registry


__all__ = [
    "ToolRegistry",
    "ToolDefinition",
    "build_default_registry",
    "AppTools",
    "SystemTools",
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
]
