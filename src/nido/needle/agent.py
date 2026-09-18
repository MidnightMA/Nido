"""Needle 3 command router and tool calling agent."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Enforce strict offline and privacy settings
os.environ["NEEDLE_TELEMETRY"] = "0"
os.environ["DO_NOT_TRACK"] = "1"

try:
    import needle
except ImportError:
    needle = None  # type: ignore[assignment]

from nido.logging import get_logger
from nido.tools.registry import ToolRegistry

logger = get_logger("nido.needle")


@dataclass
class PlannedToolCall:
    tool_name: str
    arguments: Dict[str, Any]


class NeedleCommandRouter:
    """Routes English text commands to structured tool calls using Needle 3.

    Maintains a reusable agent and guarantees only explicitly registered tools are called.
    """

    def __init__(self, registry: ToolRegistry, max_steps: int = 8, max_new_tokens: int = 128) -> None:
        self.registry = registry
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens
        self._needle_agent: Optional[object] = None
        self._init_needle()

    def _init_needle(self) -> None:
        if needle is None:
            logger.info("cactus-needle package not found. Using structured rule-based router.")
            return

        try:
            # Register tools with needle
            needle_tools = []
            for tool_def in self.registry.list_tools():
                # Wrap tool with needle.tool decorator if available
                if hasattr(needle, "tool"):
                    n_tool = needle.tool(tool_def.func)
                    needle_tools.append(n_tool)
                else:
                    needle_tools.append(tool_def.func)

            if hasattr(needle, "Agent"):
                self._needle_agent = needle.Agent(
                    tools=needle_tools,
                    max_steps=self.max_steps,
                    max_new_tokens=self.max_new_tokens,
                )
                logger.info("Needle 3 agent initialized with registered tools.")
        except Exception as e:
            logger.warning(f"Could not initialize Needle Agent: {e}. Falling back to rule router.")
            self._needle_agent = None

    def plan(self, command: str) -> List[PlannedToolCall]:
        """Analyze English command and produce an ordered list of tool calls."""
        clean_cmd = command.strip()
        if not clean_cmd:
            return []

        logger.info(f"Needle planning tool calls for command: '{clean_cmd}'")

        # If Needle agent is available, invoke it
        if self._needle_agent is not None:
            try:
                # Needle tool calling execution/planning
                calls = self._plan_with_needle(clean_cmd)
                if calls:
                    return calls
            except Exception as e:
                logger.error(f"Needle inference error: {e}. Attempting fallback router.")

        # Fallback structured deterministic parser
        return self._plan_fallback(clean_cmd)

    def _plan_with_needle(self, command: str) -> List[PlannedToolCall]:
        calls: List[PlannedToolCall] = []
        if hasattr(self._needle_agent, "plan"):
            raw_calls = self._needle_agent.plan(command)
            for call in raw_calls:
                t_name = getattr(call, "name", "") or call.get("name", "")
                t_args = getattr(call, "args", {}) or call.get("args", {})
                if self.registry.get_tool(t_name):
                    calls.append(PlannedToolCall(tool_name=t_name, arguments=t_args))
        return calls

    def _plan_fallback(self, command: str) -> List[PlannedToolCall]:
        """Deterministic semantic matcher for standard voice commands.

        Ensures zero-latency and reliable offline operation for common desktop actions.
        """
        lower = command.lower()
        calls: List[PlannedToolCall] = []

        # Split multi-action commands with 'and' or 'then'
        parts = re.split(r"\s+(?:and|then|\&)\s+", lower)

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # Check volume commands
            vol_match = re.search(r"(?:set|put)\s+volume\s+(?:to\s+)?(\d+)", part)
            if vol_match:
                calls.append(PlannedToolCall("set_volume", {"percent": int(vol_match.group(1))}))
                continue

            if "increase volume" in part or "volume up" in part:
                calls.append(PlannedToolCall("increase_volume", {"step": 5}))
                continue

            if "decrease volume" in part or "volume down" in part:
                calls.append(PlannedToolCall("decrease_volume", {"step": 5}))
                continue

            if "mute" in part or "silence" in part:
                calls.append(PlannedToolCall("mute_volume", {}))
                continue

            if "unmute" in part:
                calls.append(PlannedToolCall("unmute_volume", {}))
                continue

            # Check media playback
            if "play" in part or "pause" in part:
                calls.append(PlannedToolCall("play_pause", {}))
                continue

            if "next track" in part or "next song" in part:
                calls.append(PlannedToolCall("next_track", {}))
                continue

            if "previous track" in part or "prev track" in part:
                calls.append(PlannedToolCall("previous_track", {}))
                continue

            # Check screenshot
            if "screenshot" in part or "capture screen" in part:
                calls.append(PlannedToolCall("take_screenshot", {}))
                continue

            # Check lock screen
            if "lock screen" in part or "lock desktop" in part or "lock session" in part:
                calls.append(PlannedToolCall("lock_screen", {}))
                continue

            # Check web URL / youtube / search
            if "youtube" in part:
                calls.append(PlannedToolCall("open_url", {"url": "https://youtube.com"}))
                continue

            if "github" in part:
                calls.append(PlannedToolCall("open_url", {"url": "https://github.com"}))
                continue

            search_match = re.search(r"(?:search|google)\s+(?:for\s+)?(.+)", part)
            if search_match:
                calls.append(PlannedToolCall("search_web", {"query": search_match.group(1)}))
                continue

            url_match = re.search(r"(?:open|go to)\s+(https?://\S+|www\.\S+)", part)
            if url_match:
                calls.append(PlannedToolCall("open_url", {"url": url_match.group(1)}))
                continue

            # Check app launching
            open_match = re.search(r"(?:open|launch|run|start)\s+(?:the\s+)?([a-zA-Z0-9_\-]+)", part)
            if open_match:
                app_candidate = open_match.group(1)
                # Map candidate name
                calls.append(PlannedToolCall("open_app", {"app_name": app_candidate}))
                continue

            # Check app closing
            close_match = re.search(r"(?:close|quit|stop|exit)\s+(?:the\s+)?([a-zA-Z0-9_\-]+)", part)
            if close_match:
                calls.append(PlannedToolCall("close_app", {"app_name": close_match.group(1)}))
                continue

            # Check system info
            if "system info" in part or "system information" in part:
                calls.append(PlannedToolCall("show_system_info", {}))
                continue

        return calls
