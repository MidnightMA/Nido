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

    def decide_desktop_action(
        self,
        goal: str,
        desktop_state: str,
        tool_schemas: List[Dict[str, Any]],
        original_persian: str = "",
        action_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Select the next structured desktop action for the given goal and UI state."""
        history = action_history or []

        # If needle agent has interactive completion/routing capability, try it first
        if self._needle_agent is not None:
            try:
                action = self._decide_with_needle(goal, desktop_state, tool_schemas, original_persian, history)
                if action and action[0]:
                    return action
            except Exception as e:
                logger.debug(f"Needle desktop decision error: {e}. Falling back to semantic router.")

        return self._decide_desktop_fallback(goal, desktop_state, tool_schemas, original_persian, history)

    def _decide_with_needle(
        self,
        goal: str,
        desktop_state: str,
        tool_schemas: List[Dict[str, Any]],
        original_persian: str,
        history: List[Dict[str, Any]],
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Attempt desktop action decision using cactus-needle."""
        if hasattr(self._needle_agent, "complete"):
            prompt = (
                f"User Goal: {goal}\n"
                f"Original Text: {original_persian}\n\n"
                f"{desktop_state}\n\n"
                f"Choose the single next tool call to make progress toward the user goal."
            )
            res = self._needle_agent.complete(prompt, tools=tool_schemas)
            if isinstance(res, tuple) and len(res) == 2:
                return res[0], res[1]
            if hasattr(res, "name") and hasattr(res, "args"):
                return res.name, res.args
        return None

    def _extract_literal_text(self, goal: str, original_persian: str) -> Optional[str]:
        """Preserve exact user text without translating it into English."""
        # 1. Look for Persian text explicitly requested to be written
        if original_persian:
            # Check quotes in Persian
            p_quoted = re.search(r'["\'«]([^"\'»]+)["\'»]', original_persian)
            if p_quoted:
                return p_quoted.group(1).strip()

            # Check keywords like بنویس or تایپ کن
            p_match = re.search(r'(?:بنویس|تایپ\s*کن|بگو)\s+(.+)', original_persian)
            if p_match:
                return p_match.group(1).strip()

        # 2. Check English goal for quotes or 'write' / 'type'
        e_quoted = re.search(r'["\']([^"\']+)["\']', goal)
        if e_quoted:
            return e_quoted.group(1).strip()

        e_match = re.search(r'(?:write|type|enter)\s+(?:this\s+)?(?:sentence\s+)?(.+)', goal, re.IGNORECASE)
        if e_match:
            candidate = e_match.group(1).strip()
            # If user said "write hello", return "hello"
            return candidate

        return None

    def _parse_desktop_elements(self, desktop_state: str) -> List[Dict[str, Any]]:
        """Parse compact AVAILABLE UI lines from desktop state string."""
        elements: List[Dict[str, Any]] = []
        for line in desktop_state.splitlines():
            line = line.strip()
            match = re.match(r'\[(e\d+)\]\s+([\w\s]+?)(?:\s+"([^"]+)")?(?:\s+\(([^)]+)\))?$', line)
            if match:
                el_id, role, name, extra = match.groups()
                elements.append({
                    "id": el_id,
                    "role": role.strip().lower(),
                    "name": (name or "").strip(),
                    "extra": (extra or "").strip().lower(),
                })
        return elements

    def _decide_desktop_fallback(
        self,
        goal: str,
        desktop_state: str,
        tool_schemas: List[Dict[str, Any]],
        original_persian: str,
        history: List[Dict[str, Any]],
    ) -> Tuple[str, Dict[str, Any]]:
        """Deterministic semantic matcher for multi-step desktop tasks."""
        goal_lower = goal.lower()
        elements = self._parse_desktop_elements(desktop_state)
        tool_names = {t["name"] for t in tool_schemas}

        # Check if an action was already performed in history
        last_action = history[-1] if history else None
        has_set_text = any(h.get("action") == "set_ui_text" or h.get("action") == "insert_ui_text" for h in history)
        has_opened_app = any(h.get("action") == "open_app" for h in history)

        # 1. Text Entry Tasks (e.g. "open Kate and write hello", "write this Persian sentence")
        literal_text = self._extract_literal_text(goal, original_persian)
        if literal_text and not has_set_text:
            # Find editable element
            editable = next(
                (el for el in elements if "editable" in el["extra"] or el["role"] in ["text field", "text area", "document"]),
                None,
            )
            if editable and "set_ui_text" in tool_names:
                return "set_ui_text", {"element_id": editable["id"], "text": literal_text}

            # If no editable element visible, check if we need to launch the app
            open_match = re.search(r"(?:open|launch)\s+([a-zA-Z0-9_\-]+)", goal_lower)
            if open_match and not has_opened_app:
                app_name = open_match.group(1)
                if "open_app" in tool_names:
                    return "open_app", {"app_name": app_name}

        # If text was just set successfully, we are done!
        if has_set_text:
            return "done", {"summary": f"Entered text '{literal_text or ''}'"}

        # 2. Navigation / Button Clicking (e.g. "go to Downloads", "click Save", "press photo button", "new tab")
        # Extract target keywords
        click_match = re.search(r"(?:click|press|activate|select|go to|navigate to)\s+(?:the\s+)?([a-zA-Z0-9_\-\s]+)", goal_lower)
        if click_match:
            raw_target = click_match.group(1).strip()
            # Clean target word
            target_words = [w for w in re.split(r"\s+", raw_target) if w not in ("button", "item", "menu", "tab")]
            if target_words:
                target_phrase = " ".join(target_words)
                # Find matching element in elements
                matched_el = None
                for el in elements:
                    el_name_lower = el["name"].lower()
                    if target_phrase in el_name_lower or any(w in el_name_lower for w in target_words):
                        matched_el = el
                        break

                if matched_el:
                    # Check if already clicked in history
                    already_clicked = any(
                        h.get("element_id") == matched_el["id"] and h.get("success") for h in history
                    )
                    if not already_clicked:
                        if "activate_ui_element" in tool_names:
                            return "activate_ui_element", {"element_id": matched_el["id"]}
                        if "select_ui_element" in tool_names:
                            return "select_ui_element", {"element_id": matched_el["id"]}

        # 3. App Launching when app not yet active
        open_match = re.search(r"(?:open|launch)\s+([a-zA-Z0-9_\-]+)", goal_lower)
        if open_match and not has_opened_app:
            app_name = open_match.group(1)
            # Check if active window already belongs to app
            active_win_match = re.search(r"active window:\s*(.+)", desktop_state, re.IGNORECASE)
            active_win = active_win_match.group(1).lower() if active_win_match else ""
            if app_name not in active_win and "open_app" in tool_names:
                return "open_app", {"app_name": app_name}

        # 4. Calculation tasks (e.g. "calculate 123 times 456")
        calc_match = re.search(r"calculate\s+([\d\s\+\-\*\/\.xXtimes]+)", goal_lower)
        if calc_match:
            expr = calc_match.group(1).replace("times", "*").replace("x", "*").strip()
            # If calculator has an editable text or display, set it or type it
            editable = next((el for el in elements if "editable" in el["extra"] or "text" in el["role"]), None)
            if editable and not has_set_text and "set_ui_text" in tool_names:
                return "set_ui_text", {"element_id": editable["id"], "text": expr}

        # If any useful action was performed and no pending steps, return done
        if history and any(h.get("success") for h in history):
            return "done", {"summary": f"Completed desktop task for goal: '{goal}'"}

        # Default fallback: wait or done
        if "wait" in tool_names and not history:
            return "wait", {"duration_ms": 200}

        return "done", {"summary": f"Finished desktop interaction for: '{goal}'"}

