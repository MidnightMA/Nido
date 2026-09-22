"""Dynamic tool builder and validator for desktop interaction."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from nido.accessibility.models import DesktopSnapshot, UIElement


class DynamicDesktopToolBuilder:
    """Builds per-snapshot dynamic tool schemas and validates structured model calls."""

    def build_tool_schemas(self, snapshot: DesktopSnapshot) -> List[Dict[str, Any]]:
        """Generate structured tool definitions valid for the given DesktopSnapshot."""
        all_ids = [el.id for el in snapshot.elements]

        # Categorize element IDs for tighter enum validation
        actionable_ids = [
            el.id for el in snapshot.elements
            if el.actions or el.role in ["button", "menu item", "checkbox", "radio button", "tab", "hyperlink"]
        ] or all_ids

        editable_ids = [
            el.id for el in snapshot.elements
            if "editable" in el.states or el.role in ["text field", "password field", "document"]
        ] or all_ids

        selectable_ids = [
            el.id for el in snapshot.elements
            if "selectable" in el.states or el.role in ["list item", "radio button", "checkbox", "tab", "cell"]
        ] or all_ids

        scrollable_ids = [
            el.id for el in snapshot.elements
            if el.role in ["scroll container", "scrollbar", "table", "list", "document"]
        ] or all_ids

        tools: List[Dict[str, Any]] = []

        if actionable_ids:
            tools.append({
                "name": "activate_ui_element",
                "description": "Activate, click, or press one of the currently available UI elements.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "element_id": {
                            "type": "string",
                            "enum": actionable_ids,
                            "description": "The ephemeral ID of the UI element to activate.",
                        }
                    },
                    "required": ["element_id"],
                },
            })

        if all_ids:
            tools.append({
                "name": "focus_ui_element",
                "description": "Move keyboard focus to a currently available UI element.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "element_id": {
                            "type": "string",
                            "enum": all_ids,
                            "description": "The ephemeral ID of the UI element to focus.",
                        }
                    },
                    "required": ["element_id"],
                },
            })

        if editable_ids:
            tools.append({
                "name": "set_ui_text",
                "description": "Replace the entire text content in an editable UI element.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "element_id": {
                            "type": "string",
                            "enum": editable_ids,
                            "description": "The ephemeral ID of the editable element.",
                        },
                        "text": {
                            "type": "string",
                            "maxLength": 4096,
                            "description": "The exact text to set.",
                        },
                    },
                    "required": ["element_id", "text"],
                },
            })
            tools.append({
                "name": "insert_ui_text",
                "description": "Insert text into an editable UI element at the current cursor position.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "element_id": {
                            "type": "string",
                            "enum": editable_ids,
                            "description": "The ephemeral ID of the editable element.",
                        },
                        "text": {
                            "type": "string",
                            "maxLength": 4096,
                            "description": "The text to insert.",
                        },
                    },
                    "required": ["element_id", "text"],
                },
            })

        if selectable_ids:
            tools.append({
                "name": "select_ui_element",
                "description": "Select an item or toggle an element in the current UI.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "element_id": {
                            "type": "string",
                            "enum": selectable_ids,
                            "description": "The ephemeral ID of the element to select.",
                        }
                    },
                    "required": ["element_id"],
                },
            })

        if scrollable_ids:
            tools.append({
                "name": "scroll_ui",
                "description": "Scroll a scrollable container in the current UI.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "element_id": {
                            "type": "string",
                            "enum": scrollable_ids,
                            "description": "The ephemeral ID of the scrollable container.",
                        },
                        "direction": {
                            "type": "string",
                            "enum": ["up", "down", "left", "right"],
                            "description": "Scroll direction.",
                        },
                    },
                    "required": ["element_id", "direction"],
                },
            })

        # General keyboard and desktop tools
        tools.append({
            "name": "press_key",
            "description": "Press a keyboard key (e.g. Return, Escape, Tab, BackSpace, Down).",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Key name to press.",
                    }
                },
                "required": ["key"],
            },
        })

        tools.append({
            "name": "press_hotkey",
            "description": "Press a key combination (e.g. ['Control', 's'], ['Control', 't'], ['Alt', 'F4']).",
            "parameters": {
                "type": "object",
                "properties": {
                    "keys": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of modifier and character keys to press together.",
                    }
                },
                "required": ["keys"],
            },
        })

        tools.append({
            "name": "open_app",
            "description": "Launch a desktop application by name (e.g. kate, dolphin, kcalc).",
            "parameters": {
                "type": "object",
                "properties": {
                    "app_name": {
                        "type": "string",
                        "description": "Application name to launch.",
                    }
                },
                "required": ["app_name"],
            },
        })

        tools.append({
            "name": "wait",
            "description": "Wait briefly for the UI to settle or an application to load.",
            "parameters": {
                "type": "object",
                "properties": {
                    "duration_ms": {
                        "type": "integer",
                        "minimum": 50,
                        "maximum": 5000,
                        "description": "Time to wait in milliseconds.",
                    }
                },
                "required": ["duration_ms"],
            },
        })

        tools.append({
            "name": "done",
            "description": "Signal that the desktop interaction goal has been completed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Summary of the completed task.",
                    }
                },
                "required": ["summary"],
            },
        })

        return tools

    def validate_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        snapshot: DesktopSnapshot,
    ) -> Tuple[bool, Optional[str]]:
        """Validate a tool call against the current DesktopSnapshot."""
        valid_tools = {
            "activate_ui_element",
            "focus_ui_element",
            "set_ui_text",
            "insert_ui_text",
            "select_ui_element",
            "scroll_ui",
            "press_key",
            "press_hotkey",
            "open_app",
            "wait",
            "done",
        }

        if tool_name not in valid_tools:
            return False, f"Unknown desktop action '{tool_name}'."

        # Stale element protection: if element_id is specified, must exist in current snapshot
        if "element_id" in arguments:
            el_id = str(arguments["element_id"])
            if not snapshot.get_element(el_id):
                return (
                    False,
                    f"stale_ui_element: Element '{el_id}' does not exist in the current UI snapshot.",
                )

        if tool_name in ("set_ui_text", "insert_ui_text"):
            if "text" not in arguments or not isinstance(arguments["text"], str):
                return False, f"Action '{tool_name}' requires string argument 'text'."

        if tool_name == "scroll_ui":
            direction = arguments.get("direction", "").lower()
            if direction not in ("up", "down", "left", "right"):
                return False, f"Invalid scroll direction '{direction}'."

        if tool_name == "press_key":
            if not arguments.get("key"):
                return False, "Action 'press_key' requires argument 'key'."

        if tool_name == "press_hotkey":
            keys = arguments.get("keys")
            if not isinstance(keys, list) or not keys:
                return False, "Action 'press_hotkey' requires non-empty list 'keys'."

        if tool_name == "open_app":
            if not arguments.get("app_name"):
                return False, "Action 'open_app' requires argument 'app_name'."

        return True, None
