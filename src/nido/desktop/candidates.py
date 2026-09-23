"""Action candidate construction and deterministic argument binding for English commands."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from nido.accessibility.models import DesktopSnapshot, UIElement
from nido.logging import get_logger
from nido.tools.registry import ToolRegistry

logger = get_logger("nido.desktop.candidates")

# English number words to integers
ENGLISH_NUMBERS: Dict[str, int] = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "fifteen": 15,
    "twenty": 20,
    "twenty five": 25,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "one hundred": 100,
    "hundred": 100,
}

# Common English app name synonyms
APP_SYNONYMS: Dict[str, str] = {
    "notes": "kate",
    "note": "kate",
    "text editor": "kate",
    "editor": "kate",
    "files": "dolphin",
    "file manager": "dolphin",
    "downloads": "dolphin",
    "calculator": "kcalc",
    "calc": "kcalc",
    "terminal": "konsole",
    "browser": "firefox",
    "web browser": "firefox",
    "chrome": "google-chrome",
}


@dataclass
class ActionCandidate:
    """A safe, fully-bound action candidate presented to Laya for choice selection."""

    id: str  # "A1", "A2", etc.
    label: str  # Human/model readable description, e.g. 'activate [e3] button "Save"'
    action_type: str  # Internal operation identifier
    arguments: Dict[str, Any] = field(default_factory=dict)
    source: str = "desktop"  # "desktop" | "static" | "control"
    element_id: Optional[str] = None


def extract_calculation_expression(user_text: str) -> Optional[str]:
    """Extract math expression from user text (e.g. 'calculate 2*95' -> '2 * 95')."""
    if not user_text:
        return None

    # 1. Calculation verbs: calculate / compute / solve
    match = re.search(
        r'(?:calculate|compute|solve)\s+([0-9\s\+\-\*\/\.xXtimesplusminus\(\)\=]+)',
        user_text,
        re.IGNORECASE,
    )
    if match:
        expr = match.group(1).strip()
        expr = expr.replace("times", "*").replace("x", "*").replace("X", "*")
        expr = expr.replace("plus", "+")
        expr = expr.replace("minus", "-")
        expr = re.sub(r'[\=\s]*$', '', expr)
        clean_expr = re.sub(r'\s*([\+\-\*\/])\s*', r' \1 ', expr).strip()
        if any(op in clean_expr for op in "+-*/") or clean_expr.isdigit():
            return clean_expr

    # 2. Standalone math expression like "2*95" or "123 + 456"
    expr_match = re.search(r'(\d+[\s\+\-\*\/\.xX]+\d+[\d\s\+\-\*\/\.xX]*)', user_text)
    if expr_match:
        expr = expr_match.group(1).strip()
        expr = expr.replace("times", "*").replace("x", "*").replace("X", "*")
        clean_expr = re.sub(r'\s*([\+\-\*\/])\s*', r' \1 ', expr).strip()
        if any(op in clean_expr for op in "+-*/"):
            return clean_expr

    return None


def extract_literal_payload(user_text: str) -> Optional[str]:
    """Extract literal user-intended text without translating or altering it.

    Preserves exact punctuation, case, and spacing.
    """
    if not user_text:
        return None

    clean = user_text.strip()

    # 1. Quoted text: "...", '...'
    quoted_match = re.search(r'["\']([^"\']+)["\']', clean)
    if quoted_match:
        return quoted_match.group(1).strip()

    # 2. English typing verbs: e.g. "write Hello world", "type test", "and write Hello, world!"
    en_match = re.search(
        r'(?:(?:and\s+)?(?:write|type|enter))\s+(?:the\s+(?:text|sentence|word)\s+)?(.+)',
        clean,
        re.IGNORECASE,
    )
    if en_match:
        return en_match.group(1).strip()

    # 3. Calculation expressions: calculate 2*95
    calc_expr = extract_calculation_expression(clean)
    if calc_expr:
        return calc_expr

    return None


def extract_volume_percent(user_text: str) -> Optional[int]:
    """Deterministically parse requested volume percentage from English text."""
    lower = user_text.lower()

    # Check for digit percentage, e.g. "30 percent", "to 30%", "volume 30"
    match = re.search(r'(\d{1,3})\s*(?:percent|%|\b)', lower)
    if match and match.group(1):
        try:
            val = int(match.group(1))
            if 0 <= val <= 100:
                return val
        except ValueError:
            pass

    # Check English number words
    for word, val in ENGLISH_NUMBERS.items():
        if word in lower:
            return val

    return None


def extract_url_or_search(user_text: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract URL or web search query from English user text."""
    # Direct URL
    url_match = re.search(r'(https?://\S+|www\.\S+)', user_text)
    if url_match:
        url = url_match.group(1)
        if not url.startswith("http"):
            url = f"https://{url}"
        return url, None

    lower = user_text.lower()
    # Common site shortcuts
    if "youtube" in lower:
        return "https://youtube.com", None
    if "github" in lower:
        return "https://github.com", None

    # Search queries: "search for ...", "google ..."
    search_match = re.search(r'(?:search|google)\s+(?:for\s+|about\s+)?(.+)', user_text, re.IGNORECASE)
    if search_match:
        return None, search_match.group(1).strip()

    return None, None


class CandidateBuilder:
    """Builds a compact, typed action space from current snapshot, goal, and tools."""

    def __init__(
        self,
        app_map: Optional[Dict[str, str]] = None,
        max_candidates: int = 24,
    ) -> None:
        self.app_map = app_map or {}
        self.max_candidates = max_candidates

    def build_candidates(
        self,
        goal: str,
        snapshot: DesktopSnapshot,
        registry: Optional[ToolRegistry] = None,
        action_history: Optional[List[Dict[str, Any]]] = None,
    ) -> List[ActionCandidate]:
        """Construct prioritized list of ActionCandidate objects."""
        candidates: List[ActionCandidate] = []
        counter = 1
        history = action_history or []
        literal_payload = extract_literal_payload(goal)

        def next_id() -> str:
            nonlocal counter
            cid = f"A{counter}"
            counter += 1
            return cid

        # 1. Desktop Actions from Accessibility Snapshot
        actionable_elements: List[UIElement] = []
        editable_elements: List[UIElement] = []
        selectable_elements: List[UIElement] = []
        scrollable_elements: List[UIElement] = []

        for el in snapshot.elements:
            if not el.enabled or not el.visible:
                continue

            role = (el.role or "").lower()
            states = set(el.states or [])

            # Check editable
            if "editable" in states or role in ("text field", "password field", "document"):
                editable_elements.append(el)

            # Check clickable / actionable
            if el.actions or role in ("button", "menu item", "checkbox", "radio button", "tab", "hyperlink", "menu", "list item"):
                actionable_elements.append(el)

            # Check selectable
            if "selectable" in states or role in ("list item", "cell", "tab"):
                selectable_elements.append(el)

            # Check scrollable
            if role in ("scroll container", "scrollbar", "table", "list", "document"):
                scrollable_elements.append(el)

        # 1.1 Editable elements: set text / insert text
        has_set_text = any(h.get("action") in ("set_ui_text", "insert_ui_text", "type_text") for h in history)
        display_elements = [
            el for el in snapshot.elements
            if (el.name and "display" in el.name.lower()) or el.role in ("text field", "text", "entry")
        ]
        text_targets = editable_elements or display_elements

        if text_targets and literal_payload and not has_set_text:
            for el in text_targets[:2]:
                el_name = el.name or el.role
                candidates.append(ActionCandidate(
                    id=next_id(),
                    label=f'Set requested text in [{el.id}] {el.role} "{el_name}"',
                    action_type="set_ui_text",
                    arguments={"element_id": el.id, "text": literal_payload},
                    source="desktop",
                    element_id=el.id,
                ))
        elif snapshot.active_application and literal_payload and not has_set_text:
            # Fallback to direct keyboard typing if active application has no accessible text widget
            candidates.append(ActionCandidate(
                id=next_id(),
                label=f'Type "{literal_payload}" into active window',
                action_type="type_text",
                arguments={"text": f"{literal_payload}\n"},
                source="desktop",
            ))

        # 1.2 Focus editable or notable elements
        for el in editable_elements[:2]:
            if not el.focused:
                el_name = el.name or el.role
                candidates.append(ActionCandidate(
                    id=next_id(),
                    label=f'Focus [{el.id}] {el.role} "{el_name}"',
                    action_type="focus_ui_element",
                    arguments={"element_id": el.id},
                    source="desktop",
                    element_id=el.id,
                ))

        # 1.3 Actionable elements (buttons, menus, links, list items)
        for el in actionable_elements[:12]:
            el_name = el.name or el.role
            candidates.append(ActionCandidate(
                id=next_id(),
                label=f'Activate [{el.id}] {el.role} "{el_name}"',
                action_type="activate_ui_element",
                arguments={"element_id": el.id},
                source="desktop",
                element_id=el.id,
            ))

        # 1.4 Selectable elements
        for el in selectable_elements[:4]:
            if "selected" not in el.states:
                el_name = el.name or el.role
                candidates.append(ActionCandidate(
                    id=next_id(),
                    label=f'Select [{el.id}] {el.role} "{el_name}"',
                    action_type="select_ui_element",
                    arguments={"element_id": el.id},
                    source="desktop",
                    element_id=el.id,
                ))

        # 1.5 Scrollable elements
        for el in scrollable_elements[:2]:
            el_name = el.name or el.role
            candidates.append(ActionCandidate(
                id=next_id(),
                label=f'Scroll [{el.id}] {el.role} "{el_name}" down',
                action_type="scroll_ui",
                arguments={"element_id": el.id, "direction": "down"},
                source="desktop",
                element_id=el.id,
            ))

        # 2. Static Tools from Registry / Whitelist
        # 2.1 App Launching Candidates
        goal_lower = goal.lower()
        matched_apps: Set[str] = set()

        # Check synonyms
        for alias, app_key in APP_SYNONYMS.items():
            if alias in goal_lower:
                matched_apps.add(app_key)

        # Check configured app map
        for k, exec_name in self.app_map.items():
            if k in goal_lower or exec_name in goal_lower:
                matched_apps.add(exec_name)

        already_opened_apps = {
            str(h.get("arguments", {}).get("app_name", "")).lower()
            for h in history
            if h.get("action") == "open_app" and h.get("success")
        }
        active_app_lower = (snapshot.active_application or "").lower()

        for app_name in matched_apps:
            app_lower = app_name.lower()
            if app_lower in already_opened_apps:
                continue
            if active_app_lower and (app_lower in active_app_lower or active_app_lower in app_lower):
                continue
            candidates.append(ActionCandidate(
                id=next_id(),
                label=f'Open application: {app_name}',
                action_type="open_app",
                arguments={"app_name": app_name},
                source="static",
            ))

        # 2.2 Volume tools
        if any(w in goal_lower for w in ["volume", "sound"]):
            vol_val = extract_volume_percent(goal)
            if vol_val is not None:
                candidates.append(ActionCandidate(
                    id=next_id(),
                    label=f'Set system volume to {vol_val}%',
                    action_type="set_volume",
                    arguments={"percent": vol_val},
                    source="static",
                ))
            elif any(w in goal_lower for w in ["up", "increase", "raise"]):
                candidates.append(ActionCandidate(
                    id=next_id(),
                    label='Increase system volume',
                    action_type="increase_volume",
                    arguments={"step": 5},
                    source="static",
                ))
            elif any(w in goal_lower for w in ["down", "decrease", "lower"]):
                candidates.append(ActionCandidate(
                    id=next_id(),
                    label='Decrease system volume',
                    action_type="decrease_volume",
                    arguments={"step": 5},
                    source="static",
                ))
            elif "mute" in goal_lower:
                candidates.append(ActionCandidate(
                    id=next_id(),
                    label='Mute audio',
                    action_type="mute_volume",
                    arguments={},
                    source="static",
                ))
            elif "unmute" in goal_lower:
                candidates.append(ActionCandidate(
                    id=next_id(),
                    label='Unmute audio',
                    action_type="unmute_volume",
                    arguments={},
                    source="static",
                ))

        # 2.3 Media playback tools
        if any(w in goal_lower for w in ["play", "pause", "music", "track"]):
            if "next" in goal_lower:
                candidates.append(ActionCandidate(
                    id=next_id(),
                    label='Next media track',
                    action_type="next_track",
                    arguments={},
                    source="static",
                ))
            elif any(w in goal_lower for w in ["previous", "prev"]):
                candidates.append(ActionCandidate(
                    id=next_id(),
                    label='Previous media track',
                    action_type="previous_track",
                    arguments={},
                    source="static",
                ))
            else:
                candidates.append(ActionCandidate(
                    id=next_id(),
                    label='Toggle media play/pause',
                    action_type="play_pause",
                    arguments={},
                    source="static",
                ))

        # 2.4 Web URL / Search tools
        url, query = extract_url_or_search(goal)
        if url:
            candidates.append(ActionCandidate(
                id=next_id(),
                label=f'Open URL: {url}',
                action_type="open_url",
                arguments={"url": url},
                source="static",
            ))
        elif query:
            candidates.append(ActionCandidate(
                id=next_id(),
                label=f'Search the web for: "{query}"',
                action_type="search_web",
                arguments={"query": query},
                source="static",
            ))

        # 2.5 System tools (screenshot, lock)
        if "screenshot" in goal_lower:
            candidates.append(ActionCandidate(
                id=next_id(),
                label='Take desktop screenshot',
                action_type="take_screenshot",
                arguments={},
                source="static",
            ))
        if "lock" in goal_lower:
            candidates.append(ActionCandidate(
                id=next_id(),
                label='Lock screen',
                action_type="lock_screen",
                arguments={},
                source="static",
            ))

        # 3. Control Candidates
        # 3.1 Wait (useful when transitioning)
        candidates.append(ActionCandidate(
            id=next_id(),
            label='Wait for UI to settle',
            action_type="wait",
            arguments={"duration_ms": 200},
            source="control",
        ))

        # 3.2 DONE (Terminal action)
        candidates.append(ActionCandidate(
            id=next_id(),
            label="The user's goal is complete",
            action_type="done",
            arguments={"summary": f"Completed task: {goal}"},
            source="control",
        ))

        # 3.3 BLOCKED
        candidates.append(ActionCandidate(
            id=next_id(),
            label='No safe or useful action available (Blocked)',
            action_type="blocked",
            arguments={"reason": "Cannot proceed further"},
            source="control",
        ))

        return candidates[:self.max_candidates]
