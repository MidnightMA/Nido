"""Action candidate construction and deterministic argument binding."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from nido.accessibility.models import DesktopSnapshot, UIElement
from nido.logging import get_logger
from nido.tools.registry import ToolRegistry

logger = get_logger("nido.desktop.candidates")

# Persian digits to standard digits
PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")

# Persian number words to integers
PERSIAN_NUMBERS: Dict[str, int] = {
    "صفر": 0,
    "ده": 10,
    "بیست": 20,
    "سی": 30,
    "چهل": 40,
    "پنجاه": 50,
    "شصت": 60,
    "هفتاد": 70,
    "هشتاد": 80,
    "نود": 90,
    "صد": 100,
}

# Mapping Persian application names / aliases to registered app keys
PERSIAN_APP_ALIASES: Dict[str, str] = {
    "کیت": "kate",
    "نوت": "kate",
    "ویرایشگر": "kate",
    "ادیتور": "kate",
    "دلفین": "dolphin",
    "فایل": "dolphin",
    "فایل‌ها": "dolphin",
    "پوشه": "dolphin",
    "دانلود": "dolphin",
    "دانلودها": "dolphin",
    "ماشین حساب": "kcalc",
    "حساب": "kcalc",
    "فایرفاکس": "firefox",
    "مرورگر": "firefox",
    "وب": "firefox",
    "کروم": "google-chrome",
    "گوگل کروم": "google-chrome",
    "کنسول": "konsole",
    "ترمینال": "konsole",
    "خط فرمان": "konsole",
    "تنظیمات": "systemsettings",
    "کنترل پنل": "systemsettings",
    "تلگرام": "telegram-desktop",
    "دیسکورد": "discord",
    "موزیک": "elisa",
    "آهنگ": "elisa",
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
    normalized = user_text.translate(PERSIAN_DIGITS)

    # 1. Calculation verbs: calculate / compute / solve / حساب کن / محاسبه کن
    match = re.search(
        r'(?:calculate|compute|solve|حساب\s*کن|محاسبه\s*کن)\s+([0-9\s\+\-\*\/\.xXtimesضربدرجمعمنهاتقسیم\(\)\=]+)',
        normalized,
        re.IGNORECASE,
    )
    if match:
        expr = match.group(1).strip()
        expr = expr.replace("ضربدر", "*").replace("times", "*").replace("x", "*").replace("X", "*")
        expr = expr.replace("تقسیم بر", "/").replace("تقسیم", "/")
        expr = expr.replace("جمع", "+").replace("plus", "+")
        expr = expr.replace("منهای", "-").replace("منها", "-").replace("minus", "-")
        expr = re.sub(r'[\=\s]*$', '', expr)
        expr = re.sub(r'\s+را$', '', expr)
        clean_expr = re.sub(r'\s*([\+\-\*\/])\s*', r' \1 ', expr).strip()
        if any(op in clean_expr for op in "+-*/") or clean_expr.isdigit():
            return clean_expr

    # 2. Standalone math expression like "2*95" or "123 + 456"
    expr_match = re.search(r'(\d+[\s\+\-\*\/\.xX]+\d+[\d\s\+\-\*\/\.xX]*)', normalized)
    if expr_match:
        expr = expr_match.group(1).strip()
        expr = expr.replace("times", "*").replace("x", "*").replace("X", "*")
        clean_expr = re.sub(r'\s*([\+\-\*\/])\s*', r' \1 ', expr).strip()
        if any(op in clean_expr for op in "+-*/"):
            return clean_expr

    return None


def extract_literal_payload(user_text: str) -> Optional[str]:
    """Extract literal user-intended text without translating or altering it."""
    if not user_text:
        return None

    clean = user_text.strip()

    # 1. Quoted text: "...", '...', «...»
    quoted_match = re.search(r'["\'«]([^"\'»]+)["\'»]', clean)
    if quoted_match:
        return quoted_match.group(1).strip()

    # 2. Persian typing verbs: بنویس ..., تایپ کن ..., بگو ...
    type_match = re.search(r'(?:بنویس|تایپ\s*کن|وارد\s*کن|بگو)\s+(.+)', clean)
    if type_match:
        payload = type_match.group(1).strip()
        # Remove trailing polite phrases if present
        payload = re.sub(r'\s+(?:لطفا|لطفاً)$', '', payload)
        return payload

    # 3. English typing verbs: write ..., type ..., enter ...
    en_match = re.search(r'(?:write|type|enter)\s+(?:this\s+)?(?:sentence\s+)?(.+)', clean, re.IGNORECASE)
    if en_match:
        return en_match.group(1).strip()

    # 4. Calculation expressions: calculate 2*95, حساب کن 2*95
    calc_expr = extract_calculation_expression(clean)
    if calc_expr:
        return calc_expr

    return None


def extract_volume_percent(user_text: str) -> Optional[int]:
    """Deterministically parse requested volume percentage from text."""
    normalized = user_text.translate(PERSIAN_DIGITS)

    # Check for digit percentage, e.g. "30 درصد", "روی 30", "30%"
    match = re.search(r'(\d{1,3})\s*(?:درصد|percent|%)', normalized)
    if match:
        return max(0, min(100, int(match.group(1))))

    # Check for "روی 30"
    match_roy = re.search(r'روی\s+(\d{1,3})', normalized)
    if match_roy:
        return max(0, min(100, int(match_roy.group(1))))

    # Check Persian number words
    for word, val in PERSIAN_NUMBERS.items():
        if word in user_text:
            return val

    return None


def extract_url_or_search(user_text: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract URL or web search query from user text."""
    # Direct URL
    url_match = re.search(r'(https?://\S+|www\.\S+)', user_text)
    if url_match:
        url = url_match.group(1)
        if not url.startswith("http"):
            url = f"https://{url}"
        return url, None

    # Common site shortcuts
    if "یوتیوب" in user_text or "youtube" in user_text.lower():
        return "https://youtube.com", None
    if "گیت هاب" in user_text or "گیتهاب" in user_text or "github" in user_text.lower():
        return "https://github.com", None

    # Search queries
    search_match = re.search(r'(?:جستجو\s*کن|سرچ\s*کن|search|google)\s+(?:درباره\s+|برای\s+)?(.+)', user_text, re.IGNORECASE)
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
        # Priority to active window elements
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
            if el.actions or role in ("button", "menu item", "checkbox", "radio button", "tab", "hyperlink", "menu"):
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

        # 1.3 Actionable elements (buttons, menus, links)
        for el in actionable_elements[:12]:
            el_name = el.name or el.role
            action_desc = "click" if "click" in el.actions else (el.actions[0] if el.actions else "activate")
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

        # Check Persian aliases
        for alias, app_key in PERSIAN_APP_ALIASES.items():
            if alias in goal:
                matched_apps.add(app_key)

        # Check configured app map
        for k, exec_name in self.app_map.items():
            if k in goal_lower or exec_name in goal_lower:
                matched_apps.add(exec_name)

        # Check what apps have ALREADY been opened in history or are already active
        already_opened_apps = {
            str(h.get("arguments", {}).get("app_name", "")).lower()
            for h in history
            if h.get("action") == "open_app" and h.get("success")
        }
        active_app_lower = (snapshot.active_application or "").lower()

        # Build open_app candidates for matched apps that haven't been opened yet and aren't active
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
        if any(w in goal for w in ["صدا", "volume", "sound"]):
            vol_val = extract_volume_percent(goal)
            if vol_val is not None:
                candidates.append(ActionCandidate(
                    id=next_id(),
                    label=f'Set system volume to {vol_val}%',
                    action_type="set_volume",
                    arguments={"percent": vol_val},
                    source="static",
                ))
            elif any(w in goal for w in ["زیاد", "بالا", "بیشتر", "up", "increase"]):
                candidates.append(ActionCandidate(
                    id=next_id(),
                    label='Increase system volume',
                    action_type="increase_volume",
                    arguments={"step": 5},
                    source="static",
                ))
            elif any(w in goal for w in ["کم", "پایین", "کمتر", "down", "decrease"]):
                candidates.append(ActionCandidate(
                    id=next_id(),
                    label='Decrease system volume',
                    action_type="decrease_volume",
                    arguments={"step": 5},
                    source="static",
                ))
            elif any(w in goal for w in ["قطع", "بی صدا", "mute"]):
                candidates.append(ActionCandidate(
                    id=next_id(),
                    label='Mute audio',
                    action_type="mute_volume",
                    arguments={},
                    source="static",
                ))
            elif any(w in goal for w in ["وصل", "با صدا", "unmute"]):
                candidates.append(ActionCandidate(
                    id=next_id(),
                    label='Unmute audio',
                    action_type="unmute_volume",
                    arguments={},
                    source="static",
                ))

        # 2.3 Media playback tools
        if any(w in goal for w in ["پخش", "آهنگ", "موزیک", "ترک", "play", "pause", "music"]):
            if any(w in goal for w in ["بعدی", "next"]):
                candidates.append(ActionCandidate(
                    id=next_id(),
                    label='Next media track',
                    action_type="next_track",
                    arguments={},
                    source="static",
                ))
            elif any(w in goal for w in ["قبلی", "previous", "prev"]):
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
        if any(w in goal for w in ["اسکرین شات", "عکس از صفحه", "تصویر از صفحه", "screenshot"]):
            candidates.append(ActionCandidate(
                id=next_id(),
                label='Take desktop screenshot',
                action_type="take_screenshot",
                arguments={},
                source="static",
            ))
        if any(w in goal for w in ["قفل", "lock"]):
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
            arguments={"reason": "Cannot advance user goal"},
            source="control",
        ))

        # Enforce max_candidates limit while keeping DONE and BLOCKED
        if len(candidates) > self.max_candidates:
            control_cands = [c for c in candidates if c.source == "control"]
            action_cands = [c for c in candidates if c.source != "control"]
            allowed_action_count = self.max_candidates - len(control_cands)
            candidates = action_cands[:allowed_action_count] + control_cands

        # Re-index candidate IDs sequentially: A1, A2, ...
        for idx, cand in enumerate(candidates, 1):
            cand.id = f"A{idx}"

        return candidates
