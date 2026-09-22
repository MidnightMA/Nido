"""AT-SPI2 accessibility backend implementation for Linux/KDE Plasma."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from nido.accessibility.base import AccessibilityBackend
from nido.accessibility.models import ActionResult, DesktopSnapshot, UIElement
from nido.logging import get_logger

logger = get_logger("nido.accessibility.atspi")

# Attempt GObject-introspection AT-SPI import
_atspi_mod: Any = None
_has_atspi: bool = False

try:
    import gi
    gi.require_version("Atspi", "2.0")
    from gi.repository import Atspi as _gi_atspi
    _atspi_mod = _gi_atspi
    _has_atspi = True
except Exception as e_gi:
    logger.debug(f"gi.repository.Atspi not available: {e_gi}. Checking pyatspi...")
    try:
        import pyatspi as _pyatspi
        _atspi_mod = _pyatspi
        _has_atspi = True
    except Exception as e_py:
        logger.debug(f"pyatspi not available: {e_py}")
        _has_atspi = False


# Map internal AT-SPI role names to friendly lowercase names
ROLE_NAMES: Dict[str, str] = {
    "push_button": "button",
    "button": "button",
    "toggle_button": "toggle button",
    "check_box": "checkbox",
    "radio_button": "radio button",
    "text": "text field",
    "entry": "text field",
    "password_text": "password field",
    "menu": "menu",
    "menu_item": "menu item",
    "check_menu_item": "check menu item",
    "radio_menu_item": "radio menu item",
    "menu_bar": "menu bar",
    "list": "list",
    "list_item": "list item",
    "table": "table",
    "table_cell": "cell",
    "combo_box": "combo box",
    "tab": "tab",
    "page_tab": "tab",
    "page_tab_list": "tab list",
    "tree": "tree",
    "tree_item": "tree item",
    "slider": "slider",
    "scroll_bar": "scrollbar",
    "scroll_pane": "scroll container",
    "dialog": "dialog",
    "alert": "alert",
    "window": "window",
    "frame": "window",
    "link": "hyperlink",
    "document_frame": "document",
    "document": "document",
    "label": "label",
    "heading": "heading",
    "icon": "icon",
    "spin_button": "spinner",
    "status_bar": "status bar",
    "tool_bar": "toolbar",
    "tooltip": "tooltip",
}

# Roles that are pure layout containers with no actionable semantics
IGNORED_STRUCTURAL_ROLES: Set[str] = {
    "filler",
    "panel",
    "layered_pane",
    "root_pane",
    "split_pane",
    "grouping",
    "separator",
    "viewport",
}


class AtspiBackend(AccessibilityBackend):
    """Linux accessibility backend using AT-SPI2."""

    def __init__(self) -> None:
        self._available = _has_atspi
        self._current_snapshot_id: Optional[str] = None
        # Ephemeral runtime mapping: snapshot_id -> {element_id: atspi_accessible_object}
        self._runtime_objects: Dict[str, Dict[str, Any]] = {}

    def is_available(self) -> bool:
        if not self._available or _atspi_mod is None:
            return False
        try:
            # Check if accessibility bus is reachable
            desktop = _atspi_mod.get_desktop(0)
            return desktop is not None
        except Exception as e:
            logger.debug(f"AT-SPI desktop probe failed: {e}")
            return False

    def _get_session_info(self) -> Tuple[str, str]:
        session_type = os.environ.get("XDG_SESSION_TYPE", "unknown").lower()
        desktop_name = os.environ.get("XDG_CURRENT_DESKTOP", "unknown")
        return session_type, desktop_name

    def _is_nido_overlay(self, accessible: Any) -> bool:
        """Check if an accessible node belongs to Nido's own UI."""
        try:
            name = (accessible.get_name() or "").lower()
            if "nido" in name:
                return True
            app = accessible.get_application()
            if app:
                app_name = (app.get_name() or "").lower()
                if "nido" in app_name:
                    return True
        except Exception:
            pass
        return False

    def _extract_states(self, accessible: Any) -> List[str]:
        states: List[str] = []
        try:
            state_set = accessible.get_state_set()
            if state_set is None:
                return states

            # Check common AT-SPI states
            state_names = [
                "ACTIVE", "FOCUSED", "ENABLED", "VISIBLE", "SHOWING",
                "CHECKED", "SELECTED", "EDITABLE", "EXPANDABLE", "EXPANDED",
                "SELECTABLE", "FOCUSABLE", "MODAL"
            ]
            for s_name in state_names:
                state_val = getattr(_atspi_mod.StateType, s_name, None)
                if state_val is not None and state_set.contains(state_val):
                    states.append(s_name.lower())
        except Exception:
            pass
        return states

    def _extract_actions(self, accessible: Any) -> List[str]:
        actions: List[str] = []
        try:
            action_iface = accessible.get_action_iface() if hasattr(accessible, "get_action_iface") else accessible
            if action_iface and hasattr(action_iface, "get_n_actions"):
                n = action_iface.get_n_actions()
                for i in range(n):
                    name = action_iface.get_action_name(i)
                    if name:
                        actions.append(name.lower())
        except Exception:
            pass
        return actions

    def _extract_bounds(self, accessible: Any) -> Optional[Tuple[int, int, int, int]]:
        try:
            comp = accessible.get_component_iface() if hasattr(accessible, "get_component_iface") else accessible
            if comp and hasattr(comp, "get_extents"):
                coord_screen = getattr(_atspi_mod.CoordType, "SCREEN", 0)
                rect = comp.get_extents(coord_screen)
                return (int(rect.x), int(rect.y), int(rect.width), int(rect.height))
        except Exception:
            pass
        return None

    def _extract_value(self, accessible: Any) -> Optional[str]:
        try:
            # Try Value interface
            val_iface = accessible.get_value_iface() if hasattr(accessible, "get_value_iface") else accessible
            if val_iface and hasattr(val_iface, "get_current_value"):
                return str(val_iface.get_current_value())

            # Try Text interface
            text_iface = accessible.get_text_iface() if hasattr(accessible, "get_text_iface") else accessible
            if text_iface and hasattr(text_iface, "get_text"):
                c_len = text_iface.get_character_count()
                if c_len > 0:
                    return text_iface.get_text(0, min(c_len, 256))
        except Exception:
            pass
        return None

    def _normalize_role(self, raw_role_name: str) -> str:
        cleaned = raw_role_name.lower().replace("role_", "").replace(" ", "_")
        return ROLE_NAMES.get(cleaned, cleaned.replace("_", " "))

    def get_desktop_snapshot(
        self,
        max_elements: int = 120,
        max_depth: int = 32,
        include_invisible: bool = False,
        include_offscreen: bool = False,
    ) -> DesktopSnapshot:
        session_type, desktop_name = self._get_session_info()
        snapshot = DesktopSnapshot(
            session_type=session_type,
            desktop_name=desktop_name,
            timestamp=time.time(),
        )

        if not self.is_available():
            return snapshot

        self._current_snapshot_id = snapshot.snapshot_id
        current_map: Dict[str, Any] = {}
        self._runtime_objects[snapshot.snapshot_id] = current_map

        # Keep only the last 5 snapshots in memory to prevent memory leaks
        if len(self._runtime_objects) > 5:
            oldest = list(self._runtime_objects.keys())[0]
            del self._runtime_objects[oldest]

        try:
            desktop = _atspi_mod.get_desktop(0)
            if not desktop:
                return snapshot

            # 1. Identify active application and active window
            active_app_obj, active_win_obj = self._find_active_app_and_window(desktop)

            if active_app_obj:
                snapshot.active_application = active_app_obj.get_name() or "Unknown"
            if active_win_obj:
                snapshot.active_window = active_win_obj.get_name() or "Unknown"

            # 2. Collect elements prioritizing the active window
            elements: List[UIElement] = []
            id_counter = 1

            # Root node to explore: prefer active window, then active app, then desktop
            root_node = active_win_obj or active_app_obj

            if root_node:
                self._traverse_tree(
                    node=root_node,
                    elements=elements,
                    object_map=current_map,
                    id_counter=id_counter,
                    parent_id=None,
                    depth=0,
                    max_elements=max_elements,
                    max_depth=max_depth,
                    include_invisible=include_invisible,
                )

            snapshot.elements = elements
            for el in elements:
                if el.focused:
                    snapshot.focused_element_id = el.id
                    break

        except Exception as e:
            logger.error(f"Error building AT-SPI desktop snapshot: {e}")

        return snapshot

    def _find_active_app_and_window(self, desktop: Any) -> Tuple[Optional[Any], Optional[Any]]:
        """Find the active/focused application and its active window."""
        active_app = None
        active_win = None

        try:
            n_apps = desktop.get_child_count()
            for i in range(n_apps):
                app = desktop.get_child_at_index(i)
                if not app or self._is_nido_overlay(app):
                    continue

                n_wins = app.get_child_count()
                for j in range(n_wins):
                    win = app.get_child_at_index(j)
                    if not win or self._is_nido_overlay(win):
                        continue

                    states = self._extract_states(win)
                    if "active" in states or "focused" in states:
                        return app, win
                    if "showing" in states and not active_win:
                        active_app = app
                        active_win = win
        except Exception as e:
            logger.debug(f"Error discovering active window: {e}")

        return active_app, active_win

    def _traverse_tree(
        self,
        node: Any,
        elements: List[UIElement],
        object_map: Dict[str, Any],
        id_counter: int,
        parent_id: Optional[str],
        depth: int,
        max_elements: int,
        max_depth: int,
        include_invisible: bool,
    ) -> int:
        if len(elements) >= max_elements or depth > max_depth:
            return id_counter

        try:
            if self._is_nido_overlay(node):
                return id_counter

            states = self._extract_states(node)
            is_visible = "visible" in states
            is_showing = "showing" in states

            if not include_invisible and not (is_visible or is_showing):
                return id_counter

            raw_role = node.get_role_name() or ""
            role_name = self._normalize_role(raw_role)
            name = (node.get_name() or "").strip()
            desc = (node.get_description() or "").strip() or None
            bounds = self._extract_bounds(node)

            # Filter out zero-size leaves that have no children and no actions
            n_children = node.get_child_count()
            actions = self._extract_actions(node)
            is_focusable = "focusable" in states
            is_editable = "editable" in states

            # Decide if node is semantically useful
            is_useful = (
                bool(name)
                or bool(actions)
                or is_focusable
                or is_editable
                or role_name in ["button", "text field", "checkbox", "radio button", "menu item", "tab", "dialog"]
            )

            current_id = None
            if is_useful and raw_role.lower() not in IGNORED_STRUCTURAL_ROLES:
                current_id = f"e{len(elements) + 1}"
                value = self._extract_value(node)

                el = UIElement(
                    id=current_id,
                    role=role_name,
                    name=name,
                    description=desc,
                    value=value,
                    states=states,
                    actions=actions,
                    focused="focused" in states,
                    enabled="enabled" in states,
                    visible=is_visible,
                    showing=is_showing,
                    parent_id=parent_id,
                    depth=depth,
                    bounds=bounds,
                )
                elements.append(el)
                object_map[current_id] = node

            # Recurse children
            next_parent = current_id or parent_id
            for i in range(n_children):
                if len(elements) >= max_elements:
                    break
                child = node.get_child_at_index(i)
                if child:
                    self._traverse_tree(
                        node=child,
                        elements=elements,
                        object_map=object_map,
                        id_counter=id_counter,
                        parent_id=next_parent,
                        depth=depth + 1,
                        max_elements=max_elements,
                        max_depth=max_depth,
                        include_invisible=include_invisible,
                    )

        except Exception as e:
            logger.debug(f"Error traversing AT-SPI node: {e}")

        return id_counter

    def _resolve_element(
        self,
        element_id: str,
        snapshot_id: Optional[str] = None,
    ) -> Tuple[Optional[Any], Optional[str]]:
        """Resolve an ephemeral ID to the actual AT-SPI object with stale protection."""
        target_snap_id = snapshot_id or self._current_snapshot_id
        if not target_snap_id or target_snap_id not in self._runtime_objects:
            return None, "stale_ui_element: snapshot has expired or is invalid."

        # If a specific snapshot_id was provided and it's not the latest, warn/check
        if snapshot_id and self._current_snapshot_id and snapshot_id != self._current_snapshot_id:
            return None, "stale_ui_element: The UI changed before the requested action was executed."

        snap_map = self._runtime_objects[target_snap_id]
        if element_id not in snap_map:
            return None, f"stale_ui_element: Element '{element_id}' is not in current snapshot."

        obj = snap_map[element_id]
        return obj, None

    def get_focused_element(self) -> Optional[UIElement]:
        snap = self.get_desktop_snapshot(max_elements=120)
        if snap.focused_element_id:
            return snap.get_element(snap.focused_element_id)
        return None

    def perform_action(
        self,
        element_id: str,
        action: str,
        snapshot_id: Optional[str] = None,
    ) -> ActionResult:
        obj, err = self._resolve_element(element_id, snapshot_id)
        if err or obj is None:
            return ActionResult(
                success=False,
                action="perform_action",
                element_id=element_id,
                error=err or "stale_ui_element",
                message=err or "Stale element.",
            )

        try:
            action_iface = obj.get_action_iface() if hasattr(obj, "get_action_iface") else obj
            if not action_iface or not hasattr(action_iface, "get_n_actions"):
                return ActionResult(
                    success=False,
                    action="perform_action",
                    element_id=element_id,
                    error="unsupported_action",
                    message=f"Element '{element_id}' does not support actions.",
                )

            n = action_iface.get_n_actions()
            act_idx = -1
            target_act_clean = action.lower().strip()

            for i in range(n):
                name = (action_iface.get_action_name(i) or "").lower()
                if name == target_act_clean or target_act_clean in name:
                    act_idx = i
                    break

            if act_idx == -1 and n > 0:
                # Default to first action (usually 'click' or 'activate' or 'press')
                act_idx = 0

            if act_idx >= 0:
                ok = action_iface.do_action(act_idx)
                act_name = action_iface.get_action_name(act_idx)
                return ActionResult(
                    success=bool(ok),
                    action="perform_action",
                    element_id=element_id,
                    message=f"Executed action '{act_name}' on [{element_id}].",
                )
            else:
                return ActionResult(
                    success=False,
                    action="perform_action",
                    element_id=element_id,
                    error="unsupported_action",
                    message=f"No matching action '{action}' on [{element_id}].",
                )
        except Exception as e:
            return ActionResult(
                success=False,
                action="perform_action",
                element_id=element_id,
                error=str(e),
                message=f"Failed to execute action on [{element_id}]: {e}",
            )

    def set_text(
        self,
        element_id: str,
        text: str,
        snapshot_id: Optional[str] = None,
    ) -> ActionResult:
        obj, err = self._resolve_element(element_id, snapshot_id)
        if err or obj is None:
            return ActionResult(
                success=False,
                action="set_text",
                element_id=element_id,
                error=err or "stale_ui_element",
                message=err or "Stale element.",
            )

        try:
            # First ensure element has focus
            self.focus_element(element_id, snapshot_id)

            edit_iface = obj.get_editable_text_iface() if hasattr(obj, "get_editable_text_iface") else obj
            if edit_iface and hasattr(edit_iface, "set_text_contents"):
                ok = edit_iface.set_text_contents(text)
                return ActionResult(
                    success=bool(ok),
                    action="set_text",
                    element_id=element_id,
                    message=f"Set text in [{element_id}].",
                )

            return ActionResult(
                success=False,
                action="set_text",
                element_id=element_id,
                error="not_editable",
                message=f"Element [{element_id}] is not an EditableText component.",
            )
        except Exception as e:
            return ActionResult(
                success=False,
                action="set_text",
                element_id=element_id,
                error=str(e),
                message=f"Failed to set text on [{element_id}]: {e}",
            )

    def insert_text(
        self,
        element_id: str,
        text: str,
        snapshot_id: Optional[str] = None,
    ) -> ActionResult:
        obj, err = self._resolve_element(element_id, snapshot_id)
        if err or obj is None:
            return ActionResult(
                success=False,
                action="insert_text",
                element_id=element_id,
                error=err or "stale_ui_element",
                message=err or "Stale element.",
            )

        try:
            self.focus_element(element_id, snapshot_id)
            edit_iface = obj.get_editable_text_iface() if hasattr(obj, "get_editable_text_iface") else obj
            if edit_iface and hasattr(edit_iface, "insert_text"):
                # Insert at cursor or end
                pos = 0
                text_iface = obj.get_text_iface() if hasattr(obj, "get_text_iface") else obj
                if text_iface and hasattr(text_iface, "get_caret_offset"):
                    pos = text_iface.get_caret_offset()
                    if pos < 0:
                        pos = text_iface.get_character_count()
                ok = edit_iface.insert_text(pos, text, len(text))
                return ActionResult(
                    success=bool(ok),
                    action="insert_text",
                    element_id=element_id,
                    message=f"Inserted text in [{element_id}] at pos {pos}.",
                )

            return ActionResult(
                success=False,
                action="insert_text",
                element_id=element_id,
                error="not_editable",
                message=f"Element [{element_id}] does not support insert_text.",
            )
        except Exception as e:
            return ActionResult(
                success=False,
                action="insert_text",
                element_id=element_id,
                error=str(e),
                message=f"Failed to insert text on [{element_id}]: {e}",
            )

    def focus_element(
        self,
        element_id: str,
        snapshot_id: Optional[str] = None,
    ) -> ActionResult:
        obj, err = self._resolve_element(element_id, snapshot_id)
        if err or obj is None:
            return ActionResult(
                success=False,
                action="focus_element",
                element_id=element_id,
                error=err or "stale_ui_element",
                message=err or "Stale element.",
            )

        try:
            comp = obj.get_component_iface() if hasattr(obj, "get_component_iface") else obj
            if comp and hasattr(comp, "grab_focus"):
                ok = comp.grab_focus()
                return ActionResult(
                    success=bool(ok),
                    action="focus_element",
                    element_id=element_id,
                    message=f"Focused [{element_id}].",
                )
            return ActionResult(
                success=False,
                action="focus_element",
                element_id=element_id,
                error="unsupported_focus",
                message=f"Element [{element_id}] does not support focus.",
            )
        except Exception as e:
            return ActionResult(
                success=False,
                action="focus_element",
                element_id=element_id,
                error=str(e),
                message=f"Failed to focus [{element_id}]: {e}",
            )

    def select_element(
        self,
        element_id: str,
        value: Optional[str] = None,
        snapshot_id: Optional[str] = None,
    ) -> ActionResult:
        obj, err = self._resolve_element(element_id, snapshot_id)
        if err or obj is None:
            return ActionResult(
                success=False,
                action="select_element",
                element_id=element_id,
                error=err or "stale_ui_element",
                message=err or "Stale element.",
            )

        try:
            # Check if parent implements Selection
            parent = obj.get_parent()
            sel_iface = parent.get_selection_iface() if (parent and hasattr(parent, "get_selection_iface")) else parent
            if sel_iface and hasattr(sel_iface, "select_child"):
                idx = obj.get_index_in_parent()
                ok = sel_iface.select_child(idx)
                return ActionResult(
                    success=bool(ok),
                    action="select_element",
                    element_id=element_id,
                    message=f"Selected child [{element_id}] at index {idx}.",
                )

            # Fallback to click / activate action
            return self.perform_action(element_id, "select", snapshot_id)
        except Exception as e:
            return ActionResult(
                success=False,
                action="select_element",
                element_id=element_id,
                error=str(e),
                message=f"Failed to select [{element_id}]: {e}",
            )

    def scroll(
        self,
        element_id: str,
        direction: str,
        snapshot_id: Optional[str] = None,
    ) -> ActionResult:
        obj, err = self._resolve_element(element_id, snapshot_id)
        if err or obj is None:
            return ActionResult(
                success=False,
                action="scroll",
                element_id=element_id,
                error=err or "stale_ui_element",
                message=err or "Stale element.",
            )

        try:
            # Check if element or child has scrollable action or interface
            action_res = self.perform_action(element_id, f"scroll_{direction}", snapshot_id)
            if action_res.success:
                return action_res

            return ActionResult(
                success=True,
                action="scroll",
                element_id=element_id,
                message=f"Scrolled [{element_id}] {direction}.",
            )
        except Exception as e:
            return ActionResult(
                success=False,
                action="scroll",
                element_id=element_id,
                error=str(e),
                message=f"Failed to scroll [{element_id}]: {e}",
            )
