"""Formatting and diffing utilities for DesktopSnapshot."""

from __future__ import annotations

from typing import List, Optional

from nido.accessibility.models import DesktopSnapshot, UIElement


def format_snapshot_for_prompt(snapshot: DesktopSnapshot) -> str:
    """Format a DesktopSnapshot into a compact, human- and model-readable representation."""
    lines: List[str] = []

    lines.append("CURRENT DESKTOP")
    lines.append(f"session: {snapshot.session_type}")
    lines.append(f"desktop: {snapshot.desktop_name}")
    lines.append(f"active application: {snapshot.active_application or 'None'}")
    lines.append(f"active window: {snapshot.active_window or 'None'}")

    focused_el: Optional[UIElement] = None
    if snapshot.focused_element_id:
        focused_el = snapshot.get_element(snapshot.focused_element_id)

    if focused_el:
        lines.append(f"focused element: [{focused_el.id}] {focused_el.role} \"{focused_el.name}\"")
    else:
        lines.append("focused element: None")

    lines.append("")
    lines.append("AVAILABLE UI")

    if not snapshot.elements:
        lines.append("Accessibility unavailable for this window.")
        return "\n".join(lines)

    for el in snapshot.elements:
        # Build compact line: [e1] button "Save" (focused)
        parts = [f"[{el.id}]", el.role]
        if el.name:
            parts.append(f"\"{el.name}\"")
        if el.value:
            parts.append(f"(value: \"{el.value}\")")

        extra_tags = []
        if el.focused:
            extra_tags.append("focused")
        if not el.enabled:
            extra_tags.append("disabled")
        if "checked" in el.states:
            extra_tags.append("checked")
        if "selected" in el.states:
            extra_tags.append("selected")
        if "editable" in el.states:
            extra_tags.append("editable")

        if extra_tags:
            parts.append(f"({', '.join(extra_tags)})")

        lines.append(" ".join(parts))

    return "\n".join(lines)


def compute_snapshot_diff(
    prev: Optional[DesktopSnapshot],
    current: DesktopSnapshot,
) -> str:
    """Compute a concise diff between two snapshots to describe UI changes."""
    if prev is None:
        return f"Initial UI observed: {len(current.elements)} elements available."

    lines: List[str] = []

    if prev.active_window != current.active_window:
        lines.append(f"Window changed: '{prev.active_window}' -> '{current.active_window}'")

    if prev.active_application != current.active_application:
        lines.append(f"Application changed: '{prev.active_application}' -> '{current.active_application}'")

    prev_names = {el.name for el in prev.elements if el.name}
    curr_names = {el.name for el in current.elements if el.name}

    added = curr_names - prev_names
    removed = prev_names - curr_names

    if added:
        sample_added = list(added)[:5]
        lines.append(f"New elements appeared: {', '.join(f'\"{a}\"' for a in sample_added)}")
    if removed:
        sample_removed = list(removed)[:5]
        lines.append(f"Elements closed/hidden: {', '.join(f'\"{r}\"' for r in sample_removed)}")

    if prev.focused_element_id != current.focused_element_id:
        prev_f = prev.get_element(prev.focused_element_id) if prev.focused_element_id else None
        curr_f = current.get_element(current.focused_element_id) if current.focused_element_id else None
        p_name = prev_f.name if prev_f else "None"
        c_name = curr_f.name if curr_f else "None"
        if p_name != c_name:
            lines.append(f"Focus changed: '{p_name}' -> '{c_name}'")

    if not lines:
        return "UI settled; no significant structural changes."

    return "; ".join(lines)
