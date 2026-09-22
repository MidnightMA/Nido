"""Accessibility subsystem package."""

from __future__ import annotations

from nido.accessibility.atspi import AtspiBackend
from nido.accessibility.base import AccessibilityBackend
from nido.accessibility.models import ActionResult, DesktopSnapshot, UIElement
from nido.accessibility.snapshot import compute_snapshot_diff, format_snapshot_for_prompt

__all__ = [
    "AccessibilityBackend",
    "AtspiBackend",
    "UIElement",
    "DesktopSnapshot",
    "ActionResult",
    "format_snapshot_for_prompt",
    "compute_snapshot_diff",
]
