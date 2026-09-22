"""Accessibility backend interface protocol."""

from __future__ import annotations

from typing import Optional, Protocol

from nido.accessibility.models import ActionResult, DesktopSnapshot, UIElement


class AccessibilityBackend(Protocol):
    """Abstract interface for desktop perception and accessibility actions."""

    def is_available(self) -> bool:
        """Return whether accessibility subsystem is available on this system."""
        ...

    def get_desktop_snapshot(
        self,
        max_elements: int = 120,
        max_depth: int = 32,
        include_invisible: bool = False,
        include_offscreen: bool = False,
    ) -> DesktopSnapshot:
        """Capture and return a normalized snapshot of the active window/desktop."""
        ...

    def get_focused_element(self) -> Optional[UIElement]:
        """Return the currently focused UIElement, if any."""
        ...

    def perform_action(
        self,
        element_id: str,
        action: str,
        snapshot_id: Optional[str] = None,
    ) -> ActionResult:
        """Perform an accessible action (e.g. click, press, activate)."""
        ...

    def set_text(
        self,
        element_id: str,
        text: str,
        snapshot_id: Optional[str] = None,
    ) -> ActionResult:
        """Set the text contents of an editable element."""
        ...

    def insert_text(
        self,
        element_id: str,
        text: str,
        snapshot_id: Optional[str] = None,
    ) -> ActionResult:
        """Insert text into an editable element at the cursor position."""
        ...

    def focus_element(
        self,
        element_id: str,
        snapshot_id: Optional[str] = None,
    ) -> ActionResult:
        """Request focus for the given element."""
        ...

    def select_element(
        self,
        element_id: str,
        value: Optional[str] = None,
        snapshot_id: Optional[str] = None,
    ) -> ActionResult:
        """Select an element or an item within a container."""
        ...

    def scroll(
        self,
        element_id: str,
        direction: str,
        snapshot_id: Optional[str] = None,
    ) -> ActionResult:
        """Scroll a scrollable container in direction (up, down, left, right)."""
        ...
