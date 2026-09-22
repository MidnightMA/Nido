"""Data models for desktop accessibility representation."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class UIElement:
    """Normalized, compact representation of an accessible UI element."""

    id: str
    role: str
    name: str
    description: Optional[str] = None
    value: Optional[str] = None
    states: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    focused: bool = False
    enabled: bool = True
    visible: bool = True
    showing: bool = True
    parent_id: Optional[str] = None
    depth: int = 0
    bounds: Optional[Tuple[int, int, int, int]] = None  # (x, y, width, height)


@dataclass
class DesktopSnapshot:
    """Normalized representation of current desktop accessibility state."""

    session_type: str = "unknown"
    desktop_name: str = "unknown"
    active_application: Optional[str] = None
    active_window: Optional[str] = None
    focused_element_id: Optional[str] = None
    elements: List[UIElement] = field(default_factory=list)
    snapshot_id: str = field(default_factory=lambda: f"snap_{time.time():.6f}")
    timestamp: float = field(default_factory=time.time)

    def get_element(self, element_id: str) -> Optional[UIElement]:
        """Find an element by its ephemeral runtime ID."""
        for el in self.elements:
            if el.id == element_id:
                return el
        return None


@dataclass
class ActionResult:
    """Structured result of executing a UI or input action."""

    success: bool
    action: str
    element_id: Optional[str] = None
    message: str = ""
    error: Optional[str] = None
