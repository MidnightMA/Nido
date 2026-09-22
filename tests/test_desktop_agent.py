"""Unit tests for DesktopAgent loop, text preservation, and recovery."""

from typing import List, Optional
import pytest

from nido.accessibility.base import AccessibilityBackend
from nido.accessibility.models import ActionResult, DesktopSnapshot, UIElement
from nido.config import DesktopConfig
from nido.desktop.agent import DesktopAgent, DesktopAgentResult
from nido.desktop.input import MockInputBackend
from nido.needle.agent import NeedleCommandRouter
from nido.tools.registry import ToolRegistry


class MockAccessibilityBackend:
    """Mock accessibility backend providing scripted snapshots and action tracking."""

    def __init__(self, snapshots: Optional[List[DesktopSnapshot]] = None) -> None:
        self.snapshots = snapshots or []
        self.call_count = 0
        self.actions_performed: List[tuple] = []
        self.texts_set: List[tuple] = []
        self._available = True

    def is_available(self) -> bool:
        return self._available

    def get_desktop_snapshot(self, **kwargs) -> DesktopSnapshot:
        if self.snapshots:
            idx = min(self.call_count, len(self.snapshots) - 1)
            self.call_count += 1
            return self.snapshots[idx]
        return DesktopSnapshot()

    def get_focused_element(self) -> Optional[UIElement]:
        return None

    def perform_action(self, element_id: str, action: str, snapshot_id: Optional[str] = None) -> ActionResult:
        self.actions_performed.append((element_id, action))
        return ActionResult(success=True, action=action, element_id=element_id, message=f"Performed {action}")

    def set_text(self, element_id: str, text: str, snapshot_id: Optional[str] = None) -> ActionResult:
        self.texts_set.append((element_id, text))
        return ActionResult(success=True, action="set_text", element_id=element_id, message=f"Set text '{text}'")

    def insert_text(self, element_id: str, text: str, snapshot_id: Optional[str] = None) -> ActionResult:
        self.texts_set.append((element_id, text))
        return ActionResult(success=True, action="insert_text", element_id=element_id, message=f"Inserted text '{text}'")

    def focus_element(self, element_id: str, snapshot_id: Optional[str] = None) -> ActionResult:
        return ActionResult(success=True, action="focus", element_id=element_id)

    def select_element(self, element_id: str, value: Optional[str] = None, snapshot_id: Optional[str] = None) -> ActionResult:
        return ActionResult(success=True, action="select", element_id=element_id)

    def scroll(self, element_id: str, direction: str, snapshot_id: Optional[str] = None) -> ActionResult:
        return ActionResult(success=True, action="scroll", element_id=element_id)


def test_desktop_agent_happy_path() -> None:
    # Setup Kate with text editor
    snap1 = DesktopSnapshot(
        active_application="Kate",
        active_window="Untitled — Kate",
        elements=[
            UIElement(id="e1", role="button", name="Save", actions=["click"]),
            UIElement(id="e2", role="text field", name="Editor", states=["editable"], focused=True),
        ],
    )
    backend = MockAccessibilityBackend(snapshots=[snap1])
    input_b = MockInputBackend()
    registry = ToolRegistry()
    router = NeedleCommandRouter(registry=registry)

    agent = DesktopAgent(
        config=DesktopConfig(max_steps=5, settle_delay_ms=0),
        router=router,
        accessibility=backend,
        input_backend=input_b,
        registry=registry,
    )

    result = agent.execute_goal("open Kate and write hello")

    assert result.success is True
    assert len(backend.texts_set) == 1
    assert backend.texts_set[0] == ("e2", "hello")
    assert result.steps >= 1


def test_desktop_agent_persian_literal_text_preservation() -> None:
    # Test that Persian text payload is preserved literally without being translated to English
    snap1 = DesktopSnapshot(
        active_application="Notes",
        active_window="Notes",
        elements=[
            UIElement(id="e1", role="text field", name="Note Content", states=["editable"]),
        ],
    )
    backend = MockAccessibilityBackend(snapshots=[snap1])
    input_b = MockInputBackend()
    registry = ToolRegistry()
    router = NeedleCommandRouter(registry=registry)

    agent = DesktopAgent(
        config=DesktopConfig(max_steps=5, settle_delay_ms=0),
        router=router,
        accessibility=backend,
        input_backend=input_b,
        registry=registry,
    )

    # User originally said: نوت را باز کن و بنویس سلام دنیا
    # Translated command might be: Open Notes and write this Persian sentence
    result = agent.execute_goal(
        translated_command="Open Notes and write this Persian sentence",
        original_persian="نوت را باز کن و بنویس سلام دنیا",
    )

    assert result.success is True
    assert len(backend.texts_set) == 1
    # Must preserve exact Persian text 'سلام دنیا'
    assert backend.texts_set[0] == ("e1", "سلام دنیا")


def test_desktop_agent_navigation_click() -> None:
    snap1 = DesktopSnapshot(
        active_application="Dolphin",
        active_window="Home — Dolphin",
        elements=[
            UIElement(id="e1", role="list item", name="Documents", actions=["click"]),
            UIElement(id="e2", role="list item", name="Downloads", actions=["click"]),
        ],
    )
    backend = MockAccessibilityBackend(snapshots=[snap1])
    input_b = MockInputBackend()
    registry = ToolRegistry()
    router = NeedleCommandRouter(registry=registry)

    agent = DesktopAgent(
        config=DesktopConfig(max_steps=5, settle_delay_ms=0),
        router=router,
        accessibility=backend,
        input_backend=input_b,
        registry=registry,
    )

    result = agent.execute_goal("go to Downloads")
    assert result.success is True
    assert ("e2", "click") in backend.actions_performed


def test_desktop_agent_unsupported_accessibility_detection() -> None:
    # Application window with 0 accessible elements
    inaccessible_snap = DesktopSnapshot(
        active_application="ClosedSourceApp",
        active_window="Secret Window",
        elements=[],
    )
    backend = MockAccessibilityBackend(snapshots=[inaccessible_snap])
    input_b = MockInputBackend()
    registry = ToolRegistry()
    router = NeedleCommandRouter(registry=registry)

    agent = DesktopAgent(
        config=DesktopConfig(max_steps=5, settle_delay_ms=0),
        router=router,
        accessibility=backend,
        input_backend=input_b,
        registry=registry,
    )

    result = agent.execute_goal("click Login")
    assert result.success is False
    assert result.error == "unsupported_accessibility"
    assert "Accessibility unavailable for window" in result.message


def test_desktop_agent_unavailable_subsystem() -> None:
    backend = MockAccessibilityBackend()
    backend._available = False
    input_b = MockInputBackend()
    registry = ToolRegistry()
    router = NeedleCommandRouter(registry=registry)

    agent = DesktopAgent(
        config=DesktopConfig(max_steps=5, settle_delay_ms=0),
        router=router,
        accessibility=backend,
        input_backend=input_b,
        registry=registry,
    )

    result = agent.execute_goal("open Kate")
    assert result.success is False
    assert result.error == "accessibility_unavailable"


def test_desktop_agent_calculator_calculation() -> None:
    # Setup calculator window
    snap = DesktopSnapshot(
        active_application="kcalc",
        active_window="KCalc",
        elements=[
            UIElement(id="e1", role="text field", name="Display", states=["editable"], focused=True),
            UIElement(id="e2", role="button", name="Equal", actions=["click"]),
        ],
    )
    backend = MockAccessibilityBackend(snapshots=[snap])
    input_b = MockInputBackend()
    registry = ToolRegistry()
    router = NeedleCommandRouter(registry=registry)

    agent = DesktopAgent(
        config=DesktopConfig(max_steps=5, settle_delay_ms=0),
        router=router,
        accessibility=backend,
        input_backend=input_b,
        registry=registry,
    )

    result = agent.execute_goal("calculate 123 times 456")
    assert result.success is True
    assert len(backend.texts_set) == 1
    assert backend.texts_set[0] == ("e1", "123 * 456")


def test_desktop_agent_stale_element_recovery() -> None:
    # First snapshot returns an element that will be rejected as stale,
    # then second snapshot updates and agent successfully acts.
    snap1 = DesktopSnapshot(
        active_application="Kate",
        active_window="Untitled — Kate",
        elements=[
            UIElement(id="e99", role="button", name="Old Button", actions=["click"]),
        ],
    )
    snap2 = DesktopSnapshot(
        active_application="Kate",
        active_window="Untitled — Kate",
        elements=[
            UIElement(id="e1", role="button", name="Save", actions=["click"]),
        ],
    )

    class CustomRouter(NeedleCommandRouter):
        def __init__(self, registry):
            super().__init__(registry=registry)
            self.turn = 0

        def decide_desktop_action(self, goal, desktop_state, tool_schemas, original_persian="", action_history=None):
            self.turn += 1
            if self.turn == 1:
                # Deliberately return stale element_id not in current snapshot!
                return "activate_ui_element", {"element_id": "e_stale"}
            if self.turn == 2:
                # On recovery, select valid element
                return "activate_ui_element", {"element_id": "e1"}
            return "done", {"summary": "Saved successfully"}

    backend = MockAccessibilityBackend(snapshots=[snap1, snap2])
    input_b = MockInputBackend()
    registry = ToolRegistry()
    router = CustomRouter(registry=registry)

    agent = DesktopAgent(
        config=DesktopConfig(max_steps=5, settle_delay_ms=0),
        router=router,
        accessibility=backend,
        input_backend=input_b,
        registry=registry,
    )

    result = agent.execute_goal("click Save")
    assert result.success is True
    assert ("e1", "click") in backend.actions_performed

