"""Unit tests for unified Laya-driven DesktopAgent loop, text preservation, and recovery."""

from typing import List, Optional
import pytest

from nido.accessibility.base import AccessibilityBackend
from nido.accessibility.models import ActionResult, DesktopSnapshot, UIElement
from nido.config import DesktopConfig
from nido.desktop.agent import DesktopAgent, DesktopAgentResult
from nido.desktop.candidates import ActionCandidate, CandidateBuilder
from nido.desktop.input import MockInputBackend
from nido.laya.agent import ActionDecision, MockLayaAgent
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
    laya = MockLayaAgent()

    agent = DesktopAgent(
        config=DesktopConfig(max_steps=5, settle_delay_ms=0),
        laya_agent=laya,
        accessibility=backend,
        input_backend=input_b,
        registry=registry,
    )

    result = agent.execute_goal("Open Kate and write hello")

    assert result.success is True
    assert len(backend.texts_set) == 1
    assert backend.texts_set[0] == ("e2", "hello")
    assert result.steps >= 1


def test_desktop_agent_english_literal_text_preservation() -> None:
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
    laya = MockLayaAgent()

    agent = DesktopAgent(
        config=DesktopConfig(max_steps=5, settle_delay_ms=0),
        laya_agent=laya,
        accessibility=backend,
        input_backend=input_b,
        registry=registry,
    )

    result = agent.execute_goal(
        goal="Open Notes and write Hello, world!",
    )

    assert result.success is True
    assert len(backend.texts_set) == 1
    # Must preserve exact English literal text 'Hello, world!'
    assert backend.texts_set[0] == ("e1", "Hello, world!")


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
    laya = MockLayaAgent()

    agent = DesktopAgent(
        config=DesktopConfig(max_steps=5, settle_delay_ms=0),
        laya_agent=laya,
        accessibility=backend,
        input_backend=input_b,
        registry=registry,
    )

    result = agent.execute_goal("Go to Downloads")
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
    laya = MockLayaAgent()

    agent = DesktopAgent(
        config=DesktopConfig(max_steps=5, settle_delay_ms=0),
        laya_agent=laya,
        accessibility=backend,
        input_backend=input_b,
        registry=registry,
    )

    result = agent.execute_goal("Click on button")
    assert result.success is False
    assert result.error == "unsupported_accessibility"
    assert "Accessibility unavailable for window" in result.message


def test_desktop_agent_unavailable_subsystem() -> None:
    backend = MockAccessibilityBackend()
    backend._available = False
    input_b = MockInputBackend()
    registry = ToolRegistry()
    laya = MockLayaAgent()

    agent = DesktopAgent(
        config=DesktopConfig(max_steps=5, settle_delay_ms=0),
        laya_agent=laya,
        accessibility=backend,
        input_backend=input_b,
        registry=registry,
    )

    result = agent.execute_goal("Open kate")
    assert result.success is False
    assert result.error == "accessibility_unavailable"


def test_desktop_agent_stale_element_recovery() -> None:
    # First snapshot contains an element that gets removed in second snapshot
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

    class StaleTestingLaya(MockLayaAgent):
        def __init__(self):
            super().__init__()
            self.turn = 0

        def predict_action(self, state_text, candidates):
            self.turn += 1
            if self.turn == 1:
                return ActionDecision(selected_id="A_stale", confidence=0.9)
            elif self.turn == 2:
                for c in candidates:
                    if c.element_id == "e1":
                        return ActionDecision(selected_id=c.id, confidence=0.95)
            done_cand = next((c for c in candidates if c.action_type == "done"), candidates[0])
            return ActionDecision(selected_id=done_cand.id, confidence=0.99)

    backend = MockAccessibilityBackend(snapshots=[snap1, snap2])
    input_b = MockInputBackend()
    registry = ToolRegistry()
    laya = StaleTestingLaya()

    agent = DesktopAgent(
        config=DesktopConfig(max_steps=5, settle_delay_ms=0),
        laya_agent=laya,
        accessibility=backend,
        input_backend=input_b,
        registry=registry,
    )

    result = agent.execute_goal("Save it")
    assert result.success is True
    assert ("e1", "click") in backend.actions_performed


def test_desktop_agent_max_steps_exceeded() -> None:
    snap = DesktopSnapshot(
        active_application="App",
        active_window="App Window",
        elements=[
            UIElement(id="e1", role="button", name="Loop", actions=["click"]),
        ],
    )

    class InfiniteWaitLaya(MockLayaAgent):
        def predict_action(self, state_text, candidates):
            wait_c = next((c for c in candidates if c.action_type == "wait"), candidates[0])
            return ActionDecision(selected_id=wait_c.id, confidence=0.5)

    backend = MockAccessibilityBackend(snapshots=[snap])
    agent = DesktopAgent(
        config=DesktopConfig(max_steps=3, settle_delay_ms=0),
        laya_agent=InfiniteWaitLaya(),
        accessibility=backend,
        input_backend=MockInputBackend(),
        registry=ToolRegistry(),
    )

    res = agent.execute_goal("Do it")
    assert res.success is False
    assert res.error == "max_steps_exceeded"
    assert res.steps == 3


def test_desktop_agent_calculator_workflow() -> None:
    # Step 1: konsole is active
    snap1 = DesktopSnapshot(
        active_application="konsole",
        active_window="Konsole",
        elements=[],
    )
    # Step 2: kcalc is active with Display text field
    snap2 = DesktopSnapshot(
        active_application="kcalc",
        active_window="KCalc",
        elements=[
            UIElement(id="e1", role="text field", name="Display", states=["editable"], focused=True),
            UIElement(id="e2", role="button", name="Equal", actions=["click"]),
        ],
    )

    backend = MockAccessibilityBackend(snapshots=[snap1, snap2])
    input_b = MockInputBackend()
    registry = ToolRegistry()

    @registry.register(name="open_app")
    def mock_open_app(app_name: str) -> dict:
        return {"success": True, "message": f"Application '{app_name}' launched."}

    agent = DesktopAgent(
        config=DesktopConfig(max_steps=5, settle_delay_ms=0),
        laya_agent=MockLayaAgent(),
        accessibility=backend,
        input_backend=input_b,
        registry=registry,
        candidate_builder=CandidateBuilder(app_map={"calculator": "kcalc"}),
    )

    result = agent.execute_goal("open calculator and calculate 2*95")
    assert result.success is True
    assert result.steps <= 3
    open_app_count = sum(1 for h in result.history if h.get("action") == "open_app")
    assert open_app_count == 1
    assert len(backend.texts_set) == 1
    assert "2 * 95" in backend.texts_set[0][1]
