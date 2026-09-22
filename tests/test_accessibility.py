"""Unit tests for accessibility models, snapshotting, and dynamic action spaces."""

import pytest
from nido.accessibility.models import ActionResult, DesktopSnapshot, UIElement
from nido.accessibility.snapshot import compute_snapshot_diff, format_snapshot_for_prompt
from nido.desktop.tool_builder import DynamicDesktopToolBuilder


def test_ui_element_and_snapshot_creation() -> None:
    el1 = UIElement(
        id="e1",
        role="button",
        name="Save",
        actions=["click", "activate"],
        states=["enabled", "visible", "showing"],
    )
    el2 = UIElement(
        id="e2",
        role="text field",
        name="Editor",
        states=["enabled", "visible", "showing", "editable", "focused"],
        focused=True,
    )
    snapshot = DesktopSnapshot(
        session_type="wayland",
        desktop_name="KDE Plasma",
        active_application="Kate",
        active_window="Untitled — Kate",
        focused_element_id="e2",
        elements=[el1, el2],
    )

    assert snapshot.get_element("e1") == el1
    assert snapshot.get_element("e2") == el2
    assert snapshot.get_element("e99") is None
    assert snapshot.focused_element_id == "e2"


def test_format_snapshot_for_prompt() -> None:
    el1 = UIElement(id="e1", role="button", name="Save", states=["enabled", "visible"])
    el2 = UIElement(id="e2", role="text field", name="Editor", states=["editable", "focused"], focused=True)
    snapshot = DesktopSnapshot(
        session_type="wayland",
        desktop_name="KDE Plasma",
        active_application="Kate",
        active_window="Untitled — Kate",
        focused_element_id="e2",
        elements=[el1, el2],
    )

    formatted = format_snapshot_for_prompt(snapshot)
    assert "session: wayland" in formatted
    assert "desktop: KDE Plasma" in formatted
    assert "active application: Kate" in formatted
    assert "active window: Untitled — Kate" in formatted
    assert '[e1] button "Save"' in formatted
    assert '[e2] text field "Editor"' in formatted
    assert "(focused" in formatted


def test_format_snapshot_unsupported_window() -> None:
    snapshot = DesktopSnapshot(
        session_type="wayland",
        desktop_name="KDE Plasma",
        active_application="UnknownApp",
        active_window="Unknown Window",
        elements=[],
    )
    formatted = format_snapshot_for_prompt(snapshot)
    assert "Accessibility unavailable for this window." in formatted


def test_compute_snapshot_diff() -> None:
    el1 = UIElement(id="e1", role="button", name="Open", states=["enabled"])
    snap1 = DesktopSnapshot(
        active_application="Dolphin",
        active_window="Downloads",
        elements=[el1],
    )

    el2 = UIElement(id="e1", role="button", name="Open", states=["enabled"])
    el3 = UIElement(id="e2", role="button", name="Cancel", states=["enabled"])
    snap2 = DesktopSnapshot(
        active_application="Dolphin",
        active_window="Open File",
        elements=[el2, el3],
    )

    diff = compute_snapshot_diff(snap1, snap2)
    assert "Downloads" in diff or "Open File" in diff
    assert "Cancel" in diff


def test_dynamic_action_space_changes_across_snapshots() -> None:
    builder = DynamicDesktopToolBuilder()

    # Snapshot A has e1, e2
    snap_a = DesktopSnapshot(
        elements=[
            UIElement(id="e1", role="button", name="New", actions=["click"]),
            UIElement(id="e2", role="button", name="Open", actions=["click"]),
        ]
    )
    schemas_a = builder.build_tool_schemas(snap_a)
    act_tool_a = next(t for t in schemas_a if t["name"] == "activate_ui_element")
    assert act_tool_a["parameters"]["properties"]["element_id"]["enum"] == ["e1", "e2"]

    # Snapshot B has e1, e2, e3, e4
    snap_b = DesktopSnapshot(
        elements=[
            UIElement(id="e1", role="dialog", name="Open File"),
            UIElement(id="e2", role="text field", name="File name", states=["editable"]),
            UIElement(id="e3", role="button", name="Cancel", actions=["click"]),
            UIElement(id="e4", role="button", name="Open", actions=["click"]),
        ]
    )
    schemas_b = builder.build_tool_schemas(snap_b)
    act_tool_b = next(t for t in schemas_b if t["name"] == "activate_ui_element")
    assert act_tool_b["parameters"]["properties"]["element_id"]["enum"] == ["e3", "e4"]

    set_tool_b = next(t for t in schemas_b if t["name"] == "set_ui_text")
    assert set_tool_b["parameters"]["properties"]["element_id"]["enum"] == ["e2"]


def test_stale_element_validation() -> None:
    builder = DynamicDesktopToolBuilder()
    snap = DesktopSnapshot(
        elements=[UIElement(id="e1", role="button", name="Save", actions=["click"])]
    )

    # Valid element_id
    ok, err = builder.validate_call("activate_ui_element", {"element_id": "e1"}, snap)
    assert ok is True
    assert err is None

    # Stale element_id (e99 is not in snap)
    ok_stale, err_stale = builder.validate_call("activate_ui_element", {"element_id": "e99"}, snap)
    assert ok_stale is False
    assert "stale_ui_element" in err_stale


def test_action_validation_rules() -> None:
    builder = DynamicDesktopToolBuilder()
    snap = DesktopSnapshot(
        elements=[UIElement(id="e1", role="text field", name="Input", states=["editable"])]
    )

    # Missing required text
    ok, err = builder.validate_call("set_ui_text", {"element_id": "e1"}, snap)
    assert ok is False
    assert "requires string argument 'text'" in err

    # Unknown action
    ok_unk, err_unk = builder.validate_call("malicious_shell_exec", {"cmd": "ls"}, snap)
    assert ok_unk is False
    assert "Unknown desktop action" in err_unk
