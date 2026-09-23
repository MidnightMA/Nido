"""Unit tests for ActionCandidate construction and deterministic argument binding in English."""

from nido.accessibility.models import DesktopSnapshot, UIElement
from nido.desktop.candidates import (
    CandidateBuilder,
    extract_calculation_expression,
    extract_literal_payload,
    extract_url_or_search,
    extract_volume_percent,
)
from nido.tools.registry import ToolRegistry


def test_extract_literal_payload_english() -> None:
    # English typing verb: "Open notes and write Hello world"
    res1 = extract_literal_payload("Open notes and write Hello world")
    assert res1 == "Hello world"

    # Exact literal payload preservation with punctuation and capitalization
    res2 = extract_literal_payload("Open Notes and write Hello, world!")
    assert res2 == "Hello, world!"

    # Single-verb typing: "type This is a test."
    res3 = extract_literal_payload("type This is a test.")
    assert res3 == "This is a test."


def test_extract_literal_payload_quotes() -> None:
    # Quoted text
    res = extract_literal_payload('open editor and write "def test():"')
    assert res == "def test():"

    res2 = extract_literal_payload("enter 'sample text'")
    assert res2 == "sample text"


def test_extract_volume_percent() -> None:
    # Standard digits
    assert extract_volume_percent("set volume to 75%") == 75
    assert extract_volume_percent("volume 30 percent") == 30

    # English number word
    assert extract_volume_percent("set volume to fifty percent") == 50
    assert extract_volume_percent("volume thirty") == 30


def test_extract_url_or_search() -> None:
    url, query = extract_url_or_search("go to https://github.com/test")
    assert url == "https://github.com/test"
    assert query is None

    url2, query2 = extract_url_or_search("open youtube")
    assert url2 == "https://youtube.com"

    url3, query3 = extract_url_or_search("search for python 3.12")
    assert url3 is None
    assert query3 == "python 3.12"


def test_extract_calculation_expression() -> None:
    # English calculation
    assert extract_calculation_expression("open calculator and calculate 2*95") == "2 * 95"
    assert extract_calculation_expression("calculate 123 times 456") == "123 * 456"
    assert extract_calculation_expression("compute 100 / 4") == "100 / 4"


def test_candidate_builder_skips_already_opened_or_active_app() -> None:
    builder = CandidateBuilder(app_map={"calculator": "kcalc", "kate": "kate"})

    # Step 1: Initial state before launch
    snap1 = DesktopSnapshot(active_application="konsole", active_window="Konsole")
    cands1 = builder.build_candidates("open calculator and calculate 2*95", snapshot=snap1)
    assert any(c.action_type == "open_app" and c.arguments.get("app_name") == "kcalc" for c in cands1)

    # Step 2: After kcalc has been launched (in action_history)
    history = [
        {"action": "open_app", "arguments": {"app_name": "kcalc"}, "success": True}
    ]
    snap2 = DesktopSnapshot(
        active_application="kcalc",
        active_window="KCalc",
        elements=[
            UIElement(id="e1", role="text field", name="Display", states=["editable"]),
        ],
    )
    cands2 = builder.build_candidates("open calculator and calculate 2*95", snapshot=snap2, action_history=history)

    # open_app: kcalc MUST NOT be in candidates!
    assert not any(c.action_type == "open_app" and c.arguments.get("app_name") == "kcalc" for c in cands2)

    # set_ui_text with '2 * 95' MUST be present!
    text_cands = [c for c in cands2 if c.action_type == "set_ui_text"]
    assert len(text_cands) >= 1
    assert "2 * 95" in text_cands[0].arguments["text"]


def test_candidate_builder_desktop_and_static() -> None:
    builder = CandidateBuilder(app_map={"kate": "kate", "dolphin": "dolphin"})
    snap = DesktopSnapshot(
        active_application="Kate",
        active_window="Untitled — Kate",
        elements=[
            UIElement(id="e1", role="button", name="Save", actions=["click"]),
            UIElement(id="e2", role="text field", name="Editor", states=["editable"], focused=True),
        ],
    )

    candidates = builder.build_candidates(
        goal="Open notes and write Hello world",
        snapshot=snap,
    )

    # Validate candidates have sequential IDs A1, A2...
    ids = [c.id for c in candidates]
    assert ids == [f"A{i+1}" for i in range(len(candidates))]

    # Must contain set_ui_text with exact English payload
    text_cands = [c for c in candidates if c.action_type == "set_ui_text"]
    assert len(text_cands) >= 1
    assert text_cands[0].arguments["text"] == "Hello world"
    assert text_cands[0].arguments["element_id"] == "e2"

    # Must contain terminal done and wait candidates
    assert any(c.action_type == "done" for c in candidates)
    assert any(c.action_type == "wait" for c in candidates)
