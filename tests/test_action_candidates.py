"""Unit tests for ActionCandidate construction and deterministic argument binding."""

from nido.accessibility.models import DesktopSnapshot, UIElement
from nido.desktop.candidates import (
    CandidateBuilder,
    extract_calculation_expression,
    extract_literal_payload,
    extract_url_or_search,
    extract_volume_percent,
)
from nido.tools.registry import ToolRegistry


def test_extract_literal_payload_persian() -> None:
    # Persian typing verb
    res1 = extract_literal_payload("نوت را باز کن و بنویس سلام دنیا")
    assert res1 == "سلام دنیا"

    # Persian quotes
    res2 = extract_literal_payload('ویرایشگر را باز کن و تایپ کن «متن تستی»')
    assert res2 == "متن تستی"

    # Persian polite ending stripped
    res3 = extract_literal_payload("بنویس پروژه جدید لطفا")
    assert res3 == "پروژه جدید"


def test_extract_literal_payload_mixed_and_english() -> None:
    # Mixed Persian and English
    res = extract_literal_payload("کروم رو باز کن و بنویس Hello, دنیا!")
    assert res == "Hello, دنیا!"

    # English quotes
    res2 = extract_literal_payload('open editor and write "def test():"')
    assert res2 == "def test():"


def test_extract_volume_percent() -> None:
    # Persian digits
    assert extract_volume_percent("صدا رو بذار روی ۳۰ درصد") == 30

    # Persian number word
    assert extract_volume_percent("صدا را روی پنجاه درصد بگذار") == 50

    # Standard digits
    assert extract_volume_percent("set volume to 75%") == 75


def test_extract_url_or_search() -> None:
    url, query = extract_url_or_search("برو به https://github.com/test")
    assert url == "https://github.com/test"
    assert query is None

    url2, query2 = extract_url_or_search("یوتیوب رو باز کن")
    assert url2 == "https://youtube.com"

    url3, query3 = extract_url_or_search("جستجو کن درباره پایتون 3.12")
    assert url3 is None
    assert query3 == "پایتون 3.12"


def test_extract_calculation_expression() -> None:
    # English calculation
    assert extract_calculation_expression("open calculator and calculate 2*95") == "2 * 95"
    assert extract_calculation_expression("calculate 123 times 456") == "123 * 456"

    # Persian calculation
    assert extract_calculation_expression("ماشین حساب را باز کن و حساب کن ۲ ضربدر ۹۵") == "2 * 95"
    assert extract_calculation_expression("حساب کن ۱۰۰ تقسیم بر ۴") == "100 / 4"


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
        goal="نوت را باز کن و بنویس سلام دنیا",
        snapshot=snap,
    )

    # Validate candidates have sequential IDs A1, A2...
    ids = [c.id for c in candidates]
    assert ids == [f"A{i+1}" for i in range(len(candidates))]

    # Must contain set_ui_text with exact Persian payload
    text_cands = [c for c in candidates if c.action_type == "set_ui_text"]
    assert len(text_cands) >= 1
    assert text_cands[0].arguments["text"] == "سلام دنیا"
    assert text_cands[0].arguments["element_id"] == "e2"

    # Must contain terminal done and wait candidates
    assert any(c.action_type == "done" for c in candidates)
    assert any(c.action_type == "wait" for c in candidates)
