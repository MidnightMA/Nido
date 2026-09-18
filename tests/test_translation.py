"""Tests for Persian->English translation subsystem."""

from nido.translation import (
    MockTranslator,
    Translator,
    preprocess_persian_text,
)


def test_preprocess_persian_text() -> None:
    raw = "  كروم    را باز   كن  "
    cleaned = preprocess_persian_text(raw)
    assert cleaned == "کروم را باز کن"
    assert "ك" not in cleaned
    assert "ي" not in cleaned


def test_mock_translator_conformance() -> None:
    translator: Translator = MockTranslator()
    res = translator.translate("کروم را باز کن")
    assert res == "Open Chrome"

    res_browser = translator.translate("مرورگر رو باز کن")
    assert res_browser == "Open browser"

    res_vol = translator.translate("صدا رو کم کن")
    assert res_vol == "Decrease volume"

    res_vol_30 = translator.translate("صدا رو بذار روی 30 درصد")
    assert res_vol_30 == "Set volume to 30 percent"


def test_translator_empty_input() -> None:
    translator: Translator = MockTranslator()
    assert translator.translate("") == ""
    assert translator.translate("   ") == ""
