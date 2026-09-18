"""Tests for Persian STT normalization and number handling."""

import numpy as np
from nido.stt import (
    MockSpeechRecognizer,
    SpeechRecognizer,
    normalize_persian_numbers,
)


def test_persian_number_normalization_single_words() -> None:
    assert normalize_persian_numbers("صفر") == "0"
    assert normalize_persian_numbers("ده") == "10"
    assert normalize_persian_numbers("بیست") == "20"
    assert normalize_persian_numbers("سی") == "30"
    assert normalize_persian_numbers("چهل") == "40"
    assert normalize_persian_numbers("پنجاه") == "50"
    assert normalize_persian_numbers("صد") == "100"


def test_persian_compound_numbers() -> None:
    # "سی و پنج" -> "35"
    assert normalize_persian_numbers("سی و پنج") == "35"
    # "صد و بیست" -> "120"
    assert normalize_persian_numbers("صد و بیست") == "120"
    # "چهل و دو" -> "42"
    assert normalize_persian_numbers("چهل و دو") == "42"


def test_persian_sentence_with_spoken_number() -> None:
    input_text = "صدا رو بذار روی سی درصد"
    normalized = normalize_persian_numbers(input_text)
    assert normalized == "صدا رو بذار روی 30 درصد"

    input_text_50 = "صدا رو بذار روی پنجاه درصد"
    assert normalize_persian_numbers(input_text_50) == "صدا رو بذار روی 50 درصد"


def test_persian_digit_conversion() -> None:
    assert normalize_persian_numbers("۱۲۳") == "123"
    assert normalize_persian_numbers("شماره ۴۵۶") == "شماره 456"


def test_mock_speech_recognizer() -> None:
    rec: SpeechRecognizer = MockSpeechRecognizer("صدا رو بذار روی سی درصد")
    dummy_audio = np.zeros(16000, dtype=np.float32)
    result = rec.transcribe(dummy_audio)
    assert result == "صدا رو بذار روی 30 درصد"

    # Empty audio returns empty string
    empty_audio = np.zeros(0, dtype=np.float32)
    assert rec.transcribe(empty_audio) == ""
