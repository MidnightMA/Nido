"""Tests for English streaming Speech-to-Text abstraction and mock recognizer."""

from unittest.mock import MagicMock
import numpy as np
from nido.config import STTConfig
from nido.stt import (
    MockStreamingSTT,
    StreamingSTT,
    ZipformerStreamingSTT,
)


def test_mock_streaming_stt_lifecycle() -> None:
    stt: StreamingSTT = MockStreamingSTT(["open notes", "create a new note"])
    assert stt.is_loaded is True

    # 1. Start session
    stt.start_session()
    assert stt.get_partial_text() == "open notes"
    assert stt.is_endpoint() is False

    # 2. Feed audio until endpoint triggers
    chunk = np.zeros(4000, dtype=np.float32)
    stt.feed_audio(chunk)
    assert stt.is_endpoint() is False

    # Feed more audio to exceed endpoint trigger threshold
    stt.feed_audio(chunk)
    assert stt.is_endpoint() is True

    # 3. Finalize utterance
    final = stt.finalize()
    assert final == "open notes"

    # 4. Next utterance
    assert stt.get_partial_text() == "create a new note"

    stt.stop_session()


def test_mock_streaming_transcribe_waveform() -> None:
    stt = MockStreamingSTT(["open calculator"])
    dummy_audio = np.zeros(16000, dtype=np.float32)
    result = stt.transcribe_waveform(dummy_audio)
    assert result == "open calculator"

    # Empty audio returns empty string
    empty_audio = np.zeros(0, dtype=np.float32)
    assert stt.transcribe_waveform(empty_audio) == ""


def test_zipformer_extract_text_uses_get_result() -> None:
    """Ensure ZipformerStreamingSTT calls recognizer.get_result(stream) and not stream.result."""
    rec = ZipformerStreamingSTT(STTConfig(model_dir="/non/existent/path"))
    assert rec.is_loaded is False

    # Mock recognizer and stream
    mock_recognizer = MagicMock()
    mock_result_obj = MagicMock()
    mock_result_obj.text = "  hello world  "
    mock_recognizer.get_result.return_value = mock_result_obj
    mock_recognizer.is_ready.return_value = False

    mock_stream = MagicMock(spec=["input_finished"])  # Spec only input_finished so stream.result would raise AttributeError

    rec._recognizer = mock_recognizer
    rec._stream = mock_stream

    # Calling get_partial_text must succeed and extract text via recognizer.get_result
    partial = rec.get_partial_text()
    assert partial == "hello world"
    mock_recognizer.get_result.assert_called_with(mock_stream)

    # Calling finalize
    final = rec.finalize()
    assert final == "hello world"
