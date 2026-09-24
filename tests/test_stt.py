"""Tests for English streaming Speech-to-Text abstraction and mock recognizer."""

from unittest.mock import MagicMock
import numpy as np
from nido.config import STTConfig
from nido.stt import (
    MockStreamingSTT,
    NemotronStreamingSTT,
    StreamingSTT,
    WhisperStreamingSTT,
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


def test_whisper_streaming_stt_protocol_and_lifecycle() -> None:
    """Verify WhisperStreamingSTT protocol adherence, buffering, and silence tracking."""
    rec = WhisperStreamingSTT(STTConfig(model_dir="/non/existent/path"))
    assert rec.is_loaded is False

    rec.start_session()
    assert rec.get_partial_text() == ""
    assert rec.is_endpoint() is False

    # Feed speech chunk (high amplitude)
    speech_chunk = np.ones(1600, dtype=np.float32) * 0.2
    rec.feed_audio(speech_chunk)
    assert rec._speech_detected is True
    assert rec._silence_duration_s == 0.0
    assert rec.is_endpoint() is False

    # Feed silence chunk
    silence_chunk = np.zeros(16000, dtype=np.float32)  # 1.0s silence > 0.8s threshold
    rec.feed_audio(silence_chunk)
    assert rec.is_endpoint() is True

    # Finalize resets
    final = rec.finalize()
    assert final == ""
    assert rec.is_endpoint() is False
    rec.stop_session()


def test_whisper_transcribe_real_audio() -> None:
    """Verify Whisper.cpp transcribes real audio using installed model and whisper-cli."""
    from pathlib import Path
    import wave
    test_wav = Path.home() / ".local/share/nido/NeMo-Speech.cpp/test_files/asr/wav/test/jfk.wav"
    if not test_wav.exists():
        return
    with wave.open(str(test_wav), "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        audio_i16 = np.frombuffer(frames, dtype=np.int16)
        audio_f32 = audio_i16.astype(np.float32) / 32768.0

    rec = WhisperStreamingSTT()
    if rec.is_loaded:
        text = rec.transcribe_waveform(audio_f32, 16000)
        assert "country" in text.lower()
























































