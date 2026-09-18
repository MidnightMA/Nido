"""Tests for audio capture and normalization."""

import numpy as np
from nido.audio.recorder import AudioRecorder, resample_audio


def test_resample_audio_48k_to_16k() -> None:
    # 48000 samples = 1 second at 48kHz
    audio_48k = np.sin(np.linspace(0, 2 * np.pi * 440, 48000, dtype=np.float32))
    resampled_16k = resample_audio(audio_48k, orig_sr=48000, target_sr=16000)

    assert len(resampled_16k) == 16000
    assert resampled_16k.dtype == np.float32


def test_resample_same_sample_rate() -> None:
    audio = np.array([0.1, -0.2, 0.5], dtype=np.float32)
    res = resample_audio(audio, orig_sr=16000, target_sr=16000)
    assert np.array_equal(audio, res)


def test_resample_empty_audio() -> None:
    empty = np.zeros(0, dtype=np.float32)
    res = resample_audio(empty, orig_sr=48000, target_sr=16000)
    assert len(res) == 0


def test_recorder_configuration() -> None:
    rec = AudioRecorder(device=None, target_sr=16000, max_seconds=10)
    assert rec.target_sr == 16000
    assert rec.max_seconds == 10
    assert not rec.is_recording

    # Stop when not started returns empty float32 array
    audio = rec.stop()
    assert isinstance(audio, np.ndarray)
    assert len(audio) == 0
