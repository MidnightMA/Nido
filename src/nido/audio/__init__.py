"""Audio recording and normalization subsystem."""

from nido.audio.capture import MicrophoneCapture, MockMicrophoneCapture
from nido.audio.recorder import AudioRecorder, AudioRecordingError, resample_audio

__all__ = [
    "AudioRecorder",
    "AudioRecordingError",
    "MicrophoneCapture",
    "MockMicrophoneCapture",
    "resample_audio",
]
