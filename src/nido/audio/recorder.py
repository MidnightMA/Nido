"""Microphone audio capture and normalization for Nido."""

from __future__ import annotations

import threading
import time
from typing import List, Optional

import numpy as np

try:
    import sounddevice as sd
except (ImportError, OSError):
    sd = None  # type: ignore[assignment]

from nido.logging import get_logger

logger = get_logger("nido.audio")


class AudioRecordingError(Exception):
    """Raised when audio capture or normalization fails."""


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
    """Resample 1D float32 audio array to target_sr using linear interpolation.

    ponytail: stdlib/numpy linear interpolation used. upgrade path: scipy.signal.resample_poly
    if higher-order anti-aliasing is required.
    """
    if orig_sr == target_sr:
        return audio.astype(np.float32)

    if len(audio) == 0:
        return np.zeros(0, dtype=np.float32)

    num_output_samples = int(round(len(audio) * float(target_sr) / float(orig_sr)))
    if num_output_samples <= 0:
        return np.zeros(0, dtype=np.float32)

    orig_indices = np.linspace(0.0, len(audio) - 1, num=len(audio), endpoint=True)
    target_indices = np.linspace(0.0, len(audio) - 1, num=num_output_samples, endpoint=True)
    resampled = np.interp(target_indices, orig_indices, audio)
    return resampled.astype(np.float32)


class AudioRecorder:
    """Manages push-to-talk in-memory microphone recording."""

    def __init__(
        self,
        device: Optional[str | int] = None,
        target_sr: int = 16000,
        max_seconds: int = 30,
    ) -> None:
        self.device = device if device != "" else None
        self.target_sr = target_sr
        self.max_seconds = max_seconds

        self._recording = False
        self._lock = threading.Lock()
        self._chunks: List[np.ndarray] = []
        self._stream: Optional[sd.InputStream] = None
        self._actual_sr: int = target_sr
        self._start_time: float = 0.0

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._recording

    def start(self) -> None:
        """Begin capturing microphone input into memory."""
        with self._lock:
            if self._recording:
                logger.warning("Recorder already running; start() ignored.")
                return

            if sd is None:
                raise AudioRecordingError("sounddevice is not available or audio system is missing.")

            self._chunks.clear()
            self._recording = True
            self._start_time = time.time()

            try:
                # Query device sample rate or fallback to 16000
                dev_info = sd.query_devices(self.device, "input")
                self._actual_sr = int(dev_info.get("default_samplerate", self.target_sr))
            except Exception as e:
                logger.warning(f"Could not query input device sample rate: {e}. Defaulting to 16000.")
                self._actual_sr = self.target_sr

            def _audio_callback(indata: np.ndarray, frames: int, time_info: object, status: sd.CallbackFlags) -> None:
                if status:
                    logger.debug(f"Audio callback status: {status}")
                with self._lock:
                    if not self._recording:
                        return
                    # indata shape is (frames, channels)
                    # Convert to mono by taking mean across channels if multi-channel
                    if indata.ndim > 1 and indata.shape[1] > 1:
                        mono = np.mean(indata, axis=1, dtype=np.float32)
                    else:
                        mono = indata.flatten().astype(np.float32)
                    self._chunks.append(mono.copy())

                    # Check max_seconds limit
                    if time.time() - self._start_time >= self.max_seconds:
                        logger.info(f"Max recording duration ({self.max_seconds}s) reached. Stopping automatically.")
                        self._recording = False

            try:
                self._stream = sd.InputStream(
                    samplerate=self._actual_sr,
                    channels=1,
                    dtype="float32",
                    device=self.device,
                    callback=_audio_callback,
                )
                self._stream.start()
                logger.debug("Microphone stream started.")
            except Exception as e:
                self._recording = False
                self._chunks.clear()
                raise AudioRecordingError(f"Failed to open audio input stream: {e}") from e

    def stop(self) -> np.ndarray:
        """Stop capturing and return normalized float32 mono audio at 16000 Hz."""
        with self._lock:
            if not self._recording and not self._chunks:
                logger.debug("Stop called while not recording.")
                return np.zeros(0, dtype=np.float32)

            self._recording = False
            stream = self._stream
            self._stream = None
            chunks = list(self._chunks)
            self._chunks.clear()

        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception as e:
                logger.warning(f"Error closing audio stream: {e}")

        if not chunks:
            logger.debug("No audio chunks collected.")
            return np.zeros(0, dtype=np.float32)

        raw = np.concatenate(chunks).astype(np.float32)

        # Resample to 16000 Hz if needed
        if self._actual_sr != self.target_sr:
            logger.debug(f"Resampling audio from {self._actual_sr}Hz to {self.target_sr}Hz.")
            raw = resample_audio(raw, self._actual_sr, self.target_sr)

        # Clip to [-1.0, 1.0] range
        np.clip(raw, -1.0, 1.0, out=raw)

        duration = len(raw) / float(self.target_sr)
        logger.debug(f"Recorded {duration:.2f}s of normalized audio ({len(raw)} samples).")
        return raw
