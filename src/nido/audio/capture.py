"""Unified microphone audio capture with streaming chunk delivery and in-memory pre-buffering."""

from __future__ import annotations

import collections
import threading
import time
from typing import Callable, List, Optional

import numpy as np

try:
    import sounddevice as sd
except (ImportError, OSError):
    sd = None  # type: ignore[assignment]

from nido.audio.recorder import AudioRecordingError, resample_audio
from nido.logging import get_logger

logger = get_logger("nido.audio.capture")

AudioChunkCallback = Callable[[np.ndarray, int], None]


class MicrophoneCapture:
    """Manages single-device continuous or bounded microphone capture.

    Maintains an in-memory ring pre-buffer so that speech uttered during key press
    classification is never lost. Dispatches small PCM chunks (50-100ms) to registered
    listeners without writing any audio to disk.
    """

    def __init__(
        self,
        device: Optional[str | int] = None,
        target_sr: int = 16000,
        chunk_duration_ms: int = 50,
        prebuffer_ms: int = 400,
    ) -> None:
        self.device = device if device != "" else None
        self.target_sr = target_sr
        self.chunk_duration_ms = max(20, min(200, chunk_duration_ms))
        self.prebuffer_ms = max(100, min(2000, prebuffer_ms))

        self._lock = threading.Lock()
        self._stream: Optional[sd.InputStream] = None
        self._actual_sr: int = target_sr
        self._running = False

        # Calculate max pre-buffer samples and chunks
        samples_per_chunk = int(self.target_sr * (self.chunk_duration_ms / 1000.0))
        max_prebuffer_samples = int(self.target_sr * (self.prebuffer_ms / 1000.0))
        self._max_prebuffer_chunks = max(1, int(np.ceil(max_prebuffer_samples / max(1, samples_per_chunk))))

        self._prebuffer: collections.deque[np.ndarray] = collections.deque(maxlen=self._max_prebuffer_chunks)
        self._listeners: List[AudioChunkCallback] = []

    @property
    def is_active(self) -> bool:
        with self._lock:
            return self._running

    def add_listener(self, listener: AudioChunkCallback) -> None:
        with self._lock:
            if listener not in self._listeners:
                self._listeners.append(listener)

    def remove_listener(self, listener: AudioChunkCallback) -> None:
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    def clear_listeners(self) -> None:
        with self._lock:
            self._listeners.clear()

    def get_prebuffer(self) -> np.ndarray:
        """Return all audio samples currently held in the pre-buffer as 1D float32 array."""
        with self._lock:
            if not self._prebuffer:
                return np.zeros(0, dtype=np.float32)
            return np.concatenate(list(self._prebuffer)).astype(np.float32)

    def clear_prebuffer(self) -> None:
        """Clear the in-memory pre-buffer."""
        with self._lock:
            self._prebuffer.clear()

    def start(self) -> None:
        """Start microphone capture."""
        with self._lock:
            if self._running:
                return

            if sd is None:
                raise AudioRecordingError("sounddevice is not available or audio system is missing.")

            self._prebuffer.clear()
            self._running = True

            try:
                dev_info = sd.query_devices(self.device, "input")
                self._actual_sr = int(dev_info.get("default_samplerate", self.target_sr))
            except Exception as e:
                logger.warning(f"Could not query input device sample rate: {e}. Defaulting to {self.target_sr}.")
                self._actual_sr = self.target_sr

            blocksize = int(self._actual_sr * (self.chunk_duration_ms / 1000.0))

            def _audio_callback(
                indata: np.ndarray,
                frames: int,
                time_info: object,
                status: sd.CallbackFlags,
            ) -> None:
                if status:
                    logger.debug(f"Audio callback status: {status}")

                with self._lock:
                    if not self._running:
                        return

                    # Convert to mono float32
                    if indata.ndim > 1 and indata.shape[1] > 1:
                        mono = np.mean(indata, axis=1, dtype=np.float32)
                    else:
                        mono = indata.flatten().astype(np.float32)

                    # Resample to 16kHz if needed
                    if self._actual_sr != self.target_sr:
                        processed = resample_audio(mono, self._actual_sr, self.target_sr)
                    else:
                        processed = mono.copy()

                    np.clip(processed, -1.0, 1.0, out=processed)

                    # Maintain in-memory pre-buffer
                    self._prebuffer.append(processed)
                    current_listeners = list(self._listeners)

                # Dispatch chunks to listeners outside lock
                for callback in current_listeners:
                    try:
                        callback(processed, self.target_sr)
                    except Exception as err:
                        logger.error(f"Error in audio listener callback: {err}")

            try:
                self._stream = sd.InputStream(
                    samplerate=self._actual_sr,
                    channels=1,
                    dtype="float32",
                    device=self.device,
                    blocksize=blocksize,
                    callback=_audio_callback,
                )
                self._stream.start()
                logger.debug(f"Microphone stream started (actual_sr={self._actual_sr}, target_sr={self.target_sr}).")
            except Exception as e:
                self._running = False
                self._prebuffer.clear()
                raise AudioRecordingError(f"Failed to open audio input stream: {e}") from e

    def stop(self) -> None:
        """Stop microphone capture and release stream."""
        with self._lock:
            if not self._running:
                return
            self._running = False
            stream = self._stream
            self._stream = None

        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception as e:
                logger.warning(f"Error closing audio stream: {e}")
        logger.debug("Microphone stream stopped.")


class MockMicrophoneCapture(MicrophoneCapture):
    """In-memory mock microphone capture for testing without audio hardware."""

    def __init__(
        self,
        target_sr: int = 16000,
        chunk_duration_ms: int = 50,
        prebuffer_ms: int = 400,
    ) -> None:
        super().__init__(
            device=None,
            target_sr=target_sr,
            chunk_duration_ms=chunk_duration_ms,
            prebuffer_ms=prebuffer_ms,
        )

    def start(self) -> None:
        with self._lock:
            self._running = True

    def stop(self) -> None:
        with self._lock:
            self._running = False

    def simulate_chunk(self, chunk: np.ndarray) -> None:
        """Simulate feeding an incoming chunk to listeners and prebuffer."""
        with self._lock:
            if not self._running:
                return
            arr = chunk.astype(np.float32)
            self._prebuffer.append(arr)
            listeners = list(self._listeners)

        for cb in listeners:
            cb(arr, self.target_sr)
