"""English Speech-to-Text using Whisper.cpp with quantized GGML models."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
import wave
from pathlib import Path
from typing import Any, List, Optional, Protocol

import numpy as np

from nido.config import STTConfig
from nido.logging import get_logger

logger = get_logger("nido.stt.whisper")


class StreamingSTT(Protocol):
    """Abstract interface for streaming English speech recognition."""

    @property
    def is_loaded(self) -> bool:
        ...

    def start_session(self) -> None:
        """Initialize a new streaming recognition session."""
        ...

    def feed_audio(self, samples: np.ndarray, sample_rate: int = 16000) -> None:
        """Feed an incremental chunk of float32 PCM audio samples."""
        ...

    def get_partial_text(self) -> str:
        """Return the current streaming partial hypothesis text."""
        ...

    def is_endpoint(self) -> bool:
        """Return True if an utterance endpoint / trailing silence is detected."""
        ...

    def finalize(self) -> str:
        """Finalize the current stream decoding and return the completed utterance."""
        ...

    def reset_utterance(self) -> None:
        """Reset the stream state to start decoding the next utterance."""
        ...

    def stop_session(self) -> None:
        """End and cleanup the current streaming session."""
        ...

    def transcribe_waveform(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """One-shot transcribe an entire waveform using the recognizer."""
        ...


class WhisperStreamingSTT:
    """Production speech recognition engine using Whisper.cpp with quantized GGML models.

    Optimized for low-latency CPU streaming and push-to-talk on Intel CPUs (AVX2).
    """

    def __init__(self, config: Optional[STTConfig] = None) -> None:
        self.config = config or STTConfig()
        self.model_dir = Path(self.config.model_dir).expanduser().resolve()
        self.model_filename = getattr(self.config, "model_file", "ggml-base.en-q5_1.bin")
        self.model_path = self._locate_model_file()
        self.threads = getattr(self.config, "threads", 4)
        self.cli_bin = self._find_cli_binary()

        self._audio_buffer: List[np.ndarray] = []
        self._partial_text: str = ""
        self._speech_detected: bool = False
        self._silence_duration_s: float = 0.0
        self._silence_threshold: float = 0.003
        self._last_decode_time: float = 0.0

        if self.is_loaded:
            size_mb = round(self.model_path.stat().st_size / (1024 * 1024), 1)  # type: ignore[union-attr]
            logger.info(
                f"Whisper.cpp loaded with model '{self.model_path.name}' ({size_mb} MB) "  # type: ignore[union-attr]
                f"using binary '{self.cli_bin}' ({self.threads} threads)."
            )
        else:
            logger.warning(
                f"Whisper.cpp not ready. Model found: {self.model_path is not None}, "
                f"CLI binary found: {self.cli_bin is not None}. Run 'nido models setup'."
            )

    @property
    def is_loaded(self) -> bool:
        return self.model_path is not None and self.model_path.is_file() and self.cli_bin is not None

    def _locate_model_file(self) -> Optional[Path]:
        """Find the GGML model file in the configured directory."""
        if not self.model_dir.is_dir():
            return None

        exact = self.model_dir / self.model_filename
        if exact.is_file():
            return exact

        # Fallback to any ggml-*.bin or *.bin in model_dir
        ggml_bins = list(self.model_dir.glob("ggml*.bin")) + list(self.model_dir.glob("*.bin"))
        if ggml_bins:
            return ggml_bins[0]

        return None

    def _find_cli_binary(self) -> Optional[str]:
        """Search for whisper-cli executable in PATH and standard locations."""
        candidates = [
            shutil.which("whisper-cli"),
            shutil.which("whisper-cpp"),
            str(Path.home() / ".local/bin/whisper-cli"),
            "/usr/local/bin/whisper-cli",
            "/usr/bin/whisper-cli",
            str(Path.home() / ".local/share/nido/whisper.cpp/build/bin/whisper-cli"),
        ]
        return next((c for c in candidates if c and Path(c).is_file() and os.access(c, os.X_OK)), None)

    def start_session(self) -> None:
        """Reset internal buffers and prepare for a streaming session."""
        self._audio_buffer.clear()
        self._partial_text = ""
        self._speech_detected = False
        self._silence_duration_s = 0.0
        self._last_decode_time = time.monotonic()

    def feed_audio(self, samples: np.ndarray, sample_rate: int = 16000) -> None:
        """Feed incremental audio samples into the buffer and update hypotheses."""
        if len(samples) == 0:
            return

        arr = samples.astype(np.float32)
        if arr.ndim > 1:
            arr = arr.flatten()

        self._audio_buffer.append(arr)

        # 1. Voice activity / silence energy estimation
        rms = float(np.sqrt(np.mean(arr ** 2)))
        chunk_s = len(arr) / float(sample_rate)

        if rms > self._silence_threshold:
            self._speech_detected = True
            self._silence_duration_s = 0.0
        else:
            if self._speech_detected:
                self._silence_duration_s += chunk_s

        # 2. Throttled partial decoding
        now = time.monotonic()
        if (now - self._last_decode_time) >= 1.0:
            self._last_decode_time = now
            if self._speech_detected and self.is_loaded:
                self._partial_text = self._decode_current_buffer(sample_rate)

    def _decode_current_buffer(self, sample_rate: int = 16000) -> str:
        """Decode currently accumulated audio buffer."""
        if not self._audio_buffer or not self.is_loaded:
            return self._partial_text

        full_audio = np.concatenate(self._audio_buffer)
        if len(full_audio) < int(0.4 * sample_rate):
            return self._partial_text

        try:
            text = self.transcribe_waveform(full_audio, sample_rate)
            if text:
                return text
        except Exception as e:
            logger.debug(f"Partial decode error: {e}")

        return self._partial_text

    def get_partial_text(self) -> str:
        if self._partial_text:
            return self._partial_text.strip()
        if self._speech_detected:
            return "Listening..."
        return ""

    def is_endpoint(self) -> bool:
        """Detect trailing silence indicating speaker pause / command completion."""
        if not self.config.enable_endpoint_detection:
            return False

        if not self._audio_buffer:
            return False

        total_samples = sum(len(c) for c in self._audio_buffer)
        total_duration_s = total_samples / 16000.0

        # Rule 1: Speech detected followed by minimum trailing silence
        if self._speech_detected and self._silence_duration_s >= self.config.rule1_min_trailing_silence:
            return True

        # Rule 3: Maximum utterance duration cap reached
        if total_duration_s >= self.config.rule3_min_utterance_length:
            return True

        return False

    def finalize(self) -> str:
        """Decode complete utterance buffer and return finalized text."""
        if not self._audio_buffer:
            return ""

        full_audio = np.concatenate(self._audio_buffer)
        text = ""

        if self.is_loaded and len(full_audio) >= int(0.2 * 16000):
            try:
                text = self.transcribe_waveform(full_audio, 16000)
            except Exception as e:
                logger.error(f"Error finalizing Whisper transcription: {e}")
                text = self._partial_text
        else:
            text = self._partial_text

        self.reset_utterance()
        return text.strip()

    def reset_utterance(self) -> None:
        """Reset utterance buffer and tracking variables."""
        self._audio_buffer.clear()
        self._partial_text = ""
        self._speech_detected = False
        self._silence_duration_s = 0.0
        self._last_decode_time = time.monotonic()

    def stop_session(self) -> None:
        """End streaming session and clear buffers."""
        self._audio_buffer.clear()
        self._partial_text = ""
        self._speech_detected = False
        self._silence_duration_s = 0.0

    def transcribe_waveform(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """One-shot transcribe an audio buffer using Whisper.cpp."""
        if len(audio) == 0:
            return ""

        if not self.is_loaded:
            raise RuntimeError(
                f"Whisper.cpp model or CLI not available. Run 'nido models setup'."
            )

        arr = audio.astype(np.float32)
        if arr.ndim > 1:
            arr = arr.flatten()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_wav = f.name

        try:
            with wave.open(temp_wav, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                int16_samples = np.clip(arr * 32767.0, -32768, 32767).astype(np.int16)
                wf.writeframes(int16_samples.tobytes())

            cmd = [
                str(self.cli_bin),
                "-m",
                str(self.model_path),
                "-f",
                temp_wav,
                "-nt",
                "-t",
                str(self.threads),
                "--language",
                "en",
            ]
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=25,
            )
            if res.returncode == 0 and res.stdout:
                # Clean up any bracketed timestamps or markers
                clean_text = res.stdout.strip()
                return clean_text
            else:
                logger.warning(f"whisper-cli failed (code={res.returncode}): {res.stderr}")
                return ""
        except Exception as e:
            logger.error(f"Whisper transcription error: {e}")
            return ""
        finally:
            Path(temp_wav).unlink(missing_ok=True)


class MockStreamingSTT:
    """Mock streaming recognizer for fast, deterministic unit testing."""

    def __init__(self, predefined_script: Optional[List[str]] = None) -> None:
        self.script = predefined_script or ["open notes", "create a new note"]
        self.current_idx = 0
        self._session_active = False
        self._partial_text = ""
        self._fed_samples_count = 0
        self._endpoint_trigger_at = 8000
        self._is_endpoint_flag = False

    @property
    def is_loaded(self) -> bool:
        return True

    def start_session(self) -> None:
        self._session_active = True
        self._fed_samples_count = 0
        self._is_endpoint_flag = False
        if self.script and self.current_idx < len(self.script):
            self._partial_text = self.script[self.current_idx]

    def feed_audio(self, samples: np.ndarray, sample_rate: int = 16000) -> None:
        self._fed_samples_count += len(samples)
        if self._fed_samples_count >= self._endpoint_trigger_at:
            self._is_endpoint_flag = True

    def get_partial_text(self) -> str:
        return self._partial_text

    def is_endpoint(self) -> bool:
        return self._is_endpoint_flag

    def finalize(self) -> str:
        res = self._partial_text
        self.reset_utterance()
        return res

    def reset_utterance(self) -> None:
        self._fed_samples_count = 0
        self._is_endpoint_flag = False
        self.current_idx += 1
        if self.script and self.current_idx < len(self.script):
            self._partial_text = self.script[self.current_idx]
        else:
            self._partial_text = ""

    def stop_session(self) -> None:
        self._session_active = False

    def transcribe_waveform(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        if len(audio) == 0:
            return ""
        if self.script and self.current_idx < len(self.script):
            res = self.script[self.current_idx]
            self.current_idx += 1
            return res
        return "mock command"
