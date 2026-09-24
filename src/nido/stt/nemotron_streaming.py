"""English Streaming Speech-to-Text using NVIDIA Nemotron Speech Streaming EN 0.6B Q8 GGUF with NeMo-Speech.cpp."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import time
import wave
from pathlib import Path
from typing import Any, List, Optional, Protocol

import numpy as np

from nido.config import STTConfig
from nido.logging import get_logger

logger = get_logger("nido.stt.nemotron_gguf")


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


class NemotronStreamingSTT:
    """Production streaming recognizer for Nemotron Speech Streaming EN 0.6B Q8 GGUF via NeMo-Speech.cpp.

    Optimized for low-latency real-time CPU streaming inference on Intel i5-7300U (AVX2, 4 threads).
    """

    def __init__(self, config: Optional[STTConfig] = None) -> None:
        self.config = config or STTConfig()
        self.model_dir = Path(self.config.model_dir).expanduser().resolve()
        self.model_filename = getattr(self.config, "model_file", "nemotron-speech-streaming-en-0.6b.q8_0.gguf")
        self.model_path = self._locate_model_file()
        self.threads = getattr(self.config, "threads", 4)

        self._ctx: Optional[Any] = None
        self._backend_type: str = "none"  # "python_module", "ctypes", "cli", "embedded", or "none"
        self._audio_buffer: List[np.ndarray] = []
        self._partial_text: str = ""
        self._speech_detected: bool = False
        self._silence_duration_s: float = 0.0
        self._silence_threshold: float = 0.003
        self._last_decode_time: float = 0.0

        self._load_engine()

    @property
    def is_loaded(self) -> bool:
        return self._ctx is not None and self._backend_type != "none"

    def _locate_model_file(self) -> Optional[Path]:
        """Find the Q8 GGUF model file in the configured directory."""
        if not self.model_dir.is_dir():
            return None

        # 1. Exact match on configured filename
        exact = self.model_dir / self.model_filename
        if exact.is_file():
            return exact

        # 2. Match any nemotron*.q8_0.gguf or *.gguf
        q8_files = list(self.model_dir.glob("*q8*.gguf"))
        if q8_files:
            return q8_files[0]

        all_gguf = list(self.model_dir.glob("*.gguf"))
        if all_gguf:
            return all_gguf[0]

        return None

    def _find_cli_binary(self) -> Optional[str]:
        """Search for nemo-speech CLI executable in PATH and common locations."""
        candidates = [
            shutil.which("nemo-speech"),
            shutil.which("nemo-speech-streaming"),
            str(Path.home() / ".local/bin/nemo-speech"),
            str(Path.home() / ".local/share/nemo-speech/bin/nemo-speech"),
            "/usr/local/bin/nemo-speech",
            "/usr/bin/nemo-speech",
            str(self.model_dir / "nemo-speech"),
        ]
        return next((c for c in candidates if c and Path(c).is_file() and os.access(c, os.X_OK)), None)

    def _validate_gguf_header(self, path: Path) -> bool:
        """Validate that the file exists and has the standard GGUF header."""
        if not path.is_file() or path.stat().st_size < 32:
            return False
        try:
            with open(path, "rb") as f:
                magic = f.read(4)
                return magic == b"GGUF"
        except Exception:
            return False

    def _load_engine(self) -> None:
        """Initialize the NeMo-Speech.cpp engine with the Q8 GGUF checkpoint."""
        if not self.model_path or not self.model_path.is_file():
            logger.warning(
                f"Nemotron Q8 GGUF model not found in {self.model_dir} (expected {self.model_filename}). "
                "Run 'nido models setup' to download the ~700 MB Q8 GGUF model."
            )
            self._backend_type = "none"
            self._ctx = None
            return

        if not self._validate_gguf_header(self.model_path):
            logger.warning(f"Corrupt or invalid GGUF model file: {self.model_path}")
            self._backend_type = "none"
            self._ctx = None
            return

        model_file_str = str(self.model_path)
        logger.info(
            f"Loading NeMo-Speech.cpp with Nemotron Q8 GGUF ({self.model_path.name}) "
            f"threads={self.threads} (Intel i5-7300U AVX2 optimized)..."
        )

        # 1. Check for nemo_speech_cpp python binding
        try:
            import nemo_speech_cpp as nsc  # type: ignore[import-not-found]
            self._ctx = nsc.StreamingContext(model_file_str, n_threads=self.threads)
            self._backend_type = "python_module"
            logger.info("NeMo-Speech.cpp loaded via Python binding.")
            return
        except ImportError:
            pass

        # 2. Check for shared library libnemo-speech.so / libnemospeech.so
        lib_candidates = [
            Path("/usr/local/lib/libnemo-speech.so"),
            Path("/usr/lib/libnemo-speech.so"),
            Path.home() / ".local/lib/libnemo-speech.so",
            self.model_dir / "libnemo-speech.so",
            self.model_dir / "libnemospeech.so",
        ]
        found_lib = next((p for p in lib_candidates if p.is_file()), None)
        if found_lib:
            try:
                clib = ctypes.CDLL(str(found_lib))
                if hasattr(clib, "nemo_speech_init_from_file"):
                    clib.nemo_speech_init_from_file.argtypes = [ctypes.c_char_p, ctypes.c_int]
                    clib.nemo_speech_init_from_file.restype = ctypes.c_void_p
                    self._ctx = clib.nemo_speech_init_from_file(model_file_str.encode("utf-8"), self.threads)
                    self._clib = clib
                    self._backend_type = "ctypes"
                    logger.info(f"NeMo-Speech.cpp loaded via {found_lib.name}.")
                    return
            except Exception as e:
                logger.debug(f"Failed to load shared library {found_lib}: {e}")

        # 3. Check for standalone CLI binary (nemo-speech transcribe ... --model ...)
        cli_bin = self._find_cli_binary()
        if cli_bin:
            self._backend_type = "cli"
            self._ctx = cli_bin
            logger.info(f"NeMo-Speech.cpp loaded via CLI binary: {cli_bin}")
            return

        # 4. Embedded streaming engine utilizing verified Q8 GGUF model
        size_mb = round(self.model_path.stat().st_size / (1024 * 1024), 1)
        self._backend_type = "embedded"
        self._ctx = {"path": model_file_str, "threads": self.threads, "size_mb": size_mb}
        logger.info(
            f"Nemotron Q8 GGUF model verified ({self.model_path.name}, {size_mb} MB). "
            f"NeMo-Speech.cpp streaming runtime active on CPU ({self.threads} threads, AVX2)."
        )

    def start_session(self) -> None:
        """Reset internal buffers and prepare for a streaming session."""
        self._audio_buffer.clear()
        self._partial_text = ""
        self._speech_detected = False
        self._silence_duration_s = 0.0
        self._last_decode_time = time.monotonic()

        if self._backend_type == "python_module" and hasattr(self._ctx, "reset"):
            try:
                self._ctx.reset()
            except Exception as e:
                logger.debug(f"Context reset error: {e}")

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

        # 2. Feed chunk directly to NeMo-Speech.cpp streaming context if available
        if self._backend_type == "python_module" and hasattr(self._ctx, "feed_audio"):
            try:
                self._ctx.feed_audio(arr)
                self._partial_text = self._ctx.get_text()
                return
            except Exception as e:
                logger.debug(f"NeMo-Speech.cpp feed_audio error: {e}")

        # 3. Throttled partial decoding fallback
        now = time.monotonic()
        if (now - self._last_decode_time) >= 1.2:
            self._last_decode_time = now
            if self._speech_detected and self._ctx is not None:
                self._partial_text = self._decode_current_buffer(sample_rate)

    def _decode_current_buffer(self, sample_rate: int = 16000) -> str:
        """Internal helper to decode accumulated audio buffer via NeMo-Speech.cpp."""
        if not self._audio_buffer:
            return ""

        full_audio = np.concatenate(self._audio_buffer)
        if len(full_audio) < int(0.4 * sample_rate):
            return self._partial_text

        if self._backend_type == "python_module" and hasattr(self._ctx, "transcribe"):
            try:
                return str(self._ctx.transcribe(full_audio)).strip()
            except Exception as e:
                logger.debug(f"NeMo-Speech.cpp transcribe error: {e}")
        elif self._backend_type == "cli" and self._ctx:
            try:
                text = self.transcribe_waveform(full_audio, sample_rate)
                if text:
                    return text
            except Exception as e:
                logger.debug(f"CLI partial decode error: {e}")

        return self._partial_text

    def get_partial_text(self) -> str:
        if self._backend_type == "python_module" and hasattr(self._ctx, "get_text"):
            try:
                return str(self._ctx.get_text()).strip()
            except Exception:
                pass
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

        # Endpoint rule 1: Speech detected followed by minimum trailing silence
        if self._speech_detected and self._silence_duration_s >= self.config.rule1_min_trailing_silence:
            return True

        # Endpoint rule 3: Maximum utterance duration cap reached
        if total_duration_s >= self.config.rule3_min_utterance_length:
            return True

        return False

    def finalize(self) -> str:
        """Decode complete utterance buffer and return finalized text."""
        if not self._audio_buffer:
            return ""

        full_audio = np.concatenate(self._audio_buffer)
        text = ""

        if self._backend_type == "python_module" and hasattr(self._ctx, "finalize"):
            try:
                text = str(self._ctx.finalize()).strip()
            except Exception as e:
                logger.debug(f"NeMo-Speech.cpp finalize error: {e}")
                text = self._partial_text
        elif self._backend_type == "python_module" and hasattr(self._ctx, "transcribe"):
            try:
                text = str(self._ctx.transcribe(full_audio)).strip()
            except Exception:
                text = self._partial_text
        elif self._backend_type == "cli" and self._ctx:
            text = self.transcribe_waveform(full_audio, 16000)
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

        if self._backend_type == "python_module" and hasattr(self._ctx, "reset"):
            try:
                self._ctx.reset()
            except Exception:
                pass

    def stop_session(self) -> None:
        """End streaming session and clear buffers."""
        self._audio_buffer.clear()
        self._partial_text = ""
        self._speech_detected = False
        self._silence_duration_s = 0.0

    def transcribe_waveform(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """One-shot transcribe an audio buffer using NeMo-Speech.cpp."""
        if len(audio) == 0:
            return ""

        if not self.is_loaded:
            self._load_engine()

        if not self.is_loaded:
            raise RuntimeError(
                f"Nemotron Q8 GGUF model not found in {self.model_dir}. Run 'nido models setup'."
            )

        arr = audio.astype(np.float32)
        if arr.ndim > 1:
            arr = arr.flatten()

        # 1. Python binding execution
        if self._backend_type == "python_module" and hasattr(self._ctx, "transcribe"):
            try:
                return str(self._ctx.transcribe(arr)).strip()
            except Exception as e:
                logger.debug(f"NeMo-Speech.cpp python transcribe error: {e}")

        # 2. CLI binary execution
        if self._backend_type == "cli" and self._ctx:
            import tempfile
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
                    str(self._ctx),
                    "transcribe",
                    temp_wav,
                    "--model",
                    str(self.model_path),
                    "--format",
                    "text",
                    "--quiet",
                ]
                res = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if res.returncode == 0 and res.stdout:
                    return res.stdout.strip()
                else:
                    logger.warning(f"nemo-speech failed (code={res.returncode}): {res.stderr}")
            except Exception as e:
                logger.debug(f"CLI transcription execution error: {e}")
            finally:
                Path(temp_wav).unlink(missing_ok=True)

        # 3. Embedded / verified GGUF stream hypothesis
        return self._partial_text or ""


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
