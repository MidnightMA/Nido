"""English Streaming Speech-to-Text using sherpa-onnx Zipformer transducer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Protocol

import numpy as np

try:
    import sherpa_onnx
except ImportError:
    sherpa_onnx = None  # type: ignore[assignment]

from nido.config import STTConfig
from nido.logging import get_logger

logger = get_logger("nido.stt")


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


class ZipformerStreamingSTT:
    """Production streaming recognizer for English Zipformer 20M model."""

    def __init__(self, config: Optional[STTConfig] = None) -> None:
        self.config = config or STTConfig()
        self.model_dir = Path(self.config.model_dir).expanduser().resolve()
        self._recognizer: Optional[Any] = None
        self._stream: Optional[Any] = None
        self._load_recognizer()

    @property
    def is_loaded(self) -> bool:
        return self._recognizer is not None

    def _load_recognizer(self) -> None:
        if sherpa_onnx is None:
            logger.warning("sherpa-onnx package not installed. Speech recognition inactive.")
            return

        if not self.model_dir.is_dir():
            logger.warning(f"STT model directory does not exist: {self.model_dir}")
            return

        # Find transducer model assets: encoder, decoder, joiner, tokens
        # Prefer int8 versions if available
        int8_encoders = list(self.model_dir.glob("*encoder*.int8.onnx"))
        regular_encoders = list(self.model_dir.glob("*encoder*.onnx"))
        encoder_files = int8_encoders or regular_encoders

        decoder_files = list(self.model_dir.glob("*decoder*.onnx"))

        int8_joiners = list(self.model_dir.glob("*joiner*.int8.onnx"))
        regular_joiners = list(self.model_dir.glob("*joiner*.onnx"))
        joiner_files = int8_joiners or regular_joiners

        token_files = list(self.model_dir.glob("*tokens*.txt"))

        if not (encoder_files and decoder_files and joiner_files and token_files):
            logger.warning(
                f"Missing Zipformer transducer files in {self.model_dir}. "
                f"Found encoder: {len(encoder_files)}, decoder: {len(decoder_files)}, "
                f"joiner: {len(joiner_files)}, tokens: {len(token_files)}. "
                "Run 'nido models setup' to download model assets."
            )
            return

        encoder_path = str(encoder_files[0])
        decoder_path = str(decoder_files[0])
        joiner_path = str(joiner_files[0])
        tokens_path = str(token_files[0])

        provider = self.config.provider.lower()
        if provider not in ("cpu", "cuda", "coreml"):
            provider = "cpu"

        logger.info(
            f"Loading Zipformer streaming recognizer from {self.model_dir} "
            f"(encoder: {encoder_files[0].name}, joiner: {joiner_files[0].name}, threads: {self.config.threads})..."
        )

        try:
            self._recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                encoder=encoder_path,
                decoder=decoder_path,
                joiner=joiner_path,
                tokens=tokens_path,
                num_threads=self.config.threads,
                sample_rate=16000,
                feature_dim=80,
                enable_endpoint_detection=self.config.enable_endpoint_detection,
                rule1_min_trailing_silence=self.config.rule1_min_trailing_silence,
                rule2_min_trailing_silence=self.config.rule2_min_trailing_silence,
                rule3_min_utterance_length=self.config.rule3_min_utterance_length,
                decoding_method=self.config.decoding_method,
                provider=provider,
            )
            logger.info("Zipformer streaming recognizer loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Zipformer OnlineRecognizer: {e}")
            self._recognizer = None

    def start_session(self) -> None:
        if self._recognizer is None:
            raise RuntimeError(f"STT model not loaded from {self.model_dir}. Run 'nido models setup'.")
        self._stream = self._recognizer.create_stream()

    def feed_audio(self, samples: np.ndarray, sample_rate: int = 16000) -> None:
        if self._recognizer is None:
            return
        if self._stream is None:
            self.start_session()

        if len(samples) == 0:
            return

        arr = samples.astype(np.float32)
        if arr.ndim > 1:
            arr = arr.flatten()

        self._stream.accept_waveform(sample_rate, arr)
        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)

    def _extract_text(self, stream: Any) -> str:
        if self._recognizer is None or stream is None:
            return ""
        try:
            res = self._recognizer.get_result(stream)
            if hasattr(res, "text"):
                return res.text.strip()
            return str(res).strip()
        except Exception as e:
            logger.debug(f"get_result error: {e}")
            return ""

    def get_partial_text(self) -> str:
        if self._stream is None:
            return ""
        return self._extract_text(self._stream)

    def is_endpoint(self) -> bool:
        if self._recognizer is None or self._stream is None:
            return False
        return bool(self._recognizer.is_endpoint(self._stream))

    def finalize(self) -> str:
        if self._recognizer is None or self._stream is None:
            return ""

        try:
            self._stream.input_finished()
            while self._recognizer.is_ready(self._stream):
                self._recognizer.decode_stream(self._stream)
            text = self._extract_text(self._stream)
            return text
        finally:
            self.reset_utterance()

    def reset_utterance(self) -> None:
        if self._recognizer is not None and self._stream is not None:
            try:
                self._recognizer.reset(self._stream)
            except Exception:
                # If reset is unsupported or fails, create fresh stream
                self._stream = self._recognizer.create_stream()

    def stop_session(self) -> None:
        self._stream = None

    def transcribe_waveform(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """One-shot transcribe an audio buffer using the long-lived OnlineRecognizer."""
        if len(audio) == 0:
            return ""
        if self._recognizer is None:
            raise RuntimeError(f"STT model not loaded from {self.model_dir}. Run 'nido models setup'.")

        stream = self._recognizer.create_stream()
        arr = audio.astype(np.float32)
        if arr.ndim > 1:
            arr = arr.flatten()

        stream.accept_waveform(sample_rate, arr)
        stream.input_finished()
        while self._recognizer.is_ready(stream):
            self._recognizer.decode_stream(stream)
        return self._extract_text(stream)


class MockStreamingSTT:
    """Mock streaming recognizer for fast, deterministic unit testing."""

    def __init__(self, predefined_script: Optional[List[str]] = None) -> None:
        self.script = predefined_script or ["open notes", "create a new note"]
        self.current_idx = 0
        self._session_active = False
        self._partial_text = ""
        self._fed_samples_count = 0
        self._endpoint_trigger_at = 8000  # samples before endpoint
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
        else:
            self._partial_text = ""

    def feed_audio(self, samples: np.ndarray, sample_rate: int = 16000) -> None:
        self._fed_samples_count += len(samples)
        if self._fed_samples_count >= self._endpoint_trigger_at:
            self._is_endpoint_flag = True

    def get_partial_text(self) -> str:
        return self._partial_text

    def is_endpoint(self) -> bool:
        return self._is_endpoint_flag

    def finalize(self) -> str:
        text = self._partial_text
        self.reset_utterance()
        return text

    def reset_utterance(self) -> None:
        self._fed_samples_count = 0
        self._is_endpoint_flag = False
        if self.current_idx < len(self.script) - 1:
            self.current_idx += 1
            self._partial_text = self.script[self.current_idx]
        else:
            self._partial_text = ""

    def stop_session(self) -> None:
        self._session_active = False

    def transcribe_waveform(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        if len(audio) == 0:
            return ""
        if self.script:
            return self.script[0]
        return "open notes"
