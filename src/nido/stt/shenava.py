"""Persian Speech-to-Text using Shenava Koochik and sherpa-onnx."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Optional, Protocol

import numpy as np

try:
    import sherpa_onnx
except ImportError:
    sherpa_onnx = None  # type: ignore[assignment]

from nido.logging import get_logger

logger = get_logger("nido.stt")

# Persian number words to digits mapping
PERSIAN_ONES: Dict[str, int] = {
    "صفر": 0,
    "یک": 1,
    "یه": 1,
    "دو": 2,
    "سه": 3,
    "چهار": 4,
    "پنج": 5,
    "شش": 6,
    "شیش": 6,
    "هفت": 7,
    "هشت": 8,
    "نه": 9,
}

PERSIAN_TEENS: Dict[str, int] = {
    "ده": 10,
    "یازده": 11,
    "دوازده": 12,
    "سیزده": 13,
    "چهارده": 14,
    "پانزده": 15,
    "پونزده": 15,
    "شانزده": 16,
    "شونزده": 16,
    "هفده": 17,
    "هجده": 17,
    "نوزده": 19,
}

PERSIAN_TENS: Dict[str, int] = {
    "بیست": 20,
    "سی": 30,
    "چهل": 40,
    "پنجاه": 50,
    "شصت": 60,
    "هفتاد": 70,
    "هشتاد": 80,
    "نود": 90,
}

PERSIAN_HUNDREDS: Dict[str, int] = {
    "صد": 100,
    "یکصد": 100,
    "دویست": 200,
    "سیصد": 300,
    "چهارصد": 400,
    "پانصد": 500,
    "پونصد": 500,
    "ششصد": 600,
    "شیشصد": 600,
    "هفتصد": 700,
    "هشتصد": 800,
    "نهصد": 900,
}

PERSIAN_ALL_NUMS: Dict[str, int] = {
    **PERSIAN_ONES,
    **PERSIAN_TEENS,
    **PERSIAN_TENS,
    **PERSIAN_HUNDREDS,
}


def normalize_persian_numbers(text: str) -> str:
    """Convert spoken Persian numbers into digits (Persian Inverse Text Normalization).

    Handles single words (e.g. "سی" -> "30", "صد" -> "100") and compound numbers
    connected by 'و' (e.g. "سی و پنج" -> "35", "صد و بیست" -> "120").
    """
    if not text:
        return ""

    tokens = text.split()
    output_tokens: list[str] = []
    i = 0

    while i < len(tokens):
        token = tokens[i].strip()

        # Check for compound numbers: e.g. "سی و پنج" or "صد و بیست"
        # Look ahead for pattern: [NUM] "و" [NUM] ...
        if token in PERSIAN_ALL_NUMS:
            total = PERSIAN_ALL_NUMS[token]
            curr_idx = i
            while curr_idx + 2 < len(tokens) and tokens[curr_idx + 1] == "و" and tokens[curr_idx + 2] in PERSIAN_ALL_NUMS:
                total += PERSIAN_ALL_NUMS[tokens[curr_idx + 2]]
                curr_idx += 2

            if curr_idx > i:
                output_tokens.append(str(total))
                i = curr_idx + 1
                continue
            else:
                output_tokens.append(str(total))
                i += 1
                continue

        # Convert Persian digits (۱۲۳) to ASCII digits (123)
        persian_to_ascii = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
        clean_token = token.translate(persian_to_ascii)
        output_tokens.append(clean_token)
        i += 1

    return " ".join(output_tokens)


class SpeechRecognizer(Protocol):
    """Protocol for Persian speech recognizers."""

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe audio samples into Persian text."""
        ...


class ShenavaSherpaRecognizer(SpeechRecognizer):
    """Offline Persian Speech Recognizer using Shenava Koochik via sherpa-onnx."""

    def __init__(self, model_dir: str | Path, threads: int = 4) -> None:
        self.model_dir = Path(model_dir).expanduser()
        self.threads = threads
        self._recognizer: Optional[object] = None
        self._load_model()

    def _load_model(self) -> None:
        if sherpa_onnx is None:
            logger.warning("sherpa_onnx is not installed. STT will not function.")
            return

        if not self.model_dir.is_dir():
            logger.warning(f"Shenava model directory does not exist: {self.model_dir}")
            return

        # Find model files
        onnx_candidates = list(self.model_dir.glob("*.onnx"))
        tokens_candidates = list(self.model_dir.glob("*tokens*.txt"))

        if not onnx_candidates or not tokens_candidates:
            logger.warning(
                f"Missing .onnx or tokens.txt in {self.model_dir}. Found onnx: {onnx_candidates}, tokens: {tokens_candidates}"
            )
            return

        model_path = str(onnx_candidates[0])
        tokens_path = str(tokens_candidates[0])

        logger.info(f"Loading Shenava Koochik sherpa-onnx model from {model_path}")
        try:
            # Shenava Koochik is an offline CTC / NeMo model
            # per-feature normalization is NOT enabled for Shenava export as required
            self._recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
                model=model_path,
                tokens=tokens_path,
                num_threads=self.threads,
                sample_rate=16000,
                feature_dim=80,
            )
            logger.info("Shenava Koochik model successfully loaded.")
        except Exception as e:
            logger.error(f"Failed to initialize sherpa_onnx recognizer: {e}")
            self._recognizer = None

    @property
    def is_loaded(self) -> bool:
        return self._recognizer is not None

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe normalized float32 16kHz mono audio into Persian text."""
        if len(audio) == 0:
            return ""

        if self._recognizer is None:
            raise RuntimeError(
                f"Shenava model not loaded. Please run 'nido models setup' to install model into {self.model_dir}."
            )

        # Ensure float32 1D array
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if audio.ndim > 1:
            audio = audio.flatten()

        try:
            stream = self._recognizer.create_stream()
            stream.accept_waveform(sample_rate, audio)
            self._recognizer.decode_stream(stream)
            raw_text = stream.result.text.strip()
        except Exception as e:
            logger.error(f"STT decoding error: {e}")
            raise RuntimeError(f"Speech recognition decoding failed: {e}") from e

        # Apply Persian ITN
        normalized_text = normalize_persian_numbers(raw_text)
        logger.debug(f"Transcribed: '{raw_text}' -> ITN normalized: '{normalized_text}'")
        return normalized_text


class MockSpeechRecognizer(SpeechRecognizer):
    """Mock speech recognizer for unit tests and offline testing."""

    def __init__(self, predefined_text: str = "کروم را باز کن") -> None:
        self.predefined_text = predefined_text

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        if len(audio) == 0:
            return ""
        return normalize_persian_numbers(self.predefined_text)
