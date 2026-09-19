"""Persian -> English translation using CTranslate2 and SentencePiece."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Protocol

try:
    import ctranslate2
except ImportError:
    ctranslate2 = None  # type: ignore[assignment]

try:
    import sentencepiece as spm
except ImportError:
    spm = None  # type: ignore[assignment]

from nido.logging import get_logger

logger = get_logger("nido.translation")

# Mapping for preserving key technical entities / aliases in Persian spoken commands
ENTITY_MAP: Dict[str, str] = {
    "کروم": "Chrome",
    "گوگل کروم": "Google Chrome",
    "فایرفاکس": "Firefox",
    "یوتیوب": "YouTube",
    "دیسکورد": "Discord",
    "گیت هاب": "GitHub",
    "گیتهاب": "GitHub",
    "وی اس کد": "VS Code",
    "پایتون": "Python",
    "داکر": "Docker",
    "سوبابیس": "Supabase",
    "ترمینال": "Terminal",
    "کنسول": "Konsole",
    "تلگرام": "Telegram",
}


def preprocess_persian_text(text: str) -> str:
    """Clean and normalize Persian text before translation."""
    if not text:
        return ""

    # Normalize Persian / Arabic character variants
    cleaned = text.replace("ي", "ی").replace("ك", "ک").strip()

    # Normalize repeated whitespace
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


class Translator(Protocol):
    """Protocol for Persian -> English translation."""

    def translate(self, text: str) -> str:
        """Translate Persian text to concise English command."""
        ...


class MarianCT2Translator(Translator):
    """Offline Persian -> English translator using HPLT Marian model in CTranslate2 INT8."""

    def __init__(
        self,
        model_dir: str | Path,
        compute_type: str = "int8",
        beam_size: int = 1,
        max_tokens: int = 64,
        threads: int = 4,
    ) -> None:
        self.model_dir = Path(model_dir).expanduser()
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.max_tokens = max_tokens
        self.threads = threads

        self._translator: Optional[ctranslate2.Translator] = None
        self._sp_source: Optional[spm.SentencePieceProcessor] = None
        self._sp_target: Optional[spm.SentencePieceProcessor] = None

        self._load_model()

    def _load_model(self) -> None:
        if ctranslate2 is None or spm is None:
            logger.warning("ctranslate2 or sentencepiece is not installed. Translation disabled.")
            return

        if not self.model_dir.is_dir():
            logger.warning(f"Translation model directory not found: {self.model_dir}")
            return

        # Check for model.bin
        model_bin = self.model_dir / "model.bin"
        if not model_bin.is_file():
            logger.warning(f"CTranslate2 model.bin not found in {self.model_dir}")
            return

        # Find sentencepiece models (source.spm / target.spm or shared spiece.model)
        spm_source_candidates = [
            self.model_dir / "source.spm",
            self.model_dir / "spm.src",
            self.model_dir / "spiece.model",
            self.model_dir / "source.model",
            self.model_dir / "model.fa-en.spm",
        ]
        spm_target_candidates = [
            self.model_dir / "target.spm",
            self.model_dir / "spm.trg",
            self.model_dir / "spiece.model",
            self.model_dir / "target.model",
            self.model_dir / "model.fa-en.spm",
        ]

        spm_source_file = next((p for p in spm_source_candidates if p.is_file()), None)
        spm_target_file = next((p for p in spm_target_candidates if p.is_file()), None)

        if not spm_source_file:
            logger.warning(f"Source SentencePiece model missing in {self.model_dir}")
            return

        try:
            logger.info(f"Loading SentencePiece source model: {spm_source_file}")
            self._sp_source = spm.SentencePieceProcessor()
            self._sp_source.load(str(spm_source_file))

            if spm_target_file and spm_target_file != spm_source_file:
                logger.info(f"Loading SentencePiece target model: {spm_target_file}")
                self._sp_target = spm.SentencePieceProcessor()
                self._sp_target.load(str(spm_target_file))
            else:
                self._sp_target = self._sp_source

            logger.info(
                f"Loading CTranslate2 translator from {self.model_dir} (compute_type={self.compute_type})"
            )
            self._translator = ctranslate2.Translator(
                str(self.model_dir),
                device="cpu",
                compute_type=self.compute_type,
                intra_threads=self.threads,
                inter_threads=1,
            )
            logger.info("CTranslate2 translation model successfully loaded.")
        except Exception as e:
            logger.error(f"Failed to load CTranslate2 translation model: {e}")
            self._translator = None

    @property
    def is_loaded(self) -> bool:
        return self._translator is not None and self._sp_source is not None

    def translate(self, text: str) -> str:
        """Translate Persian text into concise English command."""
        cleaned = preprocess_persian_text(text)
        if not cleaned:
            return ""

        if not self.is_loaded:
            raise RuntimeError(
                f"Translation model not loaded. Run 'nido models setup' to prepare {self.model_dir}."
            )

        assert self._sp_source is not None
        assert self._sp_target is not None
        assert self._translator is not None

        try:
            # Tokenize using source SentencePiece model
            tokens: List[str] = self._sp_source.encode(cleaned, out_type=str)

            # Translate using CTranslate2 with greedy search (beam_size=1) for speed
            results = self._translator.translate_batch(
                [tokens],
                beam_size=self.beam_size,
                max_decoding_length=self.max_tokens,
                sampling_topk=1,
            )

            out_tokens: List[str] = results[0].hypotheses[0]
            english_text = self._sp_target.decode(out_tokens).strip()

            # Clean output
            english_text = re.sub(r"\s+", " ", english_text)
            logger.debug(f"Translated: '{text}' -> '{english_text}'")
            return english_text
        except Exception as e:
            logger.error(f"Translation inference failed: {e}")
            raise RuntimeError(f"Translation failed: {e}") from e


class MockTranslator(Translator):
    """Mock translator for testing and development with predictable command mapping."""

    MOCK_PAIRS: Dict[str, str] = {
        "کروم را باز کن": "Open Chrome",
        "کروم رو باز کن": "Open Chrome",
        "مرورگر را باز کن": "Open browser",
        "مرورگر رو باز کن": "Open browser",
        "ترمینال را باز کن": "Open terminal",
        "یوتیوب برو": "Open YouTube",
        "یوتیوب را باز کن": "Open YouTube",
        "صدا را کم کن": "Decrease volume",
        "صدا رو کم کن": "Decrease volume",
        "صدا را زیاد کن": "Increase volume",
        "صدا رو زیاد کن": "Increase volume",
        "صدا رو قطع کن": "Mute volume",
        "صدا را قطع کن": "Mute volume",
        "صدا رو بذار روی سی درصد": "Set volume to 30 percent",
        "صدا رو بذار روی 30 درصد": "Set volume to 30 percent",
        "عکس از صفحه بگیر": "Take screenshot",
        "صفحه را قفل کن": "Lock screen",
        "صفحه رو قفل کن": "Lock screen",
    }

    def __init__(self, fallback_prefix: str = "Execute: ") -> None:
        self.fallback_prefix = fallback_prefix

    def translate(self, text: str) -> str:
        cleaned = preprocess_persian_text(text)
        if not cleaned:
            return ""

        for k, v in self.MOCK_PAIRS.items():
            if k in cleaned:
                return v

        # Fallback simple dictionary mapping
        result = cleaned
        for fa, en in ENTITY_MAP.items():
            result = result.replace(fa, en)

        return f"{self.fallback_prefix}{result}"
