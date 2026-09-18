"""Persian to English translation package."""

from nido.translation.marian_ct2 import (
    ENTITY_MAP,
    MarianCT2Translator,
    MockTranslator,
    Translator,
    preprocess_persian_text,
)

__all__ = [
    "Translator",
    "MarianCT2Translator",
    "MockTranslator",
    "ENTITY_MAP",
    "preprocess_persian_text",
]
