"""Base hotkey protocols and abstractions."""

from __future__ import annotations

from typing import Callable, Protocol

Callback = Callable[[], None]


class HotkeyBackend(Protocol):
    """Protocol for global push-to-talk hotkey backends."""

    def start(self) -> None:
        """Start listening for hotkey events."""
        ...

    def stop(self) -> None:
        """Stop listening for hotkey events."""
        ...
