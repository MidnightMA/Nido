"""Event definitions and event bus for Nido pipeline orchestration."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Any, Callable, Dict, List


class PipelineStage(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    OBSERVING = "observing"
    PLANNING = "planning"
    INTERACTING = "interacting"
    WAITING = "waiting"
    EXECUTING = "executing"
    BLOCKED = "blocked"
    DONE = "done"
    ERROR = "error"
    # Realtime & Streaming STT events
    REALTIME_STARTED = "realtime_started"
    REALTIME_STOPPING = "realtime_stopping"
    REALTIME_STOPPED = "realtime_stopped"
    STT_PARTIAL = "stt_partial"
    STT_FINAL = "stt_final"
    COMMAND_QUEUED = "command_queued"
    COMMAND_STARTED = "command_started"
    COMMAND_COMPLETED = "command_completed"
    COMMAND_FAILED = "command_failed"


@dataclass
class PipelineEvent:
    stage: PipelineStage
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


EventListener = Callable[[PipelineEvent], None]


class EventBus:
    """Thread-safe event bus for publishing and subscribing to pipeline events."""

    def __init__(self) -> None:
        self._listeners: List[EventListener] = []
        self._lock = Lock()

    def subscribe(self, listener: EventListener) -> None:
        """Register a callback to receive pipeline events."""
        with self._lock:
            if listener not in self._listeners:
                self._listeners.append(listener)

    def unsubscribe(self, listener: EventListener) -> None:
        """Unregister an event callback."""
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    def emit(self, event: PipelineEvent) -> None:
        """Publish an event to all subscribers."""
        with self._lock:
            listeners = list(self._listeners)

        for listener in listeners:
            try:
                listener(event)
            except Exception:
                # Listener exceptions must not disrupt the pipeline
                pass
