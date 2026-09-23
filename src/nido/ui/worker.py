"""Background worker threads and signal bridges for asynchronous execution in PySide6."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

try:
    from PySide6.QtCore import QObject, QThread, Signal
except ImportError:
    # Dummy fallbacks for headless / test environments without PySide6 installed
    class QObject:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class QThread(QObject):  # type: ignore[no-redef]
        def start(self) -> None:
            pass

        def quit(self) -> None:
            pass

        def wait(self) -> None:
            pass

    def Signal(*args: Any) -> Any:  # type: ignore[no-redef]
        class DummySignal:
            def connect(self, slot: Any) -> None:
                pass

            def emit(self, *a: Any) -> None:
                pass

        return DummySignal()

from nido.events import EventBus, PipelineEvent, PipelineStage
from nido.logging import get_logger
from nido.pipeline import AssistantPipeline, PipelineResult

logger = get_logger("nido.ui.worker")


class EventBridge(QObject):
    """Bridges multi-threaded pipeline and realtime events safely to Qt GUI signals."""

    event_received = Signal(str, str, dict)

    def __init__(self, event_bus: EventBus, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.event_bus = event_bus
        self.event_bus.subscribe(self._on_event)

    def _on_event(self, event: PipelineEvent) -> None:
        self.event_received.emit(event.stage.value, event.message, event.data)

    def cleanup(self) -> None:
        self.event_bus.unsubscribe(self._on_event)


class PipelineWorker(QThread):
    """Executes pipeline audio processing off the Qt GUI thread."""

    stage_changed = Signal(str, str, dict)
    pipeline_finished = Signal(object)

    def __init__(self, pipeline: AssistantPipeline, audio: np.ndarray, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.pipeline = pipeline
        self.audio = audio

    def run(self) -> None:
        logger.debug("PipelineWorker started in background thread.")

        def _on_event(event: PipelineEvent) -> None:
            self.stage_changed.emit(event.stage.value, event.message, event.data)

        self.pipeline.event_bus.subscribe(_on_event)
        try:
            result = self.pipeline.process_audio(self.audio)
            self.pipeline_finished.emit(result)
        except Exception as e:
            logger.exception(f"Worker exception in pipeline: {e}")
            err_result = PipelineResult(success=False, error=str(e))
            self.pipeline_finished.emit(err_result)
        finally:
            self.pipeline.event_bus.unsubscribe(_on_event)
            logger.debug("PipelineWorker finished.")
