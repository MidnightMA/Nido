"""Background worker thread for asynchronous pipeline execution in PySide6."""

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

from nido.events import PipelineEvent, PipelineStage
from nido.logging import get_logger
from nido.pipeline import AssistantPipeline, PipelineResult

logger = get_logger("nido.ui.worker")


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

        # Hook into pipeline event bus to bridge to Qt signals
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
