"""Central pipeline orchestration for Nido."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from nido.config import Config
from nido.events import EventBus, PipelineEvent, PipelineStage
from nido.logging import get_logger
from nido.needle.agent import NeedleCommandRouter, PlannedToolCall
from nido.stt.shenava import SpeechRecognizer
from nido.tools.registry import ToolRegistry
from nido.translation.marian_ct2 import Translator

logger = get_logger("nido.pipeline")


@dataclass
class PipelineResult:
    success: bool
    heard_persian: str = ""
    translated_english: str = ""
    planned_tools: List[PlannedToolCall] = field(default_factory=list)
    executed_results: List[Dict[str, Any]] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None


class AssistantPipeline:
    """Orchestrates audio transcription, translation, tool routing, and execution."""

    def __init__(
        self,
        config: Config,
        stt: SpeechRecognizer,
        translator: Translator,
        router: NeedleCommandRouter,
        registry: ToolRegistry,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        self.config = config
        self.stt = stt
        self.translator = translator
        self.router = router
        self.registry = registry
        self.event_bus = event_bus or EventBus()
        self.current_stage = PipelineStage.IDLE

    def _emit(self, stage: PipelineStage, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        self.current_stage = stage
        event = PipelineEvent(stage=stage, message=message, data=data or {})
        logger.info(f"Pipeline: [{stage.value.upper()}] {message}")
        self.event_bus.emit(event)

    def process_audio(self, audio: np.ndarray, sample_rate: int = 16000) -> PipelineResult:
        """Process a captured audio recording through the full voice command pipeline."""
        start_total = time.time()
        timings: Dict[str, float] = {}

        # 1. Validation
        if audio is None or len(audio) == 0:
            msg = "No audio recorded."
            self._emit(PipelineStage.ERROR, msg, {"error": msg})
            self._emit(PipelineStage.IDLE, "Ready.")
            return PipelineResult(success=False, error=msg)

        duration = len(audio) / float(sample_rate)
        if duration < 0.2:
            msg = "Recording too short (< 0.2s)."
            logger.warning(msg)
            self._emit(PipelineStage.ERROR, msg, {"error": msg})
            self._emit(PipelineStage.IDLE, "Ready.")
            return PipelineResult(success=False, error=msg)

        timings["audio_duration"] = round(duration, 3)

        # 2. STT Stage
        self._emit(PipelineStage.TRANSCRIBING, "Transcribing Persian speech...")
        t_stt_start = time.time()
        try:
            persian_text = self.stt.transcribe(audio, sample_rate=sample_rate).strip()
            timings["stt"] = round(time.time() - t_stt_start, 3)
        except Exception as e:
            timings["stt"] = round(time.time() - t_stt_start, 3)
            err_msg = f"Speech recognition failed: {e}"
            logger.error(err_msg)
            self._emit(PipelineStage.ERROR, err_msg, {"error": err_msg, "timings": timings})
            self._emit(PipelineStage.IDLE, "Ready.")
            return PipelineResult(success=False, error=err_msg, timings=timings)

        if not persian_text:
            msg = "No speech detected in audio."
            self._emit(PipelineStage.ERROR, msg, {"error": msg, "timings": timings})
            self._emit(PipelineStage.IDLE, "Ready.")
            return PipelineResult(success=False, error=msg, timings=timings)

        self._emit(
            PipelineStage.TRANSCRIBING,
            f"Heard: '{persian_text}'",
            {"persian_text": persian_text, "timings": timings},
        )

        # 3. Translation Stage
        self._emit(PipelineStage.TRANSLATING, "Translating Persian -> English...")
        t_trans_start = time.time()
        try:
            english_cmd = self.translator.translate(persian_text).strip()
            timings["translation"] = round(time.time() - t_trans_start, 3)
        except Exception as e:
            timings["translation"] = round(time.time() - t_trans_start, 3)
            err_msg = f"Translation failed: {e}"
            logger.error(err_msg)
            self._emit(
                PipelineStage.ERROR,
                err_msg,
                {"error": err_msg, "persian_text": persian_text, "timings": timings},
            )
            self._emit(PipelineStage.IDLE, "Ready.")
            return PipelineResult(
                success=False,
                heard_persian=persian_text,
                error=err_msg,
                timings=timings,
            )

        if not english_cmd:
            msg = "Translation produced empty command."
            self._emit(PipelineStage.ERROR, msg, {"error": msg, "timings": timings})
            self._emit(PipelineStage.IDLE, "Ready.")
            return PipelineResult(
                success=False,
                heard_persian=persian_text,
                error=msg,
                timings=timings,
            )

        self._emit(
            PipelineStage.TRANSLATING,
            f"Translated: '{english_cmd}'",
            {
                "persian_text": persian_text,
                "english_cmd": english_cmd,
                "timings": timings,
            },
        )

        # 4. Needle Tool Routing Stage
        self._emit(PipelineStage.THINKING, "Understanding command and selecting tools...")
        t_needle_start = time.time()
        try:
            planned_tools = self.router.plan(english_cmd)
            timings["needle"] = round(time.time() - t_needle_start, 3)
        except Exception as e:
            timings["needle"] = round(time.time() - t_needle_start, 3)
            err_msg = f"Needle tool selection failed: {e}"
            logger.error(err_msg)
            self._emit(PipelineStage.ERROR, err_msg, {"error": err_msg, "timings": timings})
            self._emit(PipelineStage.IDLE, "Ready.")
            return PipelineResult(
                success=False,
                heard_persian=persian_text,
                translated_english=english_cmd,
                error=err_msg,
                timings=timings,
            )

        if not planned_tools:
            msg = f"No suitable tool found for command: '{english_cmd}'"
            self._emit(
                PipelineStage.DONE,
                msg,
                {
                    "persian_text": persian_text,
                    "english_cmd": english_cmd,
                    "executed_results": [],
                    "timings": timings,
                },
            )
            self._emit(PipelineStage.IDLE, "Ready.")
            return PipelineResult(
                success=True,
                heard_persian=persian_text,
                translated_english=english_cmd,
                planned_tools=[],
                executed_results=[],
                timings=timings,
            )

        # 5. Tool Execution Stage
        self._emit(
            PipelineStage.EXECUTING,
            f"Executing {len(planned_tools)} action(s)...",
            {
                "persian_text": persian_text,
                "english_cmd": english_cmd,
                "tools": [t.tool_name for t in planned_tools],
            },
        )

        t_exec_start = time.time()
        executed_results: List[Dict[str, Any]] = []

        for call in planned_tools:
            res = self.registry.execute(call.tool_name, call.arguments)
            executed_results.append(res)

        timings["execution"] = round(time.time() - t_exec_start, 3)
        timings["total"] = round(time.time() - start_total, 3)

        # 6. Completion Stage
        summary_messages = [
            r.get("message", r.get("error", "Action completed")) for r in executed_results
        ]
        self._emit(
            PipelineStage.DONE,
            "; ".join(summary_messages),
            {
                "persian_text": persian_text,
                "english_cmd": english_cmd,
                "planned_tools": planned_tools,
                "executed_results": executed_results,
                "timings": timings,
            },
        )

        return PipelineResult(
            success=True,
            heard_persian=persian_text,
            translated_english=english_cmd,
            planned_tools=planned_tools,
            executed_results=executed_results,
            timings=timings,
        )
