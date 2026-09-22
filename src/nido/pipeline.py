"""Central pipeline orchestration for Nido."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from nido.accessibility.atspi import AtspiBackend
from nido.config import Config
from nido.desktop.agent import DesktopAgent
from nido.desktop.input import detect_input_backend
from nido.events import EventBus, PipelineEvent, PipelineStage
from nido.logging import get_logger
from nido.needle.agent import NeedleCommandRouter, PlannedToolCall
from nido.stt.shenava import SpeechRecognizer
from nido.tools.registry import ToolRegistry
from nido.translation.marian_ct2 import Translator

logger = get_logger("nido.pipeline")


def is_desktop_command(english_cmd: str, persian_text: str = "") -> bool:
    """Determine if a command requires interactive desktop accessibility interaction."""
    cmd_lower = english_cmd.lower()

    # Fast check: Pure static commands
    pure_static_keywords = [
        "set volume", "increase volume", "decrease volume", "volume up", "volume down",
        "mute", "unmute", "play", "pause", "next track", "previous track",
        "lock screen", "lock desktop", "show system info", "take screenshot",
        "open url", "search web", "search google"
    ]
    for static_kw in pure_static_keywords:
        if static_kw in cmd_lower and not any(w in cmd_lower for w in ["write", "type", "click", "select", "and"]):
            return False

    # Interactive GUI keywords
    gui_indicators = [
        "write", "type", "click", "press", "select", "enter",
        "calculate", "new tab", "navigate", "dialog", "button",
        "downloads", "display settings"
    ]
    if any(ind in cmd_lower for ind in gui_indicators):
        return True

    # Persian indicators
    if persian_text:
        fa_indicators = ["بنویس", "تایپ", "کلیک", "فشار", "انتخاب", "حساب کن"]
        if any(ind in persian_text for ind in fa_indicators):
            return True

    return False


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
        desktop_agent: Optional[DesktopAgent] = None,
    ) -> None:
        self.config = config
        self.stt = stt
        self.translator = translator
        self.router = router
        self.registry = registry
        self.event_bus = event_bus or EventBus()
        self.current_stage = PipelineStage.IDLE

        self.desktop_agent = desktop_agent
        if self.desktop_agent is None and self.config.desktop.enabled:
            backend = AtspiBackend()
            input_b = detect_input_backend(self.config.desktop.input_backend)
            self.desktop_agent = DesktopAgent(
                config=self.config.desktop,
                router=self.router,
                accessibility=backend,
                input_backend=input_b,
                registry=self.registry,
                event_bus=self.event_bus,
            )

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

        # 4. Desktop Agent Mode for Interactive GUI Tasks
        if self.config.desktop.enabled and self.desktop_agent and is_desktop_command(english_cmd, persian_text):
            t_desk_start = time.time()
            desktop_res = self.desktop_agent.execute_goal(
                translated_command=english_cmd,
                original_persian=persian_text,
            )
            timings["desktop_agent"] = round(time.time() - t_desk_start, 3)
            timings["total"] = round(time.time() - start_total, 3)

            stage = PipelineStage.DONE if desktop_res.success else PipelineStage.ERROR
            self._emit(
                stage,
                desktop_res.message,
                {
                    "persian_text": persian_text,
                    "english_cmd": english_cmd,
                    "executed_results": desktop_res.history,
                    "timings": timings,
                    "error": desktop_res.error,
                },
            )
            return PipelineResult(
                success=desktop_res.success,
                heard_persian=persian_text,
                translated_english=english_cmd,
                planned_tools=[],
                executed_results=desktop_res.history,
                timings=timings,
                error=desktop_res.error,
            )

        # 5. Needle Static Tool Routing Stage
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
