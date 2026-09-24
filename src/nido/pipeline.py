"""Central pipeline orchestration for Nido."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from nido.accessibility.atspi import AtspiBackend
from nido.config import Config
from nido.desktop.agent import DesktopAgent, DesktopAgentResult
from nido.desktop.candidates import CandidateBuilder
from nido.desktop.input import detect_input_backend
from nido.events import EventBus, PipelineEvent, PipelineStage
from nido.laya.agent import LayaDecisionAgent, MockLayaAgent
from nido.logging import get_logger
from nido.stt.whisper_streaming import MockStreamingSTT, StreamingSTT, WhisperStreamingSTT
from nido.tools.registry import ToolRegistry

logger = get_logger("nido.pipeline")


@dataclass
class PipelineResult:
    """Structured result of processing an audio voice command."""

    success: bool
    transcript: str = ""
    heard_persian: str = ""  # Backward-compat alias for transcript
    translated_english: str = ""  # Deprecated
    planned_tools: List[Any] = field(default_factory=list)
    executed_results: List[Dict[str, Any]] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None


class AssistantPipeline:
    """Orchestrates audio capture, English streaming STT, and Laya multi-step agent execution."""

    def __init__(
        self,
        config: Config,
        stt: Optional[Any] = None,
        desktop_agent: Optional[DesktopAgent] = None,
        laya_agent: Optional[Any] = None,
        registry: Optional[ToolRegistry] = None,
        event_bus: Optional[EventBus] = None,
        # Optional backward-compat kwargs
        translator: Optional[Any] = None,
        router: Optional[Any] = None,
    ) -> None:
        self.config = config
        self.stt = stt or WhisperStreamingSTT(config.stt)
        if not self.stt.is_loaded:
            self.stt = MockStreamingSTT()
        self.registry = registry or ToolRegistry()
        self.event_bus = event_bus or EventBus()
        self.current_stage = PipelineStage.IDLE

        self.desktop_agent = desktop_agent
        if self.desktop_agent is None:
            decision_agent = laya_agent
            if decision_agent is None and hasattr(self.config, "laya") and self.config.laya.enabled:
                decision_agent = LayaDecisionAgent(self.config.laya)
                if not decision_agent.is_loaded:
                    decision_agent = MockLayaAgent(self.config.laya)
            elif decision_agent is None:
                decision_agent = MockLayaAgent()

            acc_backend = AtspiBackend()
            input_backend = detect_input_backend(self.config.desktop.input_backend)
            cand_builder = CandidateBuilder(
                app_map=self.config.apps,
                max_candidates=getattr(self.config.laya, "max_candidates", 24) if hasattr(self.config, "laya") else 24,
            )

            self.desktop_agent = DesktopAgent(
                config=self.config.desktop,
                laya_agent=decision_agent,
                accessibility=acc_backend,
                input_backend=input_backend,
                candidate_builder=cand_builder,
                registry=self.registry,
                event_bus=self.event_bus,
            )

    def _emit(self, stage: PipelineStage, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        self.current_stage = stage
        event = PipelineEvent(stage=stage, message=message, data=data or {})
        logger.info(f"Pipeline: [{stage.value.upper()}] {message}")
        self.event_bus.emit(event)

    def process_audio(self, audio: np.ndarray, sample_rate: int = 16000) -> PipelineResult:
        """Process a captured audio recording through English STT and Laya multi-step agent."""
        start_total = time.time()
        timings: Dict[str, float] = {}

        # 1. Audio validation
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

        # 2. English STT Stage
        self._emit(PipelineStage.TRANSCRIBING, "Transcribing English speech...")
        t_stt_start = time.time()
        try:
            if hasattr(self.stt, "transcribe_waveform"):
                transcript = self.stt.transcribe_waveform(audio, sample_rate=sample_rate).strip()
            elif hasattr(self.stt, "transcribe"):
                transcript = self.stt.transcribe(audio, sample_rate=sample_rate).strip()
            else:
                transcript = ""
            timings["stt"] = round(time.time() - t_stt_start, 3)
        except Exception as e:
            timings["stt"] = round(time.time() - t_stt_start, 3)
            err_msg = f"Speech recognition failed: {e}"
            logger.error(err_msg)
            self._emit(PipelineStage.ERROR, err_msg, {"error": err_msg, "timings": timings})
            self._emit(PipelineStage.IDLE, "Ready.")
            return PipelineResult(success=False, error=err_msg, timings=timings)

        if not transcript:
            msg = "No speech detected in audio."
            self._emit(PipelineStage.ERROR, msg, {"error": msg, "timings": timings})
            self._emit(PipelineStage.IDLE, "Ready.")
            return PipelineResult(success=False, error=msg, timings=timings)

        self._emit(
            PipelineStage.TRANSCRIBING,
            f"Heard: '{transcript}'",
            {"text": transcript, "transcript": transcript, "heard_persian": transcript, "timings": timings},
        )

        # 3. Unified Laya-MLX Multi-Step Desktop Agent Loop
        t_desk_start = time.time()
        desktop_res: DesktopAgentResult = self.desktop_agent.execute_goal(
            goal=transcript,
        )
        timings["agent_loop"] = round(time.time() - t_desk_start, 3)
        timings["total"] = round(time.time() - start_total, 3)

        stage = PipelineStage.DONE if desktop_res.success else PipelineStage.ERROR
        self._emit(
            stage,
            desktop_res.message,
            {
                "text": transcript,
                "transcript": transcript,
                "heard_persian": transcript,
                "executed_results": desktop_res.history,
                "steps": desktop_res.steps,
                "timings": timings,
                "error": desktop_res.error,
            },
        )

        return PipelineResult(
            success=desktop_res.success,
            transcript=transcript,
            heard_persian=transcript,
            executed_results=desktop_res.history,
            timings=timings,
            error=desktop_res.error,
        )
