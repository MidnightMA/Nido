"""Orchestrator connecting microphone capture, streaming STT, task queue, and desktop execution."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from nido.audio.capture import MicrophoneCapture
from nido.config import Config, RealtimeConfig
from nido.desktop.agent import DesktopAgent, DesktopAgentResult
from nido.events import EventBus, PipelineEvent, PipelineStage
from nido.logging import get_logger
from nido.realtime.queue import AgentTask, AgentTaskQueue
from nido.realtime.state import F9State, F9StateMachine
from nido.stt.nemotron_streaming import NemotronStreamingSTT, StreamingSTT

logger = get_logger("nido.realtime.controller")


class RealtimeController:
    """Coordinates persistent real-time streaming voice interaction and F9 dual-mode handling."""

    def __init__(
        self,
        config: Config,
        stt: Optional[StreamingSTT] = None,
        desktop_agent: Optional[DesktopAgent] = None,
        capture: Optional[MicrophoneCapture] = None,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        self.config = config
        self.rt_config: RealtimeConfig = getattr(config, "realtime", RealtimeConfig())
        self.event_bus = event_bus or EventBus()

        self.stt: StreamingSTT = stt or NemotronStreamingSTT(config.stt)
        if not self.stt.is_loaded:
            from nido.stt.nemotron_streaming import MockStreamingSTT
            self.stt = MockStreamingSTT()
        self.desktop_agent = desktop_agent
        self.capture = capture or MicrophoneCapture(
            device=config.audio.device,
            target_sr=config.audio.sample_rate,
            chunk_duration_ms=50,
            prebuffer_ms=config.audio.prebuffer_ms,
        )

        self.task_queue = AgentTaskQueue(max_pending=self.rt_config.max_pending_commands)
        self._setup_queue_hooks()

        # State tracking
        self.state_machine = F9StateMachine(
            mode_threshold_ms=config.hotkey.mode_threshold_ms,
            on_enter_ptt=self._on_enter_ptt,
            on_enter_realtime=self._on_enter_realtime,
            on_exit_ptt=self._on_exit_ptt,
            on_exit_realtime=self._on_exit_realtime,
            on_state_changed=self._on_f9_state_changed,
        )

        self._ptt_audio_chunks: List[np.ndarray] = []
        self._last_partial_text = ""
        self._last_partial_time = 0.0
        self._realtime_active = False
        self._lock = threading.Lock()

    @property
    def is_realtime_active(self) -> bool:
        with self._lock:
            return self._realtime_active

    def _emit(self, stage: PipelineStage, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        event = PipelineEvent(stage=stage, message=message, data=data or {})
        logger.info(f"Realtime: [{stage.value.upper()}] {message}")
        self.event_bus.emit(event)

    def _setup_queue_hooks(self) -> None:
        def on_queued(task: AgentTask, qsize: int) -> None:
            self._emit(
                PipelineStage.COMMAND_QUEUED,
                f"Queued command [{task.task_id}]: '{task.goal}'",
                {"task_id": task.task_id, "goal": task.goal, "queue_size": qsize},
            )

        def on_started(task: AgentTask) -> None:
            self._emit(
                PipelineStage.COMMAND_STARTED,
                f"Starting command [{task.task_id}]: '{task.goal}'",
                {"task_id": task.task_id, "goal": task.goal},
            )

        def on_completed(task: AgentTask, result: Any) -> None:
            self._emit(
                PipelineStage.COMMAND_COMPLETED,
                f"Completed command [{task.task_id}]: '{task.goal}'",
                {"task_id": task.task_id, "goal": task.goal, "result": result},
            )

        def on_failed(task: AgentTask, error: str) -> None:
            self._emit(
                PipelineStage.COMMAND_FAILED,
                f"Failed command [{task.task_id}]: '{task.goal}' ({error})",
                {"task_id": task.task_id, "goal": task.goal, "error": error},
            )

        def on_full(goal: str) -> None:
            self._emit(
                PipelineStage.ERROR,
                f"Command queue full. Rejected: '{goal}'",
                {"error": "queue_full", "goal": goal},
            )

        self.task_queue.on_task_queued = on_queued
        self.task_queue.on_task_started = on_started
        self.task_queue.on_task_completed = on_completed
        self.task_queue.on_task_failed = on_failed
        self.task_queue.on_queue_full = on_full

    # -------------------------------------------------------------------------
    # Audio Chunk Listeners
    # -------------------------------------------------------------------------

    def _on_realtime_audio_chunk(self, chunk: np.ndarray, sample_rate: int) -> None:
        """Process an incremental audio chunk during continuous realtime mode."""
        with self._lock:
            if not self._realtime_active:
                return

        # Feed to streaming recognizer
        self.stt.feed_audio(chunk, sample_rate)

        # 1. Check for streaming endpoint
        if self.stt.is_endpoint():
            final_text = self.stt.finalize().strip()
            if final_text:
                logger.info(f"Utterance endpoint detected: '{final_text}'")
                self._emit(
                    PipelineStage.STT_FINAL,
                    f"Final: '{final_text}'",
                    {"text": final_text, "transcript": final_text},
                )
                self.task_queue.enqueue(final_text)
            self.stt.reset_utterance()
            self._last_partial_text = ""
            return

        # 2. Check for partial update
        now = time.monotonic()
        throttle_interval = self.rt_config.partial_update_interval_ms / 1000.0
        if (now - self._last_partial_time) >= throttle_interval:
            self._last_partial_time = now
            partial = self.stt.get_partial_text().strip()
            if partial and partial != self._last_partial_text:
                self._last_partial_text = partial
                self._emit(
                    PipelineStage.STT_PARTIAL,
                    f"Partial: '{partial}'",
                    {"text": partial, "partial": True},
                )

    def _on_ptt_audio_chunk(self, chunk: np.ndarray, sample_rate: int) -> None:
        """Collect audio chunks during PTT hold mode."""
        with self._lock:
            self._ptt_audio_chunks.append(chunk.copy())

    # -------------------------------------------------------------------------
    # Realtime Lifecycle
    # -------------------------------------------------------------------------

    def start_realtime(self) -> None:
        """Activate persistent real-time voice mode."""
        with self._lock:
            if self._realtime_active:
                return
            self._realtime_active = True
            self._last_partial_text = ""
            self._last_partial_time = time.monotonic()

        self.stt.start_session()
        self.capture.add_listener(self._on_realtime_audio_chunk)

        # Start audio capture if not already running
        if not self.capture.is_active:
            self.capture.start()

        # Start serialized agent worker
        self.task_queue.start_worker(self._execute_agent_task)

        self._emit(
            PipelineStage.REALTIME_STARTED,
            "Realtime listening activated. Speak commands naturally.",
            {"mode": "realtime"},
        )

    def stop_realtime(self) -> None:
        """Gracefully terminate persistent real-time voice mode."""
        with self._lock:
            if not self._realtime_active:
                return
            self._realtime_active = False

        self._emit(
            PipelineStage.REALTIME_STOPPING,
            "Stopping realtime listening...",
            {"mode": "realtime"},
        )

        self.capture.remove_listener(self._on_realtime_audio_chunk)

        # Bounded finalization: attempt to finalize any in-flight utterance
        try:
            final_text = self.stt.finalize().strip()
            if final_text and len(final_text) > 2:
                logger.info(f"Finalized speech during realtime shutdown: '{final_text}'")
                self._emit(
                    PipelineStage.STT_FINAL,
                    f"Final: '{final_text}'",
                    {"text": final_text, "transcript": final_text},
                )
                self.task_queue.enqueue(final_text)
        except Exception as e:
            logger.debug(f"Finalization on shutdown exception: {e}")
        finally:
            self.stt.stop_session()

        # If no other capture listeners, stop mic stream
        if not self.capture.is_active:
            self.capture.stop()

        self._emit(
            PipelineStage.REALTIME_STOPPED,
            "Realtime listening stopped.",
            {"mode": "idle"},
        )
        self._emit(PipelineStage.IDLE, "Ready.")

    def _execute_agent_task(self, task: AgentTask) -> Any:
        """Worker function executing a queued command through DesktopAgent."""
        if not self.desktop_agent:
            logger.warning(f"No DesktopAgent configured. Task {task.task_id} skipped.")
            return None

        result: DesktopAgentResult = self.desktop_agent.execute_goal(
            goal=task.goal,
            max_steps=self.config.desktop.max_steps,
        )
        return result

    # -------------------------------------------------------------------------
    # Hotkey State Machine Callbacks
    # -------------------------------------------------------------------------

    def on_f9_press(self) -> None:
        """Called when F9 key is pressed down."""
        # Ensure microphone capture is active and capturing prebuffer
        if not self.capture.is_active:
            self.capture.start()

        state = self.state_machine.handle_key_down()
        if state == F9State.PENDING_PRESS:
            # Check threshold in a background poll or via monotonic checks
            # Start a one-shot watchdog thread to transition to PTT if held
            def _watchdog() -> None:
                while self.state_machine.current_state == F9State.PENDING_PRESS:
                    if self.state_machine.check_hold_threshold():
                        break
                    time.sleep(0.02)

            t = threading.Thread(target=_watchdog, name="F9HoldWatchdog", daemon=True)
            t.start()

    def on_f9_release(self) -> None:
        """Called when F9 key is released."""
        self.state_machine.handle_key_up()

    def _on_enter_ptt(self) -> None:
        """F9 held beyond threshold: enter PTT recording."""
        self._emit(PipelineStage.LISTENING, "Recording push-to-talk command...", {"mode": "ptt"})
        with self._lock:
            self._ptt_audio_chunks.clear()
            # Prepend pre-buffer audio so initial speech is not lost!
            prebuffer = self.capture.get_prebuffer()
            if len(prebuffer) > 0:
                self._ptt_audio_chunks.append(prebuffer.copy())

        self.capture.add_listener(self._on_ptt_audio_chunk)

    def _on_enter_realtime(self) -> None:
        """F9 tapped before threshold: activate persistent realtime mode."""
        self.start_realtime()

    def _on_exit_ptt(self) -> None:
        """F9 released after PTT hold: stop recording and process."""
        self.capture.remove_listener(self._on_ptt_audio_chunk)
        if not self._realtime_active:
            self.capture.stop()

        with self._lock:
            chunks = list(self._ptt_audio_chunks)
            self._ptt_audio_chunks.clear()

        if not chunks:
            self._emit(PipelineStage.ERROR, "No audio recorded.", {"error": "empty_audio"})
            self._emit(PipelineStage.IDLE, "Ready.")
            self.state_machine.finish_processing()
            return

        audio = np.concatenate(chunks).astype(np.float32)

        # Process PTT in background thread
        def _process_ptt() -> None:
            try:
                self._emit(PipelineStage.TRANSCRIBING, "Transcribing English speech...")
                text = self.stt.transcribe_waveform(audio, sample_rate=self.config.audio.sample_rate).strip()
                if not text:
                    self._emit(PipelineStage.ERROR, "No speech detected.", {"error": "no_speech"})
                    return

                self._emit(PipelineStage.STT_FINAL, f"Heard: '{text}'", {"text": text, "transcript": text})

                if self.desktop_agent:
                    self.desktop_agent.execute_goal(goal=text, max_steps=self.config.desktop.max_steps)
            except Exception as e:
                logger.error(f"Error processing PTT audio: {e}")
                self._emit(PipelineStage.ERROR, f"PTT processing failed: {e}", {"error": str(e)})
            finally:
                self._emit(PipelineStage.IDLE, "Ready.")
                self.state_machine.finish_processing()

        t = threading.Thread(target=_process_ptt, name="PTTProcessor", daemon=True)
        t.start()

    def _on_exit_realtime(self) -> None:
        """F9 tapped during realtime mode: stop realtime."""
        self.stop_realtime()

    def _on_f9_state_changed(self, old_state: F9State, new_state: F9State) -> None:
        logger.debug(f"F9 State changed: {old_state.value} -> {new_state.value}")
