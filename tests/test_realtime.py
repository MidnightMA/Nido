"""Comprehensive unit tests for F9 state machine, audio pre-buffering, and realtime task queue."""

import time
import numpy as np

from nido.audio.capture import MicrophoneCapture, MockMicrophoneCapture
from nido.config import Config
from nido.desktop.agent import DesktopAgent, DesktopAgentResult
from nido.events import EventBus, PipelineEvent, PipelineStage
from nido.realtime.controller import RealtimeController
from nido.realtime.queue import AgentTask, AgentTaskQueue
from nido.realtime.state import F9State, F9StateMachine
from nido.stt.zipformer_en import MockStreamingSTT


def test_f9_tap_detection() -> None:
    """F9 DOWN, wait < threshold, F9 UP -> REALTIME_LISTENING."""
    sm = F9StateMachine(mode_threshold_ms=200)
    assert sm.current_state == F9State.IDLE

    # Key down
    s1 = sm.handle_key_down()
    assert s1 == F9State.PENDING_PRESS

    # Release quickly (< 200ms)
    time.sleep(0.05)
    s2 = sm.handle_key_up()
    assert s2 == F9State.REALTIME_LISTENING
    assert sm.current_state == F9State.REALTIME_LISTENING


def test_f9_hold_detection() -> None:
    """F9 DOWN, wait >= threshold -> PTT_RECORDING, F9 UP -> PROCESSING."""
    sm = F9StateMachine(mode_threshold_ms=100)

    # Key down
    sm.handle_key_down()
    assert sm.current_state == F9State.PENDING_PRESS

    # Wait past threshold
    time.sleep(0.12)
    transitioned = sm.check_hold_threshold()
    assert transitioned is True
    assert sm.current_state == F9State.PTT_RECORDING

    # Key up
    s2 = sm.handle_key_up()
    assert s2 == F9State.PROCESSING
    assert sm.current_state == F9State.PROCESSING

    sm.finish_processing()
    assert sm.current_state == F9State.IDLE


def test_f9_repeat_events_ignored() -> None:
    """Auto-repeat key down events while already held must not trigger duplicate transitions."""
    sm = F9StateMachine(mode_threshold_ms=200)
    sm.handle_key_down()
    assert sm.current_state == F9State.PENDING_PRESS

    # Simulate repeated down event
    s2 = sm.handle_key_down()
    assert s2 == F9State.PENDING_PRESS
    assert sm.current_state == F9State.PENDING_PRESS


def test_realtime_toggle() -> None:
    """IDLE -> tap F9 -> REALTIME -> tap F9 -> IDLE."""
    sm = F9StateMachine(mode_threshold_ms=150)
    assert sm.current_state == F9State.IDLE

    # 1. Tap to enter realtime
    sm.handle_key_down()
    sm.handle_key_up()
    assert sm.current_state == F9State.REALTIME_LISTENING

    # 2. Tap to exit realtime
    sm.handle_key_down()
    assert sm.current_state == F9State.IDLE


def test_audio_prebuffer_preservation() -> None:
    """Verify that initial audio captured before hold threshold is preserved in pre-buffer."""
    capture = MicrophoneCapture(
        device="",
        target_sr=16000,
        chunk_duration_ms=50,
        prebuffer_ms=400,
    )

    # Simulate feeding chunks into prebuffer manually
    test_chunk1 = np.ones(800, dtype=np.float32) * 0.5
    test_chunk2 = np.ones(800, dtype=np.float32) * 0.8
    with capture._lock:
        capture._prebuffer.append(test_chunk1)
        capture._prebuffer.append(test_chunk2)

    buf = capture.get_prebuffer()
    assert len(buf) == 1600
    assert np.allclose(buf[:800], 0.5)
    assert np.allclose(buf[800:], 0.8)

    capture.clear_prebuffer()
    assert len(capture.get_prebuffer()) == 0


def test_realtime_partials_and_finalization() -> None:
    """Verify streaming STT emits partial updates and finalizes into a single command."""
    config = Config()
    event_bus = EventBus()
    recorded_events = []
    event_bus.subscribe(lambda ev: recorded_events.append(ev))

    mock_stt = MockStreamingSTT(["open notes and create a new note"])
    controller = RealtimeController(
        config=config,
        stt=mock_stt,
        capture=MockMicrophoneCapture(),
        event_bus=event_bus,
    )

    controller.start_realtime()

    # Feed audio chunks to trigger partial and endpoint
    dummy_chunk = np.zeros(4000, dtype=np.float32)
    controller._on_realtime_audio_chunk(dummy_chunk, 16000)
    controller._on_realtime_audio_chunk(dummy_chunk, 16000)

    stages = [ev.stage for ev in recorded_events]
    assert PipelineStage.REALTIME_STARTED in stages
    assert PipelineStage.STT_FINAL in stages
    assert PipelineStage.COMMAND_QUEUED in stages

    controller.stop_realtime()


def test_command_queue_serial_execution() -> None:
    """Verify multiple queued commands execute sequentially in order."""
    queue = AgentTaskQueue(max_pending=4)
    execution_order = []

    def dummy_executor(task: AgentTask) -> dict:
        time.sleep(0.05)
        execution_order.append(task.goal)
        return {"success": True}

    queue.start_worker(dummy_executor)

    t1 = queue.enqueue("command 1")
    t2 = queue.enqueue("command 2")
    t3 = queue.enqueue("command 3")

    assert t1 is not None and t1.task_id == "rt-0001"
    assert t2 is not None and t2.task_id == "rt-0002"
    assert t3 is not None and t3.task_id == "rt-0003"

    # Wait for completion
    time.sleep(0.3)
    queue.stop_worker()

    assert execution_order == ["command 1", "command 2", "command 3"]


def test_command_queue_backpressure() -> None:
    """Verify bounded queue rejects tasks gracefully when full."""
    queue = AgentTaskQueue(max_pending=2)
    full_called = []
    queue.on_queue_full = lambda goal: full_called.append(goal)

    t1 = queue.enqueue("task 1")
    t2 = queue.enqueue("task 2")
    t3 = queue.enqueue("task 3")

    assert t1 is not None
    assert t2 is not None
    assert t3 is None  # Rejected
    assert len(full_called) == 1
    assert full_called[0] == "task 3"


def test_concurrent_capture_while_agent_executes() -> None:
    """Verify microphone capture and STT feeding continue while an agent task is executing."""
    config = Config()
    event_bus = EventBus()
    events = []
    event_bus.subscribe(lambda ev: events.append(ev))

    mock_stt = MockStreamingSTT(["task 1", "task 2"])

    class SlowDesktopAgent:
        def execute_goal(self, goal: str, max_steps: int = 24):
            time.sleep(0.15)
            return DesktopAgentResult(success=True, message="Done", steps=1)

    controller = RealtimeController(
        config=config,
        stt=mock_stt,
        desktop_agent=SlowDesktopAgent(),
        capture=MockMicrophoneCapture(),
        event_bus=event_bus,
    )

    controller.start_realtime()

    # Enqueue task 1
    controller.task_queue.enqueue("task 1")

    # While task 1 is executing, feed audio for task 2
    time.sleep(0.05)
    assert controller.task_queue.is_executing is True

    # Feed chunks
    chunk = np.zeros(8000, dtype=np.float32)
    controller._on_realtime_audio_chunk(chunk, 16000)

    time.sleep(0.3)
    controller.stop_realtime()

    stages = [ev.stage for ev in events]
    assert PipelineStage.COMMAND_STARTED in stages
    assert PipelineStage.COMMAND_COMPLETED in stages


def test_stop_during_speech_bounded_finalization() -> None:
    """Verify stopping realtime while speaking attempts bounded finalization and returns to IDLE."""
    config = Config()
    event_bus = EventBus()
    events = []
    event_bus.subscribe(lambda ev: events.append(ev))

    mock_stt = MockStreamingSTT(["open dolphin and go to downloads"])
    controller = RealtimeController(
        config=config,
        stt=mock_stt,
        capture=MockMicrophoneCapture(),
        event_bus=event_bus,
    )

    controller.start_realtime()
    assert controller.is_realtime_active is True

    # User speaks without endpoint
    chunk = np.zeros(2000, dtype=np.float32)
    mock_stt.feed_audio(chunk)

    # Stop realtime
    controller.stop_realtime()
    assert controller.is_realtime_active is False

    stages = [ev.stage for ev in events]
    assert PipelineStage.REALTIME_STOPPED in stages
    assert PipelineStage.IDLE in stages
