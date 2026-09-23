"""Tests for full assistant pipeline orchestration with English streaming STT and Laya agent."""

import numpy as np
import pytest

from nido.accessibility.models import DesktopSnapshot, UIElement
from nido.config import Config
from nido.desktop.agent import DesktopAgent
from nido.desktop.input import MockInputBackend
from nido.events import EventBus, PipelineEvent, PipelineStage
from nido.laya.agent import MockLayaAgent
from nido.pipeline import AssistantPipeline, PipelineResult
from nido.stt.zipformer_en import MockStreamingSTT
from nido.tools.registry import ToolRegistry
from tests.test_desktop_agent import MockAccessibilityBackend


def test_full_pipeline_happy_path() -> None:
    config = Config()
    config.desktop.settle_delay_ms = 0
    stt = MockStreamingSTT(["open chrome"])

    registry = ToolRegistry()

    @registry.register(name="open_app")
    def mock_open_app(app_name: str) -> dict:
        return {"success": True, "message": f"App {app_name} opened"}

    event_bus = EventBus()
    recorded_stages = []
    event_bus.subscribe(lambda ev: recorded_stages.append(ev.stage))

    desktop_agent = DesktopAgent(
        config=config.desktop,
        laya_agent=MockLayaAgent(config.laya),
        accessibility=MockAccessibilityBackend(),
        input_backend=MockInputBackend(),
        registry=registry,
        event_bus=event_bus,
    )

    pipeline = AssistantPipeline(
        config=config,
        stt=stt,
        desktop_agent=desktop_agent,
        registry=registry,
        event_bus=event_bus,
    )

    # 1 second of audio at 16kHz
    dummy_audio = np.zeros(16000, dtype=np.float32)
    result = pipeline.process_audio(dummy_audio)

    assert result.success is True
    assert result.transcript == "open chrome"
    assert len(result.executed_results) >= 1

    # Check stage event lifecycle
    assert PipelineStage.TRANSCRIBING in recorded_stages
    assert PipelineStage.OBSERVING in recorded_stages
    assert PipelineStage.PLANNING in recorded_stages
    assert PipelineStage.INTERACTING in recorded_stages
    assert PipelineStage.DONE in recorded_stages


def test_pipeline_empty_audio() -> None:
    config = Config()
    pipeline = AssistantPipeline(
        config=config,
        stt=MockStreamingSTT(),
    )

    empty_audio = np.zeros(0, dtype=np.float32)
    result = pipeline.process_audio(empty_audio)
    assert result.success is False
    assert "No audio recorded" in result.error


def test_pipeline_desktop_english_text_entry() -> None:
    config = Config()
    config.desktop.settle_delay_ms = 0

    snap = DesktopSnapshot(
        active_application="Kate",
        active_window="Untitled — Kate",
        elements=[
            UIElement(id="e1", role="text field", name="Editor", states=["editable"], focused=True),
        ],
    )
    acc_backend = MockAccessibilityBackend(snapshots=[snap])
    event_bus = EventBus()

    recorded_stages = []
    event_bus.subscribe(lambda ev: recorded_stages.append(ev.stage))

    desktop_agent = DesktopAgent(
        config=config.desktop,
        laya_agent=MockLayaAgent(config.laya),
        accessibility=acc_backend,
        input_backend=MockInputBackend(),
        event_bus=event_bus,
    )

    pipeline = AssistantPipeline(
        config=config,
        stt=MockStreamingSTT(["open kate and write hello world"]),
        desktop_agent=desktop_agent,
        event_bus=event_bus,
    )

    audio = np.zeros(16000, dtype=np.float32)
    result = pipeline.process_audio(audio)

    assert result.success is True
    assert result.transcript == "open kate and write hello world"
    assert len(acc_backend.texts_set) == 1
    # Verifies exact English text is preserved
    assert acc_backend.texts_set[0] == ("e1", "hello world")


def test_pipeline_tool_failure_does_not_crash() -> None:
    config = Config()
    config.desktop.settle_delay_ms = 0
    registry = ToolRegistry()

    @registry.register(name="open_app")
    def failing_tool(app_name: str) -> dict:
        raise RuntimeError("Subprocess failed to launch")

    acc_backend = MockAccessibilityBackend()
    desktop_agent = DesktopAgent(
        config=config.desktop,
        laya_agent=MockLayaAgent(config.laya),
        accessibility=acc_backend,
        input_backend=MockInputBackend(),
        registry=registry,
    )

    pipeline = AssistantPipeline(
        config=config,
        stt=MockStreamingSTT(["open chrome"]),
        desktop_agent=desktop_agent,
        registry=registry,
    )

    audio = np.zeros(16000, dtype=np.float32)
    result = pipeline.process_audio(audio)

    assert result.transcript == "open chrome"
    assert len(result.executed_results) >= 1
    assert result.executed_results[0]["success"] is False
