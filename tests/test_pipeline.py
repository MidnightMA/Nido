"""Tests for full assistant pipeline orchestration."""

import numpy as np
import pytest
from nido.config import Config
from nido.events import EventBus, PipelineEvent, PipelineStage
from nido.needle.agent import NeedleCommandRouter, PlannedToolCall
from nido.pipeline import AssistantPipeline, PipelineResult
from nido.stt.shenava import MockSpeechRecognizer
from nido.tools.registry import ToolRegistry
from nido.translation.marian_ct2 import MockTranslator


def test_full_pipeline_happy_path() -> None:
    config = Config()
    stt = MockSpeechRecognizer("کروم رو باز کن")
    translator = MockTranslator()

    registry = ToolRegistry()

    @registry.register(name="open_app")
    def mock_open_app(app_name: str) -> dict:
        return {"success": True, "message": f"App {app_name} opened"}

    router = NeedleCommandRouter(registry=registry)
    event_bus = EventBus()

    recorded_stages = []

    def on_event(ev: PipelineEvent) -> None:
        recorded_stages.append(ev.stage)

    event_bus.subscribe(on_event)

    pipeline = AssistantPipeline(
        config=config,
        stt=stt,
        translator=translator,
        router=router,
        registry=registry,
        event_bus=event_bus,
    )

    # 1 second of audio at 16kHz
    dummy_audio = np.zeros(16000, dtype=np.float32)
    result = pipeline.process_audio(dummy_audio)

    assert result.success is True
    assert result.heard_persian == "کروم رو باز کن"
    assert result.translated_english == "Open Chrome"
    assert len(result.planned_tools) >= 1
    assert result.planned_tools[0].tool_name == "open_app"
    assert len(result.executed_results) >= 1
    assert result.executed_results[0]["success"] is True

    # Check stage event lifecycle
    assert PipelineStage.TRANSCRIBING in recorded_stages
    assert PipelineStage.TRANSLATING in recorded_stages
    assert PipelineStage.THINKING in recorded_stages
    assert PipelineStage.EXECUTING in recorded_stages
    assert PipelineStage.DONE in recorded_stages


def test_pipeline_multi_tool_command() -> None:
    config = Config()

    class MultiCommandTranslator(MockTranslator):
        def translate(self, text: str) -> str:
            return "open Chrome and go to https://youtube.com"

    registry = ToolRegistry()

    @registry.register(name="open_app")
    def mock_open_app(app_name: str) -> dict:
        return {"success": True, "message": f"Opened {app_name}"}

    @registry.register(name="open_url")
    def mock_open_url(url: str) -> dict:
        return {"success": True, "message": f"Opened {url}"}

    router = NeedleCommandRouter(registry=registry)
    pipeline = AssistantPipeline(
        config=config,
        stt=MockSpeechRecognizer("کروم رو باز کن و برو یوتیوب"),
        translator=MultiCommandTranslator(),
        router=router,
        registry=registry,
    )

    audio = np.zeros(16000, dtype=np.float32)
    result = pipeline.process_audio(audio)

    assert result.success is True
    assert len(result.planned_tools) == 2
    tool_names = [t.tool_name for t in result.planned_tools]
    assert "open_app" in tool_names
    assert "open_url" in tool_names
    assert len(result.executed_results) == 2


def test_pipeline_empty_audio() -> None:
    config = Config()
    registry = ToolRegistry()
    pipeline = AssistantPipeline(
        config=config,
        stt=MockSpeechRecognizer(),
        translator=MockTranslator(),
        router=NeedleCommandRouter(registry=registry),
        registry=registry,
    )

    empty_audio = np.zeros(0, dtype=np.float32)
    result = pipeline.process_audio(empty_audio)
    assert result.success is False
    assert "No audio recorded" in result.error


def test_pipeline_tool_failure_does_not_crash() -> None:
    config = Config()
    registry = ToolRegistry()

    @registry.register(name="open_app")
    def failing_tool(app_name: str) -> dict:
        raise RuntimeError("Subprocess failed to launch")

    pipeline = AssistantPipeline(
        config=config,
        stt=MockSpeechRecognizer("کروم رو باز کن"),
        translator=MockTranslator(),
        router=NeedleCommandRouter(registry=registry),
        registry=registry,
    )

    audio = np.zeros(16000, dtype=np.float32)
    result = pipeline.process_audio(audio)

    # Assistant stays alive and returns graceful failure
    assert result.success is True
    assert len(result.executed_results) == 1
    assert result.executed_results[0]["success"] is False
    assert "Subprocess failed to launch" in result.executed_results[0]["error"]
