"""Command-line interface and daemon entry point for Nido."""

from __future__ import annotations

import argparse
import os
import signal
import sys
from pathlib import Path
from typing import Optional

from nido import __version__
from nido.audio.recorder import AudioRecorder
from nido.config import Config, load_config
from nido.events import EventBus, PipelineStage
from nido.hotkey.evdev_backend import EvdevHotkeyBackend, MockHotkeyBackend
from nido.logging import get_logger, setup_logging
from nido.models.manager import ModelManager
from nido.needle.agent import NeedleCommandRouter
from nido.pipeline import AssistantPipeline
from nido.stt.shenava import MockSpeechRecognizer, ShenavaSherpaRecognizer
from nido.tools import build_default_registry
from nido.translation.marian_ct2 import MockTranslator, MarianCT2Translator

logger = get_logger("nido.cli")


def cmd_models_status(config: Config) -> int:
    """Print status of local model files."""
    manager = ModelManager(config)
    statuses = manager.get_status()
    print("Nido Offline Models Status:")
    print("-" * 50)
    for key, stat in statuses.items():
        state_icon = "✓" if stat.installed else "✗"
        print(f"[{state_icon}] {stat.name}")
        print(f"    Directory: {stat.directory}")
        print(f"    Status:    {stat.details}")
    print("-" * 50)
    all_ok = all(s.installed for s in statuses.values())
    if not all_ok:
        print("To download and prepare missing models, run: nido models setup")
    return 0 if all_ok else 1


def cmd_models_setup(config: Config) -> int:
    """Download and prepare all offline models."""
    manager = ModelManager(config)
    success = manager.setup_all()
    return 0 if success else 1


def cmd_models_verify(config: Config) -> int:
    """Verify local models through test inference."""
    manager = ModelManager(config)
    print("Verifying model inference capabilities...")
    res = manager.verify_models()
    for name, ok in res.items():
        mark = "✓ PASS" if ok else "✗ FAIL"
        print(f"{mark}: {name}")
    all_pass = all(res.values())
    return 0 if all_pass else 1


def cmd_tools_list(config: Config) -> int:
    """List all registered tools and parameter descriptions."""
    registry = build_default_registry(config)
    tools = registry.list_tools()
    print(f"Nido Registered Tools ({len(tools)} total):")
    print("=" * 60)
    for tool in sorted(tools, key=lambda t: t.name):
        print(f"• {tool.name}")
        print(f"  Description: {tool.description}")
        if tool.parameters:
            params_str = ", ".join(
                f"{p}: {info.get('type', 'Any')}" for p, info in tool.parameters.items()
            )
            print(f"  Parameters:  {params_str}")
        print()
    return 0


def cmd_test_stt(config: Config, wav_path: str) -> int:
    """Transcribe a local WAV audio file using Shenava STT."""
    audio_path = Path(wav_path).expanduser().resolve()
    if not audio_path.is_file():
        print(f"Error: File not found: {wav_path}", file=sys.stderr)
        return 1

    try:
        import soundfile as sf
    except ImportError:
        import wave
        import numpy as np

        with wave.open(str(audio_path), "rb") as wf:
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            raw_bytes = wf.readframes(n_frames)
            audio_data = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        audio_data, sr = sf.read(str(audio_path), dtype="float32")

    manager = ModelManager(config)
    statuses = manager.get_status()
    if statuses["stt"].installed:
        recognizer = ShenavaSherpaRecognizer(config.stt.model_dir, threads=config.stt.threads)
    else:
        print("Note: Shenava model not installed; using mock recognizer.")
        recognizer = MockSpeechRecognizer()

    result = recognizer.transcribe(audio_data, sample_rate=sr)
    print(f"Transcription: {result}")
    return 0


def cmd_test_translate(config: Config, persian_text: str) -> int:
    """Translate Persian text to English command using Marian CTranslate2."""
    manager = ModelManager(config)
    statuses = manager.get_status()
    if statuses["translation"].installed:
        translator = MarianCT2Translator(
            model_dir=config.translation.model_dir,
            compute_type=config.translation.compute_type,
            beam_size=config.translation.beam_size,
            max_tokens=config.translation.max_tokens,
        )
    else:
        print("Note: Translation model not installed; using mock translator.")
        translator = MockTranslator()

    english = translator.translate(persian_text)
    print(f"Persian:    {persian_text}")
    print(f"Translated: {english}")
    return 0


def cmd_test_command(config: Config, command: str) -> int:
    """Test Needle tool routing and safe tool execution for an English command."""
    registry = build_default_registry(config)
    router = NeedleCommandRouter(
        registry=registry,
        max_steps=config.needle.max_steps,
        max_new_tokens=config.needle.max_new_tokens,
    )

    print(f"Command: '{command}'")
    calls = router.plan(command)
    print(f"Selected Tools ({len(calls)}):")
    for idx, c in enumerate(calls, 1):
        print(f"  {idx}. {c.tool_name}({c.arguments})")

    if not calls:
        print("No matching tool planned.")
        return 0

    print("\nExecuting Planned Tools:")
    for c in calls:
        res = registry.execute(c.tool_name, c.arguments)
        status = "✓ SUCCESS" if res.get("success") else "✗ FAILED"
        print(f"  [{status}] {c.tool_name} -> {res.get('message', res.get('error', ''))}")

    return 0


def run_daemon(config: Config, debug: bool = False) -> int:
    """Start Nido push-to-talk daemon with PySide6 overlay."""
    logger.info(f"Starting Nido daemon v{__version__}...")

    # Check model status
    manager = ModelManager(config)
    if not manager.all_installed():
        print("=" * 60)
        print("Notice: Offline models are not yet installed.")
        print("Run 'nido models setup' to download and prepare models.")
        print("Starting in mock-assisted mode so hotkey and UI can be tested.")
        print("=" * 60)

    # Initialize STT
    if manager.get_status()["stt"].installed:
        stt = ShenavaSherpaRecognizer(config.stt.model_dir, threads=config.stt.threads)
    else:
        stt = MockSpeechRecognizer()

    # Initialize Translator
    if manager.get_status()["translation"].installed:
        translator = MarianCT2Translator(
            model_dir=config.translation.model_dir,
            compute_type=config.translation.compute_type,
            beam_size=config.translation.beam_size,
            max_tokens=config.translation.max_tokens,
        )
    else:
        translator = MockTranslator()

    # Initialize Registry, Router, and Pipeline
    registry = build_default_registry(config)
    router = NeedleCommandRouter(
        registry=registry,
        max_steps=config.needle.max_steps,
        max_new_tokens=config.needle.max_new_tokens,
    )
    event_bus = EventBus()
    pipeline = AssistantPipeline(
        config=config,
        stt=stt,
        translator=translator,
        router=router,
        registry=registry,
        event_bus=event_bus,
    )

    # Initialize Audio Recorder
    recorder = AudioRecorder(
        device=config.audio.device,
        target_sr=config.audio.sample_rate,
        max_seconds=config.audio.max_seconds,
    )

    # Initialize PySide6 GUI
    try:
        from PySide6.QtCore import QObject, Signal
        from PySide6.QtWidgets import QApplication
        from nido.ui.overlay import NidoOverlay
        from nido.ui.worker import PipelineWorker

        app = QApplication(sys.argv)
        overlay = NidoOverlay(config.ui)
    except Exception as e:
        logger.warning(f"Could not initialize PySide6 GUI ({e}). Running in headless console mode.")
        app = None
        overlay = None

    # Push-to-talk event coordination
    active_worker: Optional[PipelineWorker] = None

    class TriggerBridge(QObject):
        sig_press = Signal()
        sig_release = Signal(object)

    bridge = TriggerBridge() if app is not None else None

    def _on_gui_press() -> None:
        if overlay is not None:
            overlay.set_listening()

    def _on_gui_release(audio_data: object) -> None:
        nonlocal active_worker
        import numpy as np

        if not isinstance(audio_data, np.ndarray) or len(audio_data) == 0:
            if overlay is not None:
                overlay.update_stage(
                    PipelineStage.ERROR.value,
                    "No audio recorded",
                    {"error": "Empty recording"},
                )
            return

        worker = PipelineWorker(pipeline, audio_data)
        active_worker = worker
        if overlay is not None:
            worker.stage_changed.connect(overlay.update_stage)
        worker.start()

    if bridge is not None:
        bridge.sig_press.connect(_on_gui_press)
        bridge.sig_release.connect(_on_gui_release)

    def on_hotkey_press() -> None:
        logger.debug("Hotkey press detected.")
        try:
            recorder.start()
            if bridge is not None:
                bridge.sig_press.emit()
            else:
                print("🎙 [LISTENING...]")
        except Exception as e:
            logger.error(f"Error on hotkey press: {e}")

    def on_hotkey_release() -> None:
        logger.debug("Hotkey release detected.")
        try:
            audio = recorder.stop()
            if bridge is not None:
                bridge.sig_release.emit(audio)
            else:
                print("🧠 [PROCESSING...]")
                pipeline.process_audio(audio)
        except Exception as e:
            logger.error(f"Error on hotkey release: {e}")

    # Start hotkey listener
    hotkey_backend: object
    try:
        hotkey_backend = EvdevHotkeyBackend(
            on_press=on_hotkey_press,
            on_release=on_hotkey_release,
            key_name=config.hotkey.key,
            device_path=config.hotkey.device,
        )
        hotkey_backend.start()
    except Exception as e:
        logger.warning(f"Could not start evdev hotkey backend: {e}. Falling back to mock hotkey.")
        hotkey_backend = MockHotkeyBackend(on_press=on_hotkey_press, on_release=on_hotkey_release)
        hotkey_backend.start()

    # Handle OS termination signals
    def _sig_handler(sig: int, frame: object) -> None:
        logger.info("Termination signal received. Shutting down Nido...")
        hotkey_backend.stop()
        if app is not None:
            app.quit()
        else:
            sys.exit(0)

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    print(f"Nido is running. Hold {config.hotkey.key} to speak Persian commands.")

    if app is not None:
        return app.exec()
    else:
        signal.pause()
        return 0


def main(argv: Optional[list[str]] = None) -> int:
    """CLI argument parser and dispatcher."""
    parser = argparse.ArgumentParser(
        prog="nido",
        description="Nido: Lightweight, fully offline Persian voice command assistant for Linux/KDE.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument("--config", type=str, default=None, help="Path to config.toml.")

    subparsers = parser.add_subparsers(dest="subcommand", help="Subcommands")

    # models subcommands
    models_p = subparsers.add_parser("models", help="Model management commands.")
    models_sub = models_p.add_subparsers(dest="models_action", required=True)
    models_sub.add_parser("status", help="Check local model file availability.")
    models_sub.add_parser("setup", help="Download and prepare offline model assets.")
    models_sub.add_parser("verify", help="Verify model inference integrity.")

    # tools subcommands
    tools_p = subparsers.add_parser("tools", help="Tool inspection commands.")
    tools_sub = tools_p.add_subparsers(dest="tools_action", required=True)
    tools_sub.add_parser("list", help="List registered tools and schemas.")

    # standalone diagnostic commands
    stt_p = subparsers.add_parser("test-stt", help="Test Persian STT on a WAV file.")
    stt_p.add_argument("file", type=str, help="Path to input audio file.")

    trans_p = subparsers.add_parser("test-translate", help="Test Persian->English translation.")
    trans_p.add_argument("text", type=str, help="Persian text to translate.")

    cmd_p = subparsers.add_parser("test-command", help="Test Needle tool routing for a command.")
    cmd_p.add_argument("command", type=str, help="English command to route.")

    args = parser.parse_args(argv)

    setup_logging(debug=args.debug)
    config = load_config(args.config)

    if args.subcommand == "models":
        if args.models_action == "status":
            return cmd_models_status(config)
        elif args.models_action == "setup":
            return cmd_models_setup(config)
        elif args.models_action == "verify":
            return cmd_models_verify(config)

    elif args.subcommand == "tools":
        if args.tools_action == "list":
            return cmd_tools_list(config)

    elif args.subcommand == "test-stt":
        return cmd_test_stt(config, args.file)

    elif args.subcommand == "test-translate":
        return cmd_test_translate(config, args.text)

    elif args.subcommand == "test-command":
        return cmd_test_command(config, args.command)

    else:
        # Default: run push-to-talk daemon
        return run_daemon(config, debug=args.debug)

    return 0


if __name__ == "__main__":
    sys.exit(main())
