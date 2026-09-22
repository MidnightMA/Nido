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
from nido.pipeline import AssistantPipeline
from nido.stt.shenava import MockSpeechRecognizer, ShenavaSherpaRecognizer
from nido.tools import build_default_registry

logger = get_logger("nido.cli")


def cmd_models_status(config: Config) -> int:
    """Print status of local model files."""
    manager = ModelManager(config)
    statuses = manager.get_status()
    print("Nido Offline Models Status:")
    print("-" * 50)

    # Shenava STT
    stt = statuses.get("stt")
    if stt:
        state_icon = "✓" if stt.installed else "✗"
        print(f"[{state_icon}] {stt.name}")
        print(f"    Directory: {stt.directory}")
        print(f"    Status:    {stt.details}")

    # Laya MLX
    laya = statuses.get("laya")
    if laya:
        state_icon = "✓" if laya.installed else "✗"
        print(f"\n[{state_icon}] {laya.name}")
        print(f"    Checkpoint:    {laya.checkpoint}")
        print(f"    Directory:     {laya.directory}")
        print(f"    Present:       {'yes' if laya.installed else 'no'}")
        print(f"    Size:          {laya.size_mb} MB")
        print(f"    Runtime:       {laya.runtime}")
        print(f"    Device:        {laya.device}")
        print(f"    Offline-Ready: {'yes' if laya.offline_ready else 'no'}")

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


def cmd_accessibility_status(config: Config) -> int:
    """Print accessibility system status and active desktop environment."""
    from nido.accessibility.atspi import AtspiBackend
    from nido.desktop.input import detect_input_backend

    backend = AtspiBackend()
    input_b = detect_input_backend(config.desktop.input_backend)
    available = backend.is_available()

    print("Nido Desktop Accessibility Status:")
    print("-" * 50)
    print(f"AT-SPI2 Available:     {'✓ YES' if available else '✗ NO (Accessibility bus unreachable)'}")
    snapshot = backend.get_desktop_snapshot(max_elements=10)
    print(f"Session Type:          {snapshot.session_type}")
    print(f"Desktop Environment:   {snapshot.desktop_name}")
    print(f"Active Application:    {snapshot.active_application or 'None'}")
    print(f"Active Window:         {snapshot.active_window or 'None'}")
    print(f"Input Backend:         {type(input_b).__name__}")
    print("-" * 50)
    if not available:
        print("To enable AT-SPI2 on Ubuntu/Kubuntu:")
        print("  sudo apt install at-spi2-core libatk-adaptor")
        print("  gsettings set org.gnome.desktop.interface toolkit-accessibility true")
    return 0 if available else 1


def cmd_accessibility_tree(config: Config) -> int:
    """Print the accessible element tree of the currently active window."""
    from nido.accessibility.atspi import AtspiBackend
    from nido.accessibility.snapshot import format_snapshot_for_prompt

    backend = AtspiBackend()
    if not backend.is_available():
        print("Error: AT-SPI2 accessibility subsystem is not available.", file=sys.stderr)
        return 1

    snapshot = backend.get_desktop_snapshot(
        max_elements=config.desktop.max_elements,
        max_depth=config.desktop.max_depth,
    )
    print(format_snapshot_for_prompt(snapshot))
    return 0


def cmd_accessibility_inspect(config: Config) -> int:
    """Inspect detailed properties of accessible elements in the active window."""
    from nido.accessibility.atspi import AtspiBackend

    backend = AtspiBackend()
    if not backend.is_available():
        print("Error: AT-SPI2 accessibility subsystem is not available.", file=sys.stderr)
        return 1

    snapshot = backend.get_desktop_snapshot(max_elements=50)
    print(f"Inspecting active window: '{snapshot.active_window}' (App: {snapshot.active_application})")
    print(f"Total elements: {len(snapshot.elements)}")
    print("=" * 60)
    for el in snapshot.elements:
        print(f"[{el.id}] Role: {el.role:<15} Name: \"{el.name}\"")
        if el.description:
            print(f"     Description: {el.description}")
        if el.value:
            print(f"     Value:       {el.value}")
        if el.states:
            print(f"     States:      {', '.join(el.states)}")
        if el.actions:
            print(f"     Actions:     {', '.join(el.actions)}")
        if el.bounds:
            print(f"     Bounds:      x={el.bounds[0]}, y={el.bounds[1]}, w={el.bounds[2]}, h={el.bounds[3]}")
        print()
    return 0


def cmd_test_laya(config: Config, goal: str) -> int:
    """Test Laya action candidate selection on Persian text with mock/real state."""
    from nido.accessibility.models import DesktopSnapshot, UIElement
    from nido.desktop.candidates import CandidateBuilder
    from nido.laya.agent import LayaDecisionAgent, MockLayaAgent

    print(f"Testing Laya decision selection for Persian goal: '{goal}'")
    print("=" * 60)

    # Create sample snapshot with realistic Kate editor
    snapshot = DesktopSnapshot(
        session_type="wayland",
        desktop_name="KDE Plasma",
        active_application="Kate",
        active_window="Untitled — Kate",
        elements=[
            UIElement(id="e1", role="menu item", name="File", actions=["click"]),
            UIElement(id="e2", role="menu item", name="Edit", actions=["click"]),
            UIElement(id="e3", role="button", name="Save", actions=["click"]),
            UIElement(id="e4", role="text field", name="Editor", states=["editable"], focused=True),
        ],
    )

    registry = build_default_registry(config)
    builder = CandidateBuilder(app_map=config.apps, max_candidates=config.laya.max_candidates)
    candidates = builder.build_candidates(goal=goal, snapshot=snapshot, registry=registry)

    print(f"Generated Candidates ({len(candidates)}):")
    for c in candidates:
        print(f"  [{c.id}] {c.label} ({c.action_type})")
    print()

    # Load agent or mock
    manager = ModelManager(config)
    laya_installed = manager.get_status()["laya"].installed
    if laya_installed:
        agent = LayaDecisionAgent(config.laya)
        if not agent.is_loaded:
            agent = MockLayaAgent(config.laya)
    else:
        print("Note: Laya model not installed; using MockLayaAgent.")
        agent = MockLayaAgent(config.laya)

    state_text = f"USER GOAL\n{goal}\n\nCURRENT STEP\n1 / 24\n\nAVAILABLE ACTIONS\n" + "\n".join(f"[{c.id}] {c.label}" for c in candidates)
    decision = agent.predict_action(state_text, candidates)

    selected_cand = next((c for c in candidates if c.id == decision.selected_id), None)
    cand_label = selected_cand.label if selected_cand else decision.selected_id

    print("Decision Result:")
    print(f"  Selected ID:   {decision.selected_id}")
    print(f"  Action:        {cand_label}")
    print(f"  Confidence:    {decision.confidence:.2f}")
    if decision.probabilities:
        print("  Top Probabilities:")
        sorted_probs = sorted(decision.probabilities.items(), key=lambda x: x[1], reverse=True)[:5]
        for cid, p in sorted_probs:
            print(f"    {cid}: {p:.3f}")
    return 0


def cmd_test_desktop(config: Config, command: str) -> int:
    """Execute a desktop interaction goal directly through the Laya multi-step agent loop."""
    from nido.accessibility.atspi import AtspiBackend
    from nido.desktop.agent import DesktopAgent
    from nido.desktop.candidates import CandidateBuilder
    from nido.desktop.input import detect_input_backend
    from nido.laya.agent import LayaDecisionAgent, MockLayaAgent

    backend = AtspiBackend()
    input_b = detect_input_backend(config.desktop.input_backend)
    registry = build_default_registry(config)

    manager = ModelManager(config)
    if manager.get_status()["laya"].installed:
        laya_agent = LayaDecisionAgent(config.laya)
        if not laya_agent.is_loaded:
            laya_agent = MockLayaAgent(config.laya)
    else:
        laya_agent = MockLayaAgent(config.laya)

    cand_builder = CandidateBuilder(app_map=config.apps, max_candidates=config.laya.max_candidates)
    agent = DesktopAgent(
        config=config.desktop,
        laya_agent=laya_agent,
        accessibility=backend,
        input_backend=input_b,
        candidate_builder=cand_builder,
        registry=registry,
    )

    print(f"Testing desktop interaction for goal: '{command}'")
    print("=" * 60)
    res = agent.execute_goal(original_persian=command)
    print("=" * 60)
    status_str = "✓ SUCCESS" if res.success else "✗ FAILED"
    print(f"Outcome: {status_str} in {res.steps} step(s)")
    print(f"Message: {res.message}")
    if res.history:
        print("\nAction History:")
        for h in res.history:
            s_mark = "✓" if h.get("success") else "✗"
            print(f"  Step {h['step']}: [{s_mark}] {h['action']}({h.get('arguments')}) -> {h.get('message') or h.get('error')}")
    return 0 if res.success else 1


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

    # Initialize Registry, Laya Agent, and Pipeline
    registry = build_default_registry(config)
    event_bus = EventBus()
    pipeline = AssistantPipeline(
        config=config,
        stt=stt,
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
    from nido.hotkey.evdev_backend import check_input_permissions
    has_perm, perm_msg = check_input_permissions()
    if not has_perm:
        print("!" * 60)
        print(f"Warning: {perm_msg}")
        print("Push-to-talk (F9) requires access to /dev/input/event*.")
        print("Run: sudo usermod -aG input $USER")
        print("Note: You must log out and log back in for the group to activate.")
        print("!" * 60)

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

    # accessibility subcommands
    access_p = subparsers.add_parser("accessibility", help="Accessibility inspection and diagnostics.")
    access_sub = access_p.add_subparsers(dest="access_action", required=True)
    access_sub.add_parser("status", help="Check AT-SPI2 and desktop accessibility status.")
    access_sub.add_parser("tree", help="Print accessible tree of active window.")
    access_sub.add_parser("inspect", help="Inspect detailed element properties in active window.")

    # standalone diagnostic commands
    stt_p = subparsers.add_parser("test-stt", help="Test Persian STT on a WAV file.")
    stt_p.add_argument("file", type=str, help="Path to input audio file.")

    laya_p = subparsers.add_parser("test-laya", help="Test Laya action candidate selection for a Persian goal.")
    laya_p.add_argument("goal", type=str, help="Persian goal (e.g. 'نوت را باز کن و بنویس سلام دنیا').")

    desk_p = subparsers.add_parser("test-desktop", help="Test multi-step desktop agent interaction.")
    desk_p.add_argument("goal", type=str, help="Desktop interaction goal in Persian.")

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

    elif args.subcommand == "accessibility":
        if args.access_action == "status":
            return cmd_accessibility_status(config)
        elif args.access_action == "tree":
            return cmd_accessibility_tree(config)
        elif args.access_action == "inspect":
            return cmd_accessibility_inspect(config)

    elif args.subcommand == "test-stt":
        return cmd_test_stt(config, args.file)

    elif args.subcommand == "test-laya":
        return cmd_test_laya(config, args.goal)

    elif args.subcommand == "test-desktop":
        return cmd_test_desktop(config, args.goal)

    else:
        # Default: run push-to-talk daemon
        return run_daemon(config, debug=args.debug)

    return 0


if __name__ == "__main__":
    sys.exit(main())
