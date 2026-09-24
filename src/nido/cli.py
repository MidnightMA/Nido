"""Command-line interface and daemon entry point for Nido."""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from nido import __version__
from nido.audio.capture import MicrophoneCapture
from nido.audio.recorder import AudioRecorder
from nido.config import Config, load_config
from nido.events import EventBus, PipelineEvent, PipelineStage
from nido.hotkey.evdev_backend import EvdevHotkeyBackend, MockHotkeyBackend
from nido.logging import get_logger, setup_logging
from nido.models.manager import ModelManager
from nido.pipeline import AssistantPipeline
from nido.realtime.controller import RealtimeController
from nido.stt.whisper_streaming import (
    MockStreamingSTT,
    StreamingSTT,
    WhisperStreamingSTT,
)
from nido.tools import build_default_registry

logger = get_logger("nido.cli")


def cmd_models_status(config: Config) -> int:
    """Print status of local model files."""
    manager = ModelManager(config)
    statuses = manager.get_status()
    print("Nido Offline Models Status:")
    print("-" * 50)

    # STT
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
    """Transcribe a local WAV audio file using English Zipformer STT."""
    audio_path = Path(wav_path).expanduser().resolve()
    if not audio_path.is_file():
        print(f"Error: File not found: {wav_path}", file=sys.stderr)
        return 1

    try:
        import soundfile as sf
    except ImportError:
        import wave

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
        recognizer: StreamingSTT = ZipformerStreamingSTT(config.stt)
    else:
        print("Note: Zipformer model not installed; using mock recognizer.")
        recognizer = MockStreamingSTT()

    result = recognizer.transcribe_waveform(audio_data, sample_rate=sr)
    print(f"Transcription: {result}")
    return 0


def cmd_test_stt_live(config: Config) -> int:
    """Open microphone and stream real-time English speech transcription to terminal."""
    manager = ModelManager(config)
    if manager.get_status()["stt"].installed:
        stt: StreamingSTT = WhisperStreamingSTT(config.stt)
        if not stt.is_loaded:
            stt = MockStreamingSTT(["open notes", "create a new note", "write hello world"])
    else:
        print("Note: Whisper model not installed; using mock streaming recognizer.")
        stt = MockStreamingSTT(["open notes", "create a new note", "write hello world"])

    capture = MicrophoneCapture(
        device=config.audio.device,
        target_sr=config.audio.sample_rate,
        chunk_duration_ms=50,
        prebuffer_ms=config.audio.prebuffer_ms,
    )

    print("🎙 Opening microphone for streaming English speech recognition...")
    print("Speak commands into your microphone. Press Ctrl+C to exit.\n")

    stt.start_session()
    last_partial = ""

    def on_chunk(chunk: np.ndarray, sr: int) -> None:
        nonlocal last_partial
        stt.feed_audio(chunk, sr)
        if stt.is_endpoint():
            final = stt.finalize().strip()
            if final:
                print(f"\r\033[K[FINAL] {final}\n", flush=True)
            stt.reset_utterance()
            last_partial = ""
        else:
            partial = stt.get_partial_text().strip()
            if partial and partial != last_partial:
                last_partial = partial
                print(f"\r\033[K[LIVE]  {partial}", end="", flush=True)

    capture.add_listener(on_chunk)
    capture.start()

    stop_flag = False

    def _sig_int(sig: int, frame: object) -> None:
        nonlocal stop_flag
        stop_flag = True

    signal.signal(signal.SIGINT, _sig_int)
    signal.signal(signal.SIGTERM, _sig_int)

    try:
        while not stop_flag:
            time.sleep(0.1)
    finally:
        capture.stop()
        stt.stop_session()
        print("\nLive STT stopped.")
    return 0


def cmd_test_realtime(config: Config) -> int:
    """Launch interactive realtime voice mode in terminal without requiring F9 hotkey."""
    print("🎙 Starting Nido Realtime Voice Mode in terminal...")
    print("Commands will be recognized and queued to DesktopAgent automatically.")
    print("Press Ctrl+C to stop.\n")

    event_bus = EventBus()

    def _on_event(event: PipelineEvent) -> None:
        stage = event.stage
        if stage == PipelineStage.STT_PARTIAL:
            partial = event.data.get("text", "")
            print(f"\r\033[K[LIVE]    {partial}", end="", flush=True)
        elif stage == PipelineStage.STT_FINAL:
            final = event.data.get("text", "")
            print(f"\r\033[K[FINAL]   {final}\n", flush=True)
        elif stage == PipelineStage.COMMAND_QUEUED:
            print(f"[QUEUED]  Task {event.data.get('task_id')}: '{event.data.get('goal')}' (queue size: {event.data.get('queue_size')})")
        elif stage == PipelineStage.COMMAND_STARTED:
            print(f"[EXEC]    Starting Task {event.data.get('task_id')}: '{event.data.get('goal')}'")
        elif stage == PipelineStage.COMMAND_COMPLETED:
            print(f"[DONE]    Completed Task {event.data.get('task_id')}")
        elif stage == PipelineStage.COMMAND_FAILED:
            print(f"[FAILED]  Task {event.data.get('task_id')}: {event.data.get('error')}")

    event_bus.subscribe(_on_event)

    manager = ModelManager(config)
    stt: StreamingSTT
    if manager.get_status()["stt"].installed:
        stt = WhisperStreamingSTT(config.stt)
        if not stt.is_loaded:
            stt = MockStreamingSTT(["open notes", "create a new note", "write hello world"])
    else:
        print("Note: Whisper model not installed; using mock streaming recognizer.")
        stt = MockStreamingSTT(["open notes", "create a new note", "write hello world"])

    pipeline = AssistantPipeline(config=config, stt=stt, event_bus=event_bus)
    controller = RealtimeController(
        config=config,
        stt=stt,
        desktop_agent=pipeline.desktop_agent,
        event_bus=event_bus,
    )

    controller.start_realtime()

    stop_flag = False

    def _sig_int(sig: int, frame: object) -> None:
        nonlocal stop_flag
        stop_flag = True

    signal.signal(signal.SIGINT, _sig_int)
    signal.signal(signal.SIGTERM, _sig_int)

    try:
        while not stop_flag:
            time.sleep(0.1)
    finally:
        controller.stop_realtime()
        print("\nRealtime mode stopped.")
    return 0


def cmd_accessibility_status(config: Config) -> int:
    """Print accessibility system status and active desktop environment."""
    from nido.accessibility.atspi import AtspiBackend
    from nido.desktop.input import detect_input_backend

    backend = AtspiBackend()
    input_b = detect_input_backend(config.desktop.input_backend)
    available = backend.is_available()

    print("Desktop Accessibility Diagnostics:")
    print("-" * 50)
    print(f"AT-SPI2 Accessible:   {'✓ Available' if available else '✗ Unavailable'}")
    print(f"Session Type:         {backend.get_session_type()}")
    print(f"Desktop Environment:  {backend.get_desktop_environment()}")
    print(f"Active Input Backend: {type(input_b).__name__}")
    print("-" * 50)

    if not available:
        print("Hint: Ensure AT-SPI2 is enabled in your desktop environment:")
        print("  gsettings set org.gnome.desktop.interface toolkit-accessibility true")
        return 1
    return 0


def cmd_accessibility_tree(config: Config) -> int:
    """Print hierarchical accessibility tree for the active window."""
    from nido.accessibility.atspi import AtspiBackend

    backend = AtspiBackend()
    if not backend.is_available():
        print("Error: AT-SPI2 accessibility is not available.", file=sys.stderr)
        return 1

    snapshot = backend.get_desktop_snapshot(
        max_elements=config.desktop.max_elements,
        max_depth=config.desktop.max_depth,
        include_invisible=config.desktop.include_invisible,
        include_offscreen=config.desktop.include_offscreen,
    )

    print(f"Active Application: {snapshot.active_application or 'None'}")
    print(f"Active Window:      {snapshot.active_window or 'None'}")
    print(f"Total Elements:     {len(snapshot.elements)}")
    print("=" * 60)

    for el in snapshot.elements:
        indent = "  " * el.depth
        states_str = f"[{','.join(el.states)}]" if el.states else ""
        actions_str = f"actions=({','.join(el.actions)})" if el.actions else ""
        name_str = f'"{el.name}"' if el.name else "<unnamed>"
        print(f"{indent}[{el.id}] {el.role} {name_str} {states_str} {actions_str}")
    return 0


def cmd_accessibility_inspect(config: Config) -> int:
    """Inspect detailed properties of all visible elements in the active window."""
    from nido.accessibility.atspi import AtspiBackend

    backend = AtspiBackend()
    snapshot = backend.get_desktop_snapshot(
        max_elements=config.desktop.max_elements,
        max_depth=config.desktop.max_depth,
    )

    print(f"Detailed Inspection for '{snapshot.active_window}':")
    print("=" * 60)
    for el in snapshot.elements:
        print(f"Element [{el.id}]:")
        print(f"     Role:        {el.role}")
        print(f"     Name:        {el.name}")
        print(f"     Description: {el.description}")
        print(f"     States:      {el.states}")
        print(f"     Actions:     {el.actions}")
        if el.value:
            print(f"     Value:       {el.value}")
        if el.text_content:
            print(f"     Text:        {el.text_content[:60]}")
        if el.bounds:
            print(f"     Bounds:      x={el.bounds[0]}, y={el.bounds[1]}, w={el.bounds[2]}, h={el.bounds[3]}")
        print()
    return 0


def cmd_test_laya(config: Config, goal: str) -> int:
    """Test Laya action candidate selection for an English goal with mock/real state."""
    from nido.accessibility.models import DesktopSnapshot, UIElement
    from nido.desktop.candidates import CandidateBuilder
    from nido.laya.agent import LayaDecisionAgent, MockLayaAgent

    print(f"Testing Laya decision selection for goal: '{goal}'")
    print("=" * 60)

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
    res = agent.execute_goal(goal=command)
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
    """Start Nido push-to-talk and realtime daemon with PySide6 overlay."""
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
        stt: StreamingSTT = WhisperStreamingSTT(config.stt)
        if not stt.is_loaded:
            stt = MockStreamingSTT()
    else:
        stt = MockStreamingSTT()

    # Initialize Pipeline & Desktop Agent
    registry = build_default_registry(config)
    event_bus = EventBus()
    pipeline = AssistantPipeline(
        config=config,
        stt=stt,
        registry=registry,
        event_bus=event_bus,
    )

    # Initialize Realtime Controller
    controller = RealtimeController(
        config=config,
        stt=stt,
        desktop_agent=pipeline.desktop_agent,
        event_bus=event_bus,
    )

    # Initialize PySide6 GUI
    try:
        from PySide6.QtWidgets import QApplication
        from nido.ui.overlay import NidoOverlay
        from nido.ui.worker import EventBridge

        app = QApplication(sys.argv)
        overlay = NidoOverlay(config.ui)
        bridge = EventBridge(event_bus)
        bridge.event_received.connect(overlay.update_stage)
    except Exception as e:
        logger.warning(f"Could not initialize PySide6 GUI ({e}). Running in headless console mode.")
        app = None
        overlay = None
        bridge = None

    # Start hotkey listener
    from nido.hotkey.evdev_backend import check_input_permissions
    has_perm, perm_msg = check_input_permissions()
    if not has_perm:
        print("!" * 60)
        print(f"Warning: {perm_msg}")
        print("Dual-mode hotkey (F9) requires access to /dev/input/event*.")
        print("Run: sudo usermod -aG input $USER")
        print("Note: You must log out and log back in for the group to activate.")
        print("!" * 60)

    hotkey_backend: object
    try:
        hotkey_backend = EvdevHotkeyBackend(
            on_press=controller.on_f9_press,
            on_release=controller.on_f9_release,
            key_name=config.hotkey.key,
            device_path=config.hotkey.device,
        )
        hotkey_backend.start()
    except Exception as e:
        logger.warning(f"Could not start evdev hotkey backend: {e}. Falling back to mock hotkey.")
        hotkey_backend = MockHotkeyBackend(
            on_press=controller.on_f9_press,
            on_release=controller.on_f9_release,
        )
        hotkey_backend.start()

    def _sig_handler(sig: int, frame: object) -> None:
        logger.info("Termination signal received. Shutting down Nido...")
        hotkey_backend.stop()
        if controller.is_realtime_active:
            controller.stop_realtime()
        if bridge is not None:
            bridge.cleanup()
        if app is not None:
            app.quit()
        else:
            sys.exit(0)

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    print(f"Nido is running. Hold {config.hotkey.key} for push-to-talk, or tap {config.hotkey.key} for realtime mode.")

    if app is not None:
        return app.exec()
    else:
        signal.pause()
        return 0


def main(argv: Optional[list[str]] = None) -> int:
    """CLI argument parser and dispatcher."""
    parser = argparse.ArgumentParser(
        prog="nido",
        description="Nido: Lightweight, fully offline English voice-controlled desktop assistant for Linux/KDE.",
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

    # diagnostic commands
    stt_p = subparsers.add_parser("test-stt", help="Test English Zipformer STT on a WAV file.")
    stt_p.add_argument("file", type=str, help="Path to input audio file.")

    subparsers.add_parser("test-stt-live", help="Open microphone and test live streaming STT.")
    subparsers.add_parser("test-realtime", help="Run interactive realtime voice mode in terminal.")

    laya_p = subparsers.add_parser("test-laya", help="Test Laya action candidate selection for an English goal.")
    laya_p.add_argument("goal", type=str, help="English goal (e.g. 'Open notes and write Hello world').")

    desk_p = subparsers.add_parser("test-desktop", help="Test multi-step desktop agent interaction.")
    desk_p.add_argument("goal", type=str, help="Desktop interaction goal in English.")

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

    elif args.subcommand == "test-stt-live":
        return cmd_test_stt_live(config)

    elif args.subcommand == "test-realtime":
        return cmd_test_realtime(config)

    elif args.subcommand == "test-laya":
        return cmd_test_laya(config, args.goal)

    elif args.subcommand == "test-desktop":
        return cmd_test_desktop(config, args.goal)

    else:
        # Default: run dual-mode daemon
        return run_daemon(config, debug=args.debug)

    return 0


if __name__ == "__main__":
    sys.exit(main())
