"""Configuration management for Nido."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]


@dataclass
class AssistantConfig:
    name: str = "Nido"
    language: str = "en"


@dataclass
class HotkeyConfig:
    key: str = "KEY_F9"
    device: str = ""
    mode_threshold_ms: int = 300


@dataclass
class AudioConfig:
    device: str = ""
    sample_rate: int = 16000
    channels: int = 1
    max_seconds: int = 30
    prebuffer_ms: int = 400


@dataclass
class UIConfig:
    position: str = "top-right"
    timeout_ms: int = 4000
    compact: bool = False


@dataclass
class STTConfig:
    model_dir: str = "~/.local/share/nido/models/sherpa-onnx-streaming-zipformer-en-2023-06-26"
    threads: int = 2
    provider: str = "cpu"
    decoding_method: str = "greedy_search"
    enable_endpoint_detection: bool = True
    rule1_min_trailing_silence: float = 0.8
    rule2_min_trailing_silence: float = 0.6
    rule3_min_utterance_length: float = 20.0


@dataclass
class RealtimeConfig:
    enabled: bool = True
    max_pending_commands: int = 8
    partial_update_interval_ms: int = 80
    finalization_timeout_ms: int = 1500
    show_transcript_history: bool = True
    transcript_history_size: int = 5
    dispatch_stable_partials: bool = False


@dataclass
class LayaConfig:
    enabled: bool = True
    model_dir: str = "~/.local/share/nido/models/laya-multilingual-mlx"
    dtype: str = "float16"
    device: str = "auto"
    batch_size: int = 16
    compile: bool = True
    cache_prompts: bool = True
    pad_to_multiple: int = 16
    max_elements: int = 120
    max_candidates: int = 24
    shortlist_size: int = 20
    confidence_threshold: float = 0.0


@dataclass
class DesktopConfig:
    enabled: bool = True
    max_steps: int = 24
    max_elements: int = 120
    max_depth: int = 32
    settle_delay_ms: int = 120
    action_timeout_ms: int = 2000
    snapshot_timeout_ms: int = 1500
    include_invisible: bool = False
    include_offscreen: bool = False
    accessibility_backend: str = "auto"
    input_backend: str = "auto"


@dataclass
class SystemConfig:
    auto_start: bool = True
    allow_shutdown: bool = False
    allow_reboot: bool = False


DEFAULT_APPS: Dict[str, str] = {
    "browser": "firefox",
    "chrome": "google-chrome",
    "terminal": "konsole",
    "editor": "code",
    "file_manager": "dolphin",
    "files": "dolphin",
    "calculator": "kcalc",
    "settings": "systemsettings",
    "music": "elisa",
    "telegram": "telegram-desktop",
    "discord": "discord",
}


@dataclass
class Config:
    assistant: AssistantConfig = field(default_factory=AssistantConfig)
    hotkey: HotkeyConfig = field(default_factory=HotkeyConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    realtime: RealtimeConfig = field(default_factory=RealtimeConfig)
    laya: LayaConfig = field(default_factory=LayaConfig)
    apps: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_APPS))
    desktop: DesktopConfig = field(default_factory=DesktopConfig)
    system: SystemConfig = field(default_factory=SystemConfig)


def get_default_config_path() -> Path:
    """Return default user configuration file path following XDG spec."""
    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config_home:
        base = Path(xdg_config_home)
    else:
        base = Path.home() / ".config"
    return base / "nido" / "config.toml"


def load_config(config_path: str | Path | None = None) -> Config:
    """Load configuration from TOML file, falling back to defaults for missing keys."""
    config = Config()

    target_path = Path(config_path).expanduser() if config_path else get_default_config_path()

    if not target_path.is_file():
        return config

    try:
        with open(target_path, "rb") as f:
            data: Dict[str, Any] = tomllib.load(f)
    except Exception:
        # On error reading or parsing, return defaults
        return config

    if "assistant" in data and isinstance(data["assistant"], dict):
        a = data["assistant"]
        config.assistant = AssistantConfig(
            name=a.get("name", config.assistant.name),
            language=a.get("language", config.assistant.language),
        )

    if "hotkey" in data and isinstance(data["hotkey"], dict):
        h = data["hotkey"]
        config.hotkey = HotkeyConfig(
            key=h.get("key", config.hotkey.key),
            device=h.get("device", config.hotkey.device),
            mode_threshold_ms=int(h.get("mode_threshold_ms", config.hotkey.mode_threshold_ms)),
        )

    if "audio" in data and isinstance(data["audio"], dict):
        au = data["audio"]
        config.audio = AudioConfig(
            device=au.get("device", config.audio.device),
            sample_rate=int(au.get("sample_rate", config.audio.sample_rate)),
            channels=int(au.get("channels", config.audio.channels)),
            max_seconds=int(au.get("max_seconds", config.audio.max_seconds)),
            prebuffer_ms=int(au.get("prebuffer_ms", config.audio.prebuffer_ms)),
        )

    if "ui" in data and isinstance(data["ui"], dict):
        u = data["ui"]
        config.ui = UIConfig(
            position=u.get("position", config.ui.position),
            timeout_ms=int(u.get("timeout_ms", config.ui.timeout_ms)),
            compact=bool(u.get("compact", config.ui.compact)),
        )

    if "stt" in data and isinstance(data["stt"], dict):
        s = data["stt"]
        stt_model_dir = s.get("model_dir", config.stt.model_dir)
        if "20M" in stt_model_dir or "shenava" in stt_model_dir:
            stt_model_dir = config.stt.model_dir
        config.stt = STTConfig(
            model_dir=stt_model_dir,
            threads=int(s.get("threads", config.stt.threads)),
            provider=str(s.get("provider", config.stt.provider)),
            decoding_method=str(s.get("decoding_method", config.stt.decoding_method)),
            enable_endpoint_detection=bool(s.get("enable_endpoint_detection", config.stt.enable_endpoint_detection)),
            rule1_min_trailing_silence=float(s.get("rule1_min_trailing_silence", config.stt.rule1_min_trailing_silence)),
            rule2_min_trailing_silence=float(s.get("rule2_min_trailing_silence", config.stt.rule2_min_trailing_silence)),
            rule3_min_utterance_length=float(s.get("rule3_min_utterance_length", config.stt.rule3_min_utterance_length)),
        )

    if "realtime" in data and isinstance(data["realtime"], dict):
        rt = data["realtime"]
        config.realtime = RealtimeConfig(
            enabled=bool(rt.get("enabled", config.realtime.enabled)),
            max_pending_commands=int(rt.get("max_pending_commands", config.realtime.max_pending_commands)),
            partial_update_interval_ms=int(rt.get("partial_update_interval_ms", config.realtime.partial_update_interval_ms)),
            finalization_timeout_ms=int(rt.get("finalization_timeout_ms", config.realtime.finalization_timeout_ms)),
            show_transcript_history=bool(rt.get("show_transcript_history", config.realtime.show_transcript_history)),
            transcript_history_size=int(rt.get("transcript_history_size", config.realtime.transcript_history_size)),
            dispatch_stable_partials=bool(rt.get("dispatch_stable_partials", config.realtime.dispatch_stable_partials)),
        )

    if "laya" in data and isinstance(data["laya"], dict):
        ly = data["laya"]
        config.laya = LayaConfig(
            enabled=bool(ly.get("enabled", config.laya.enabled)),
            model_dir=str(ly.get("model_dir", config.laya.model_dir)),
            dtype=str(ly.get("dtype", config.laya.dtype)),
            device=str(ly.get("device", config.laya.device)),
            batch_size=int(ly.get("batch_size", config.laya.batch_size)),
            compile=bool(ly.get("compile", config.laya.compile)),
            cache_prompts=bool(ly.get("cache_prompts", config.laya.cache_prompts)),
            pad_to_multiple=int(ly.get("pad_to_multiple", config.laya.pad_to_multiple)),
            max_elements=int(ly.get("max_elements", config.laya.max_elements)),
            max_candidates=int(ly.get("max_candidates", config.laya.max_candidates)),
            shortlist_size=int(ly.get("shortlist_size", config.laya.shortlist_size)),
            confidence_threshold=float(ly.get("confidence_threshold", config.laya.confidence_threshold)),
        )

    if "apps" in data and isinstance(data["apps"], dict):
        for k, v in data["apps"].items():
            if isinstance(k, str) and isinstance(v, str):
                config.apps[k.lower()] = v

    if "desktop" in data and isinstance(data["desktop"], dict):
        d = data["desktop"]
        config.desktop = DesktopConfig(
            enabled=bool(d.get("enabled", config.desktop.enabled)),
            max_steps=int(d.get("max_steps", config.desktop.max_steps)),
            max_elements=int(d.get("max_elements", config.desktop.max_elements)),
            max_depth=int(d.get("max_depth", config.desktop.max_depth)),
            settle_delay_ms=int(d.get("settle_delay_ms", config.desktop.settle_delay_ms)),
            action_timeout_ms=int(d.get("action_timeout_ms", config.desktop.action_timeout_ms)),
            snapshot_timeout_ms=int(d.get("snapshot_timeout_ms", config.desktop.snapshot_timeout_ms)),
            include_invisible=bool(d.get("include_invisible", config.desktop.include_invisible)),
            include_offscreen=bool(d.get("include_offscreen", config.desktop.include_offscreen)),
            accessibility_backend=str(d.get("accessibility_backend", config.desktop.accessibility_backend)),
            input_backend=str(d.get("input_backend", config.desktop.input_backend)),
        )

    if "system" in data and isinstance(data["system"], dict):
        sys_cfg = data["system"]
        config.system = SystemConfig(
            auto_start=bool(sys_cfg.get("auto_start", config.system.auto_start)),
            allow_shutdown=bool(sys_cfg.get("allow_shutdown", config.system.allow_shutdown)),
            allow_reboot=bool(sys_cfg.get("allow_reboot", config.system.allow_reboot)),
        )

    return config
