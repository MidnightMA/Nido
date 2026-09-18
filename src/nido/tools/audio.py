"""System audio volume control tools."""

from __future__ import annotations

import shutil
import subprocess
from typing import Any, Dict

from nido.logging import get_logger

logger = get_logger("nido.tools.audio")


def _run_audio_cmd(cmd: list[str]) -> bool:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return proc.returncode == 0
    except Exception as e:
        logger.error(f"Failed to execute audio command {cmd}: {e}")
        return False


def set_volume(percent: int) -> Dict[str, Any]:
    """Set system audio output volume.

    Args:
        percent: Volume percentage from 0 to 100.
    """
    percent = max(0, min(100, int(percent)))

    # Try pactl first (standard on modern Linux/PipeWire/PulseAudio)
    if shutil.which("pactl"):
        if _run_audio_cmd(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{percent}%"]):
            return {"success": True, "message": f"Volume set to {percent}%.", "volume": percent}

    # Try pamixer
    if shutil.which("pamixer"):
        if _run_audio_cmd(["pamixer", "--set-volume", str(percent)]):
            return {"success": True, "message": f"Volume set to {percent}%.", "volume": percent}

    # Try amixer (ALSA)
    if shutil.which("amixer"):
        if _run_audio_cmd(["amixer", "set", "Master", f"{percent}%"]):
            return {"success": True, "message": f"Volume set to {percent}%.", "volume": percent}

    return {"success": False, "error": "No supported audio control utility found (pactl/pamixer/amixer)."}


def increase_volume(step: int = 5) -> Dict[str, Any]:
    """Increase system audio volume by a percentage step.

    Args:
        step: Percentage increase, default is 5%.
    """
    step = max(1, min(50, int(step)))

    if shutil.which("pactl"):
        if _run_audio_cmd(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"+{step}%"]):
            return {"success": True, "message": f"Volume increased by {step}%."}

    if shutil.which("pamixer"):
        if _run_audio_cmd(["pamixer", "--increase", str(step)]):
            return {"success": True, "message": f"Volume increased by {step}%."}

    if shutil.which("amixer"):
        if _run_audio_cmd(["amixer", "set", "Master", f"{step}%+"]):
            return {"success": True, "message": f"Volume increased by {step}%."}

    return {"success": False, "error": "No supported audio control utility found."}


def decrease_volume(step: int = 5) -> Dict[str, Any]:
    """Decrease system audio volume by a percentage step.

    Args:
        step: Percentage decrease, default is 5%.
    """
    step = max(1, min(50, int(step)))

    if shutil.which("pactl"):
        if _run_audio_cmd(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"-{step}%"]):
            return {"success": True, "message": f"Volume decreased by {step}%."}

    if shutil.which("pamixer"):
        if _run_audio_cmd(["pamixer", "--decrease", str(step)]):
            return {"success": True, "message": f"Volume decreased by {step}%."}

    if shutil.which("amixer"):
        if _run_audio_cmd(["amixer", "set", "Master", f"{step}%-"]):
            return {"success": True, "message": f"Volume decreased by {step}%."}

    return {"success": False, "error": "No supported audio control utility found."}


def mute_volume() -> Dict[str, Any]:
    """Mute system audio output."""
    if shutil.which("pactl"):
        if _run_audio_cmd(["pactl", "set-sink-mute", "@DEFAULT_SINK@", "1"]):
            return {"success": True, "message": "Audio muted."}

    if shutil.which("pamixer"):
        if _run_audio_cmd(["pamixer", "--mute"]):
            return {"success": True, "message": "Audio muted."}

    if shutil.which("amixer"):
        if _run_audio_cmd(["amixer", "set", "Master", "mute"]):
            return {"success": True, "message": "Audio muted."}

    return {"success": False, "error": "No supported audio control utility found."}


def unmute_volume() -> Dict[str, Any]:
    """Unmute system audio output."""
    if shutil.which("pactl"):
        if _run_audio_cmd(["pactl", "set-sink-mute", "@DEFAULT_SINK@", "0"]):
            return {"success": True, "message": "Audio unmuted."}

    if shutil.which("pamixer"):
        if _run_audio_cmd(["pamixer", "--unmute"]):
            return {"success": True, "message": "Audio unmuted."}

    if shutil.which("amixer"):
        if _run_audio_cmd(["amixer", "set", "Master", "unmute"]):
            return {"success": True, "message": "Audio unmuted."}

    return {"success": False, "error": "No supported audio control utility found."}
