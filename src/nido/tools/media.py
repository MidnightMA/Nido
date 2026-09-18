"""Media playback control tools using MPRIS and playerctl."""

from __future__ import annotations

import shutil
import subprocess
from typing import Any, Dict

from nido.logging import get_logger

logger = get_logger("nido.tools.media")


def _run_playerctl_cmd(subcmd: str) -> Dict[str, Any]:
    playerctl = shutil.which("playerctl")
    if not playerctl:
        return {
            "success": False,
            "error": "playerctl utility not found. Install playerctl for MPRIS media control.",
        }

    try:
        proc = subprocess.run([playerctl, subcmd], capture_output=True, text=True, check=False)
        if proc.returncode == 0:
            return {"success": True, "message": f"Media command '{subcmd}' executed."}
        else:
            err = proc.stderr.strip() or f"No active media player responded to '{subcmd}'."
            return {"success": False, "error": err}
    except Exception as e:
        return {"success": False, "error": f"Failed to run playerctl: {e}"}


def play_pause() -> Dict[str, Any]:
    """Toggle play/pause on the currently active media player."""
    return _run_playerctl_cmd("play-pause")


def next_track() -> Dict[str, Any]:
    """Skip to the next audio/video track."""
    return _run_playerctl_cmd("next")


def previous_track() -> Dict[str, Any]:
    """Return to the previous audio/video track."""
    return _run_playerctl_cmd("previous")
