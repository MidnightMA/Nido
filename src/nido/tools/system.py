"""System management, diagnostics, and desktop control tools."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from typing import Any, Dict

from nido.logging import get_logger

logger = get_logger("nido.tools.system")


class SystemTools:
    """Provides system diagnostics and screen management tools."""

    def __init__(self, allow_shutdown: bool = False, allow_reboot: bool = False) -> None:
        self.allow_shutdown = allow_shutdown
        self.allow_reboot = allow_reboot

    def take_screenshot(self) -> Dict[str, Any]:
        """Capture a screenshot of the desktop."""
        # Check for spectacle (KDE Plasma standard)
        spectacle = shutil.which("spectacle")
        if spectacle:
            try:
                subprocess.Popen([spectacle, "-b", "-n"], shell=False)
                return {"success": True, "message": "Screenshot captured using Spectacle."}
            except Exception as e:
                return {"success": False, "error": f"Spectacle failed: {e}"}

        # Check for grim (Wayland)
        grim = shutil.which("grim")
        if grim:
            try:
                save_path = os.path.expanduser("~/Pictures/screenshot.png")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                subprocess.run([grim, save_path], check=True)
                return {"success": True, "message": f"Screenshot saved to {save_path}"}
            except Exception as e:
                return {"success": False, "error": f"grim failed: {e}"}

        # Check for scrot (X11)
        scrot = shutil.which("scrot")
        if scrot:
            try:
                subprocess.Popen([scrot], shell=False)
                return {"success": True, "message": "Screenshot captured using scrot."}
            except Exception as e:
                return {"success": False, "error": f"scrot failed: {e}"}

        return {"success": False, "error": "No screenshot utility found (spectacle/grim/scrot)."}

    def lock_screen(self) -> Dict[str, Any]:
        """Lock the desktop screen session."""
        loginctl = shutil.which("loginctl")
        if loginctl:
            try:
                subprocess.run([loginctl, "lock-session"], check=True)
                return {"success": True, "message": "Screen locked successfully."}
            except Exception as e:
                return {"success": False, "error": f"Failed to lock session: {e}"}

        return {"success": False, "error": "loginctl utility not available."}

    def show_system_info(self) -> Dict[str, Any]:
        """Display basic system information (OS, kernel, CPU, memory)."""
        info: Dict[str, Any] = {
            "os": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "node": platform.node(),
        }

        # Read memory info from /proc/meminfo if on Linux
        meminfo_path = "/proc/meminfo"
        if os.path.isfile(meminfo_path):
            try:
                with open(meminfo_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            info["mem_total"] = line.split(":")[1].strip()
                        elif line.startswith("MemAvailable:"):
                            info["mem_available"] = line.split(":")[1].strip()
            except Exception:
                pass

        summary = f"{info['os']} {info['release']} on {info['machine']}"
        if "mem_available" in info:
            summary += f" (Available RAM: {info['mem_available']})"

        return {
            "success": True,
            "message": summary,
            "details": info,
        }

    def shutdown(self) -> Dict[str, Any]:
        """Power off the computer. Disabled by default in settings for safety."""
        if not self.allow_shutdown:
            return {
                "success": False,
                "error": "Shutdown is disabled by default. Enable 'allow_shutdown = true' in config.toml.",
            }
        try:
            subprocess.run(["systemctl", "poweroff"], check=True)
            return {"success": True, "message": "System powering off."}
        except Exception as e:
            return {"success": False, "error": f"Failed to initiate shutdown: {e}"}

    def reboot(self) -> Dict[str, Any]:
        """Reboot the computer. Disabled by default in settings for safety."""
        if not self.allow_reboot:
            return {
                "success": False,
                "error": "Reboot is disabled by default. Enable 'allow_reboot = true' in config.toml.",
            }
        try:
            subprocess.run(["systemctl", "reboot"], check=True)
            return {"success": True, "message": "System rebooting."}
        except Exception as e:
            return {"success": False, "error": f"Failed to initiate reboot: {e}"}
