"""Application launching and closing tools with strict whitelisting."""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any, Dict, Optional

from nido.config import DEFAULT_APPS
from nido.logging import get_logger

logger = get_logger("nido.tools.apps")


class AppTools:
    """Manages whitelisted desktop application execution."""

    def __init__(self, whitelist: Optional[Dict[str, str]] = None) -> None:
        self.whitelist: Dict[str, str] = dict(DEFAULT_APPS)
        if whitelist:
            for k, v in whitelist.items():
                self.whitelist[k.lower()] = v

    def open_app(self, app_name: str) -> Dict[str, Any]:
        """Launch a desktop application from the configured whitelist.

        Args:
            app_name: Name or alias of the application (e.g. 'browser', 'terminal', 'code', 'firefox', 'chrome').
        """
        normalized = app_name.strip().lower()

        # Check alias in whitelist
        binary = self.whitelist.get(normalized)
        if not binary:
            # Check if app_name is one of the values in the whitelist
            for _, val in self.whitelist.items():
                if val.lower() == normalized:
                    binary = val
                    break

        if not binary:
            return {
                "success": False,
                "error": f"Application '{app_name}' is not in the whitelist.",
                "allowed_apps": list(self.whitelist.keys()),
            }

        executable = shutil.which(binary)
        if not executable:
            return {
                "success": False,
                "error": f"Application binary '{binary}' not found on system.",
            }

        try:
            # Ensure accessibility is enabled for launched Qt and GTK applications
            env = os.environ.copy()
            env["QT_LINUX_ACCESSIBILITY_ALWAYS_ON"] = "1"
            env["QT_ACCESSIBILITY"] = "1"

            # Launch detached subprocess with shell=False
            subprocess.Popen(
                [executable],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                shell=False,
                env=env,
            )
            return {
                "success": True,
                "message": f"Application '{app_name}' ({binary}) launched successfully.",
                "binary": binary,
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to launch '{binary}': {e}",
            }

    def close_app(self, app_name: str) -> Dict[str, Any]:
        """Terminate a running whitelisted application.

        Args:
            app_name: Name or alias of the application to terminate.
        """
        normalized = app_name.strip().lower()
        binary = self.whitelist.get(normalized)
        if not binary:
            for _, val in self.whitelist.items():
                if val.lower() == normalized:
                    binary = val
                    break

        if not binary:
            return {
                "success": False,
                "error": f"Application '{app_name}' is not in the whitelist.",
            }

        try:
            # Use killall or pkill
            pkill = shutil.which("pkill")
            if not pkill:
                return {"success": False, "error": "pkill utility not found."}

            proc = subprocess.run(
                [pkill, "-x", binary],
                capture_output=True,
                text=True,
                shell=False,
            )
            if proc.returncode == 0:
                return {
                    "success": True,
                    "message": f"Closed application '{app_name}' ({binary}).",
                }
            else:
                return {
                    "success": False,
                    "error": f"No running process found for '{binary}'.",
                }
        except Exception as e:
            return {"success": False, "error": f"Failed to close '{binary}': {e}"}
