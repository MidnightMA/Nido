"""Safe file navigation and search tools."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from nido.logging import get_logger

logger = get_logger("nido.tools.files")


def open_file(path: str) -> Dict[str, Any]:
    """Open a file with the default desktop application.

    Args:
        path: Path to the file to open.
    """
    file_path = Path(path).expanduser().resolve()
    if not file_path.exists():
        return {"success": False, "error": f"File does not exist: '{path}'"}
    if not file_path.is_file():
        return {"success": False, "error": f"Path is not a regular file: '{path}'"}

    xdg_open = shutil.which("xdg-open")
    if not xdg_open:
        return {"success": False, "error": "xdg-open utility not found."}

    try:
        subprocess.Popen(
            [xdg_open, str(file_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            shell=False,
        )
        return {"success": True, "message": f"Opened file: {file_path.name}", "path": str(file_path)}
    except Exception as e:
        return {"success": False, "error": f"Failed to open file: {e}"}


def open_folder(path: str) -> Dict[str, Any]:
    """Open a folder in the system file manager (e.g. Dolphin).

    Args:
        path: Directory path to open (e.g. '~/Downloads' or '~/Documents').
    """
    dir_path = Path(path).expanduser().resolve()
    if not dir_path.exists():
        return {"success": False, "error": f"Folder does not exist: '{path}'"}
    if not dir_path.is_dir():
        return {"success": False, "error": f"Path is not a directory: '{path}'"}

    xdg_open = shutil.which("xdg-open")
    if not xdg_open:
        return {"success": False, "error": "xdg-open utility not found."}

    try:
        subprocess.Popen(
            [xdg_open, str(dir_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            shell=False,
        )
        return {"success": True, "message": f"Opened folder: {dir_path.name}", "path": str(dir_path)}
    except Exception as e:
        return {"success": False, "error": f"Failed to open folder: {e}"}


def find_files(query: str, search_dir: str = "~") -> Dict[str, Any]:
    """Search for files matching a name query within a directory.

    Args:
        query: Substring or pattern to match in file names.
        search_dir: Base directory to search within (default: '~').
    """
    base = Path(search_dir).expanduser().resolve()
    if not base.is_dir():
        return {"success": False, "error": f"Search directory invalid: {search_dir}"}

    results: List[str] = []
    pattern = f"*{query}*"

    try:
        # Search up to max 10 files to keep result concise
        for p in base.rglob(pattern):
            if p.is_file():
                results.append(str(p))
                if len(results) >= 10:
                    break
        return {
            "success": True,
            "count": len(results),
            "files": results,
            "message": f"Found {len(results)} files matching '{query}'.",
        }
    except Exception as e:
        return {"success": False, "error": f"File search error: {e}"}
