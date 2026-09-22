"""Synthetic keyboard and coordinate input backend for Linux desktop."""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import List, Protocol

from nido.logging import get_logger

logger = get_logger("nido.desktop.input")


class InputBackend(Protocol):
    """Abstract protocol for synthetic keyboard and mouse input."""

    def is_available(self) -> bool:
        ...

    def press_key(self, key: str) -> bool:
        ...

    def press_hotkey(self, keys: List[str]) -> bool:
        ...

    def type_text(self, text: str) -> bool:
        ...

    def click_coords(self, x: int, y: int) -> bool:
        ...


class WaylandInputBackend:
    """Wayland-compatible synthetic input using wtype or ydotool."""

    def __init__(self) -> None:
        self.wtype_path = shutil.which("wtype")
        self.ydotool_path = shutil.which("ydotool")

    def is_available(self) -> bool:
        return bool(self.wtype_path or self.ydotool_path)

    def press_key(self, key: str) -> bool:
        if self.wtype_path:
            try:
                subprocess.run([self.wtype_path, "-k", key], check=True, timeout=2)
                return True
            except Exception as e:
                logger.debug(f"wtype press_key failed: {e}")
        if self.ydotool_path:
            try:
                subprocess.run([self.ydotool_path, "key", f"{key}:1", f"{key}:0"], check=True, timeout=2)
                return True
            except Exception as e:
                logger.debug(f"ydotool press_key failed: {e}")
        return False

    def press_hotkey(self, keys: List[str]) -> bool:
        if not keys:
            return False
        # Normalize key sequence
        combo = "+".join(keys)
        if self.wtype_path:
            try:
                # wtype modifiers: -M ctrl -k c -m ctrl
                cmd = [self.wtype_path]
                for k in keys[:-1]:
                    cmd.extend(["-M", k])
                cmd.extend(["-k", keys[-1]])
                for k in reversed(keys[:-1]):
                    cmd.extend(["-m", k])
                subprocess.run(cmd, check=True, timeout=2)
                return True
            except Exception as e:
                logger.debug(f"wtype hotkey failed: {e}")
        return self.press_key(combo)

    def type_text(self, text: str) -> bool:
        if self.wtype_path:
            try:
                subprocess.run([self.wtype_path, text], check=True, timeout=3)
                return True
            except Exception as e:
                logger.debug(f"wtype type_text failed: {e}")
        if self.ydotool_path:
            try:
                subprocess.run([self.ydotool_path, "type", text], check=True, timeout=3)
                return True
            except Exception as e:
                logger.debug(f"ydotool type_text failed: {e}")
        return False

    def click_coords(self, x: int, y: int) -> bool:
        if self.ydotool_path:
            try:
                # Move absolute then left click (0xC0 is left click in ydotool)
                subprocess.run([self.ydotool_path, "mousemove", "-a", str(x), str(y)], check=True, timeout=2)
                subprocess.run([self.ydotool_path, "click", "0xC0"], check=True, timeout=2)
                return True
            except Exception as e:
                logger.debug(f"ydotool click failed: {e}")
        return False


class X11InputBackend:
    """X11 synthetic input backend using xdotool."""

    def __init__(self) -> None:
        self.xdotool_path = shutil.which("xdotool")

    def is_available(self) -> bool:
        return bool(self.xdotool_path)

    def press_key(self, key: str) -> bool:
        if not self.xdotool_path:
            return False
        try:
            subprocess.run([self.xdotool_path, "key", key], check=True, timeout=2)
            return True
        except Exception as e:
            logger.debug(f"xdotool press_key failed: {e}")
            return False

    def press_hotkey(self, keys: List[str]) -> bool:
        if not self.xdotool_path or not keys:
            return False
        combo = "+".join(keys)
        try:
            subprocess.run([self.xdotool_path, "key", combo], check=True, timeout=2)
            return True
        except Exception as e:
            logger.debug(f"xdotool hotkey failed: {e}")
            return False

    def type_text(self, text: str) -> bool:
        if not self.xdotool_path:
            return False
        try:
            subprocess.run([self.xdotool_path, "type", "--clearmodifiers", text], check=True, timeout=3)
            return True
        except Exception as e:
            logger.debug(f"xdotool type_text failed: {e}")
            return False

    def click_coords(self, x: int, y: int) -> bool:
        if not self.xdotool_path:
            return False
        try:
            subprocess.run([self.xdotool_path, "mousemove", str(x), str(y), "click", "1"], check=True, timeout=2)
            return True
        except Exception as e:
            logger.debug(f"xdotool click failed: {e}")
            return False


class MockInputBackend:
    """Mock input backend for testing and headless environments."""

    def __init__(self) -> None:
        self.pressed_keys: List[str] = []
        self.typed_texts: List[str] = []
        self.clicks: List[Tuple[int, int]] = []

    def is_available(self) -> bool:
        return True

    def press_key(self, key: str) -> bool:
        self.pressed_keys.append(key)
        return True

    def press_hotkey(self, keys: List[str]) -> bool:
        self.pressed_keys.append("+".join(keys))
        return True

    def type_text(self, text: str) -> bool:
        self.typed_texts.append(text)
        return True

    def click_coords(self, x: int, y: int) -> bool:
        self.clicks.append((x, y))
        return True


def detect_input_backend(preference: str = "auto") -> InputBackend:
    """Detect and return the appropriate input backend for the desktop session."""
    pref = preference.lower()
    if pref == "mock":
        return MockInputBackend()
    if pref == "wayland":
        return WaylandInputBackend()
    if pref == "x11":
        return X11InputBackend()

    session = os.environ.get("XDG_SESSION_TYPE", "").lower()
    if session == "wayland":
        wb = WaylandInputBackend()
        if wb.is_available():
            return wb
    elif session == "x11":
        xb = X11InputBackend()
        if xb.is_available():
            return xb

    # Auto probe
    wb = WaylandInputBackend()
    if wb.is_available():
        return wb
    xb = X11InputBackend()
    if xb.is_available():
        return xb

    logger.info("No system input tools (wtype/ydotool/xdotool) found. Using MockInputBackend.")
    return MockInputBackend()
