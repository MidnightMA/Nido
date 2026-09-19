"""evdev-based global push-to-talk hotkey backend for Linux (X11 & Wayland)."""

from __future__ import annotations

import glob
import os
import select
import threading
import time
from typing import Callable, List, Optional

try:
    import evdev
    from evdev import InputDevice, ecodes
except ImportError:
    evdev = None  # type: ignore[assignment]
    ecodes = None  # type: ignore[assignment]
    InputDevice = None  # type: ignore[assignment]

from nido.hotkey.base import HotkeyBackend
from nido.logging import get_logger

logger = get_logger("nido.hotkey")


def check_input_permissions() -> tuple[bool, str]:
    """Check if /dev/input/event* nodes exist and are readable by the current user."""
    nodes = glob.glob("/dev/input/event*")
    if not nodes:
        return False, "No /dev/input/event* input nodes found."
    readable = [p for p in nodes if os.access(p, os.R_OK)]
    if not readable:
        return (
            False,
            "Permission denied on all /dev/input/event* devices. "
            "Add user to 'input' group: 'sudo usermod -aG input $USER' and log out/in.",
        )
    return True, f"Found {len(readable)} readable input device(s)."


def find_keyboard_devices(target_keycode: int) -> List[str]:
    """Find all input devices that report keyboard capability and target_keycode."""
    if evdev is None:
        return []

    devices: List[str] = []
    permission_denied_count = 0
    total_nodes = 0

    for path in glob.glob("/dev/input/event*"):
        total_nodes += 1
        try:
            device = evdev.InputDevice(path)
            capabilities = device.capabilities(verbose=False)
            if ecodes.EV_KEY in capabilities:
                keys = capabilities[ecodes.EV_KEY]
                if target_keycode in keys:
                    devices.append(path)
            device.close()
        except PermissionError:
            permission_denied_count += 1
        except OSError:
            continue

    if permission_denied_count > 0 and not devices:
        logger.warning(
            f"Permission denied accessing {permission_denied_count}/{total_nodes} input devices. "
            "Ensure your user is in the 'input' group: 'sudo usermod -aG input $USER' "
            "and restart your session/logout."
        )

    return devices


class EvdevHotkeyBackend(HotkeyBackend):
    """Listens for push-to-talk key press and release using Linux evdev.

    Works under both Wayland and X11 by reading kernel input events directly.
    Requires user to be in the 'input' group:
        sudo usermod -aG input $USER
    """

    def __init__(
        self,
        on_press: Callable[[], None],
        on_release: Callable[[], None],
        key_name: str = "KEY_F9",
        device_path: Optional[str] = None,
    ) -> None:
        self.on_press = on_press
        self.on_release = on_release
        self.key_name = key_name
        self.device_path = device_path if device_path else None

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._is_pressed = False
        self._lock = threading.Lock()

        # Resolve target key code
        if ecodes is not None and hasattr(ecodes, key_name):
            self.target_keycode = getattr(ecodes, key_name)
        else:
            self.target_keycode = 67  # Default KEY_F9 code

    @property
    def is_pressed(self) -> bool:
        with self._lock:
            return self._is_pressed

    def start(self) -> None:
        """Start the background evdev event listener thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, name="EvdevHotkeyThread", daemon=True)
        self._thread.start()
        logger.info(f"Evdev hotkey listener started for key {self.key_name} (code: {self.target_keycode}).")

    def stop(self) -> None:
        """Stop listening for hotkey events."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        with self._lock:
            if self._is_pressed:
                self._is_pressed = False
                try:
                    self.on_release()
                except Exception as e:
                    logger.error(f"Error calling on_release during stop: {e}")
        logger.info("Evdev hotkey listener stopped.")

    def _open_devices(self) -> List[evdev.InputDevice]:
        """Open target devices or auto-discovered keyboards."""
        if evdev is None:
            return []

        paths: List[str] = []
        if self.device_path:
            paths = [self.device_path]
        else:
            paths = find_keyboard_devices(self.target_keycode)

        opened: List[evdev.InputDevice] = []
        for path in paths:
            try:
                dev = evdev.InputDevice(path)
                opened.append(dev)
                logger.debug(f"Monitoring keyboard device: {dev.name} ({path})")
            except (PermissionError, OSError) as e:
                logger.warning(f"Could not open device {path}: {e}")

        return opened

    def _run_loop(self) -> None:
        if evdev is None:
            logger.error("evdev package not installed. Hotkey listener cannot run.")
            return

        while self._running:
            devices = self._open_devices()
            if not devices:
                logger.warning(
                    f"No keyboard device with key {self.key_name} accessible. "
                    "Ensure user is in 'input' group: sudo usermod -aG input $USER. "
                    "Retrying discovery in 3 seconds..."
                )
                time.sleep(3.0)
                continue

            dev_map = {dev.fd: dev for dev in devices}

            try:
                while self._running:
                    r, _, _ = select.select(list(dev_map.keys()), [], [], 0.5)
                    if not r:
                        continue

                    for fd in r:
                        dev = dev_map[fd]
                        for event in dev.read():
                            if event.type == ecodes.EV_KEY and event.code == self.target_keycode:
                                self._handle_key_event(event.value)
            except (OSError, Exception) as e:
                logger.warning(f"Device read error: {e}. Reconnecting...")
            finally:
                for dev in devices:
                    try:
                        dev.close()
                    except Exception:
                        pass
                time.sleep(1.0)

    def _handle_key_event(self, value: int) -> None:
        """Process key events: 1=down, 0=up, 2=repeat."""
        # Value 2 is auto-repeat: ignore completely
        if value == 2:
            return

        with self._lock:
            if value == 1:
                # Key press
                if not self._is_pressed:
                    self._is_pressed = True
                    logger.debug("Push-to-talk key pressed (F9 DOWN).")
                    try:
                        self.on_press()
                    except Exception as e:
                        logger.error(f"Error in on_press handler: {e}")
            elif value == 0:
                # Key release
                if self._is_pressed:
                    self._is_pressed = False
                    logger.debug("Push-to-talk key released (F9 UP).")
                    try:
                        self.on_release()
                    except Exception as e:
                        logger.error(f"Error in on_release handler: {e}")


class MockHotkeyBackend(HotkeyBackend):
    """In-memory mock hotkey backend for unit testing and headless environments."""

    def __init__(
        self,
        on_press: Callable[[], None],
        on_release: Callable[[], None],
    ) -> None:
        self.on_press = on_press
        self.on_release = on_release
        self.is_running = False
        self.is_pressed = False

    def start(self) -> None:
        self.is_running = True

    def stop(self) -> None:
        self.is_running = False
        if self.is_pressed:
            self.is_pressed = False
            self.on_release()

    def simulate_press(self) -> None:
        """Simulate physical key press, ignoring duplicate press while already held."""
        if not self.is_pressed:
            self.is_pressed = True
            self.on_press()

    def simulate_release(self) -> None:
        """Simulate physical key release, ignoring duplicate release if not held."""
        if self.is_pressed:
            self.is_pressed = False
            self.on_release()

    def simulate_repeat(self) -> None:
        """Simulate key auto-repeat, which must do nothing."""
        pass
