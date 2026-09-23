"""Deterministic F9 hotkey state machine distinguishing push-to-talk hold vs realtime tap."""

from __future__ import annotations

import enum
import threading
import time
from typing import Callable, Optional

from nido.logging import get_logger

logger = get_logger("nido.realtime.state")


class F9State(str, enum.Enum):
    IDLE = "IDLE"
    PENDING_PRESS = "PENDING_PRESS"
    PTT_RECORDING = "PTT_RECORDING"
    REALTIME_LISTENING = "REALTIME_LISTENING"
    STOPPING_REALTIME = "STOPPING_REALTIME"
    PROCESSING = "PROCESSING"


class F9StateMachine:
    """Explicit state machine for F9 dual-mode interaction.

    Transitions:
      IDLE
       └─ F9 DOWN → PENDING_PRESS (starts pre-buffering, records monotonic timestamp)
      PENDING_PRESS
       ├─ hold threshold exceeded → PTT_RECORDING (hold detected, prepend pre-buffer)
       └─ F9 UP before threshold → REALTIME_LISTENING (tap detected, activate persistent realtime)
      PTT_RECORDING
       └─ F9 UP → PROCESSING (stop recording, finalize, dispatch to Laya)
      REALTIME_LISTENING
       └─ F9 DOWN → STOPPING_REALTIME → IDLE (graceful stop, bounded finalization)
    """

    def __init__(
        self,
        mode_threshold_ms: int = 300,
        on_enter_ptt: Optional[Callable[[], None]] = None,
        on_enter_realtime: Optional[Callable[[], None]] = None,
        on_exit_ptt: Optional[Callable[[], None]] = None,
        on_exit_realtime: Optional[Callable[[], None]] = None,
        on_state_changed: Optional[Callable[[F9State, F9State], None]] = None,
    ) -> None:
        self.mode_threshold_s = max(0.05, float(mode_threshold_ms) / 1000.0)
        self.on_enter_ptt = on_enter_ptt
        self.on_enter_realtime = on_enter_realtime
        self.on_exit_ptt = on_exit_ptt
        self.on_exit_realtime = on_exit_realtime
        self.on_state_changed = on_state_changed

        self._state = F9State.IDLE
        self._press_time: float = 0.0
        self._lock = threading.Lock()

    @property
    def current_state(self) -> F9State:
        with self._lock:
            return self._state

    def _transition(self, new_state: F9State) -> None:
        old_state = self._state
        if old_state == new_state:
            return
        self._state = new_state
        logger.info(f"F9State transition: {old_state.value} -> {new_state.value}")
        if self.on_state_changed:
            try:
                self.on_state_changed(old_state, new_state)
            except Exception as e:
                logger.error(f"Error in on_state_changed callback: {e}")

    def handle_key_down(self) -> F9State:
        """Handle raw F9 press down event."""
        with self._lock:
            if self._state == F9State.IDLE:
                self._press_time = time.monotonic()
                self._transition(F9State.PENDING_PRESS)
                return self._state

            if self._state == F9State.REALTIME_LISTENING:
                # Key press in realtime mode triggers toggle exit
                self._transition(F9State.STOPPING_REALTIME)
                if self.on_exit_realtime:
                    try:
                        self.on_exit_realtime()
                    except Exception as e:
                        logger.error(f"Error in on_exit_realtime callback: {e}")
                self._transition(F9State.IDLE)
                return self._state

            logger.debug(f"F9 DOWN ignored in state {self._state.value}")
            return self._state

    def check_hold_threshold(self) -> bool:
        """Check if hold threshold has elapsed while in PENDING_PRESS.

        Returns True if transition to PTT_RECORDING occurred.
        """
        with self._lock:
            if self._state == F9State.PENDING_PRESS:
                elapsed = time.monotonic() - self._press_time
                if elapsed >= self.mode_threshold_s:
                    self._transition(F9State.PTT_RECORDING)
                    if self.on_enter_ptt:
                        try:
                            self.on_enter_ptt()
                        except Exception as e:
                            logger.error(f"Error in on_enter_ptt callback: {e}")
                    return True
            return False

    def handle_key_up(self) -> F9State:
        """Handle raw F9 release event."""
        with self._lock:
            if self._state == F9State.PENDING_PRESS:
                # Key released before threshold: TAP -> activate REALTIME
                elapsed = time.monotonic() - self._press_time
                if elapsed < self.mode_threshold_s:
                    self._transition(F9State.REALTIME_LISTENING)
                    if self.on_enter_realtime:
                        try:
                            self.on_enter_realtime()
                        except Exception as e:
                            logger.error(f"Error in on_enter_realtime callback: {e}")
                    return self._state
                else:
                    # Threshold elapsed just before key up
                    self._transition(F9State.PTT_RECORDING)

            if self._state == F9State.PTT_RECORDING:
                self._transition(F9State.PROCESSING)
                if self.on_exit_ptt:
                    try:
                        self.on_exit_ptt()
                    except Exception as e:
                        logger.error(f"Error in on_exit_ptt callback: {e}")
                return self._state

            if self._state == F9State.STOPPING_REALTIME:
                self._transition(F9State.IDLE)
                return self._state

            logger.debug(f"F9 UP ignored in state {self._state.value}")
            return self._state

    def finish_processing(self) -> None:
        """Transition back to IDLE after PTT processing finishes."""
        with self._lock:
            if self._state == F9State.PROCESSING:
                self._transition(F9State.IDLE)
