"""Tests for push-to-talk hotkey event handling."""

from nido.hotkey import EvdevHotkeyBackend, MockHotkeyBackend


def test_mock_hotkey_lifecycle() -> None:
    press_count = 0
    release_count = 0

    def on_press() -> None:
        nonlocal press_count
        press_count += 1

    def on_release() -> None:
        nonlocal release_count
        release_count += 1

    backend = MockHotkeyBackend(on_press=on_press, on_release=on_release)
    backend.start()

    # 1. F9 DOWN -> starts recording
    backend.simulate_press()
    assert press_count == 1
    assert release_count == 0
    assert backend.is_pressed is True

    # 2. Repeated events (auto-repeat) while held -> must not duplicate
    backend.simulate_repeat()
    backend.simulate_press()  # duplicate press while already held
    assert press_count == 1
    assert release_count == 0

    # 3. F9 UP -> stops recording
    backend.simulate_release()
    assert press_count == 1
    assert release_count == 1
    assert backend.is_pressed is False

    # Duplicate release -> must not trigger multiple times
    backend.simulate_release()
    assert release_count == 1

    # 4. Second press after completion works
    backend.simulate_press()
    assert press_count == 2
    backend.simulate_release()
    assert release_count == 2

    backend.stop()


def test_evdev_backend_event_handler_filtering() -> None:
    press_count = 0
    release_count = 0

    def on_press() -> None:
        nonlocal press_count
        press_count += 1

    def on_release() -> None:
        nonlocal release_count
        release_count += 1

    backend = EvdevHotkeyBackend(
        on_press=on_press,
        on_release=on_release,
        key_name="KEY_F9",
    )

    # Simulate raw evdev event values:
    # 1 = key down, 2 = key repeat, 0 = key up

    # Key Down (1)
    backend._handle_key_event(1)
    assert press_count == 1
    assert release_count == 0
    assert backend.is_pressed is True

    # Key Repeat (2) -> ignored
    backend._handle_key_event(2)
    backend._handle_key_event(2)
    assert press_count == 1
    assert release_count == 0

    # Redundant Key Down (1) -> ignored because already pressed
    backend._handle_key_event(1)
    assert press_count == 1

    # Key Up (0)
    backend._handle_key_event(0)
    assert press_count == 1
    assert release_count == 1
    assert backend.is_pressed is False

    # Redundant Key Up (0) -> ignored
    backend._handle_key_event(0)
    assert release_count == 1
