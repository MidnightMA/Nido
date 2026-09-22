# Nido (نیدو)

> **Lightweight, fully offline Persian voice command assistant for KDE Plasma and Linux desktops.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![Desktop: KDE Plasma](https://img.shields.io/badge/Desktop-KDE%20Plasma%20%2F%20Linux-blueviolet.svg)](https://kde.org/plasma-desktop/)
[![Privacy: 100% Offline](https://img.shields.io/badge/Privacy-100%25%20Offline-success.svg)](#privacy-and-offline-guarantee)

---

## What is Nido?

**Nido** is a push-to-talk voice assistant engineered specifically for Linux desktop environments (optimized for KDE Plasma on Wayland and X11). With Nido, you hold down **F9**, speak a voice command in Persian (فارسی), and release the key.

Nido transcribes the Persian utterance locally, translates it to a concise English command via quantized local neural translation, routes it to structured tool calls using **Needle 3**, and executes the requested system actions safely.

**No cloud APIs. No telemetry. Zero audio leaves your machine.**

---

## Key Features

* **True Push-to-Talk (F9)**: Microphone records strictly while F9 is held down. Release triggers instant audio processing.
* **100% Offline Inference**:
  * **Persian STT**: [Shenava Koochik v1.0](https://huggingface.co/Reza2kn/Shenava-Koochik-v1.0-sherpa-onnx) via `sherpa-onnx` (16 kHz float32 mono).
  * **Persian → English Translation**: [HPLT Marian Opus MT](https://huggingface.co/HPLT/translate-fa-en-v2.0-hplt_opus) accelerated with CTranslate2 (INT8 CPU quantization).
  * **Command Routing**: [Needle 3](https://github.com/Cactus-Compute/needle3) by Cactus Compute with strict typed tool schema definitions.
* **Persian Spoken Number Handling (ITN)**: Built-in Persian Inverse Text Normalization parses numbers like "سی درصد" → `30` or "صد و بیست" → `120`.
* **KDE Plasma Visual Overlay**: A sleek, non-intrusive PySide6 frameless overlay showing the live pipeline state, heard Persian text, translated command, selected tools, execution status, and latency benchmarks.
* **Safe, Whitelisted Tools**:
  * Desktop applications (whitelisted aliases for browsers, terminals, editors, file managers)
  * Browser navigation (HTTP/HTTPS URL validation, Google/DuckDuckGo web search)
  * System audio volume (`pactl` / `pamixer` / `amixer` with 0–100% clamping)
  * Media playback control (MPRIS / `playerctl` for play, pause, skip)
  * Safe file navigation (`xdg-open` for files and folders)
  * Desktop utilities (Spectacle screenshots, screen locking via `loginctl`, hardware telemetry)
* **KDE / Linux Integration**: Systemd user service (`nido.service`) and XDG autostart desktop entry.

---

## Architecture

```text
[ Physical Keyboard (F9 Held) ]
               │
               ▼
   [ evdev Hotkey Backend ]
  (Detects F9 Down / Up, ignores repeat)
               │
               ▼
    [ Audio Recorder Stream ]
  (16 kHz Mono float32 in-memory buffer)
               │
          [ F9 Released ]
               │
               ▼
┌──────────────────────────────────────────────┐
│            AssistantPipeline                 │
│                                              │
│ 1. Shenava STT (sherpa-onnx)                 │
│    Audio Samples ──► Persian Text + ITN      │
│                                              │
│ 2. Marian Translator (CTranslate2 INT8)      │
│    Persian Text ──► Concise English Command  │
│                                              │
│ 3. Needle 3 Command Router                   │
│    English Command ──► Structured Tool Call  │
│                                              │
│ 4. Tool Registry                             │
│    Executes whitelisted Linux action         │
└──────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│        PySide6 Frameless Overlay             │
│  • Heard: "کروم رو باز کن"                   │
│  • Translated: "Open Chrome"                 │
│  • Action: open_app("google-chrome")         │
│  • Status: ✓ Chrome launched (Total: 1.18s)  │
└──────────────────────────────────────────────┘
```

---

## Model Stack

| Component | Model / Upstream | Runtime Engine | Format | Offline Path |
| :--- | :--- | :--- | :--- | :--- |
| **Persian STT** | `Reza2kn/Shenava-Koochik-v1.0-sherpa-onnx` | `sherpa-onnx` | ONNX (float32) | `~/.local/share/nido/models/shenava/` |
| **Translation** | `HPLT/translate-fa-en-v2.0-hplt_opus` | `ctranslate2` | INT8 Marian | `~/.local/share/nido/models/translation-ct2/` |
| **Tool Router**| `Cactus-Compute/needle3` | `cactus-needle` | Structured Tool Router | Reusable in-memory router |

---

## Installation

### 1. Prerequisites

Ensure system audio utilities and development headers are installed:

```bash
# Ubuntu / Debian / KDE Neon / Kubuntu
sudo apt install python3-pip python3-venv libasound2-dev pulseaudio-utils playerctl spectacle \
    python3-gi gir1.2-atspi-2.0 at-spi2-core libatk-adaptor wtype libcairo2-dev

# Arch Linux / Manjaro
sudo pacman -S python python-pip pipewire-pulse playerctl spectacle
```

### 2. Configure evdev Permissions (Push-to-Talk)

For global hotkey detection under both **Wayland** and **X11**, Nido uses the Linux `evdev` kernel subsystem. Add your user account to the `input` group:

```bash
sudo usermod -aG input $USER
```

> **Note:** Log out and log back in for group membership changes to take effect.

### 3. Clone and Setup Environment

```bash
git clone https://github.com/MidnightMA/Nido.git
cd Nido

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"
```

---

## Model Setup

Download and prepare the offline models:

```bash
# Download Shenava STT and convert Marian translation to CTranslate2 INT8
nido models setup

# Check status of local model assets
nido models status

# Verify local model inference
nido models verify
```

---

## Running Nido

### Running from Terminal

```bash
# Start Nido push-to-talk daemon
nido

# Start with verbose debug logging
nido --debug

# Start with a custom configuration file
nido --config ~/.config/nido/config.toml
```

### Usage

1. Press and hold **F9**.
2. Speak your command in Persian (e.g., *"کروم رو باز کن و برو یوتیوب"*).
3. Release **F9**.
4. Nido processes the command and displays progress on the top-right overlay.

---

## Accessibility-Driven Desktop Controller

Nido includes a general desktop interaction controller for **KDE Plasma 6** and Linux desktops (Wayland & X11). It can execute multi-step interactive GUI tasks (clicking buttons, entering text, selecting items, navigating menus, switching windows, and scrolling) using **Linux AT-SPI accessibility information**.

### Core Principles

* **Accessibility-First & OCR-Free**: Nido observes the desktop via native AT-SPI metadata exposed by applications and KDE Plasma. **No OCR, no screenshots, and no vision models are used for UI perception.**
* **Dynamic Action Space**: At each UI step, Nido builds ephemeral target IDs (`[e1]`, `[e2]`, ...) corresponding only to currently visible, actionable elements.
* **Stale Element Protection**: If the UI changes before a requested action is executed, stale element references are rejected immediately (`stale_ui_element`), prompting a fresh observation instead of misclicking.
* **Exact User Text Preservation**: When typing text (e.g. *"نوت را باز کن و بنویس سلام دنیا"*), Nido preserves the exact literal Persian text payload without translating it into English.
* **Hybrid Execution**: Simple commands (volume, mute, lock screen) remain fast static tools, while interactive GUI requests branch into the iterative desktop agent loop.
* **Honest Detection**: If an application does not expose accessible controls, Nido reports `Accessibility unavailable for this window` rather than hallucinating actions.

### Architecture Loop

```text
Goal ("Open Kate and write hello")
  ↓
Observe Desktop (AT-SPI2)
  ↓
Build Compact State & Dynamic Action Space ([e1] button "New", [e2] text area "Editor")
  ↓
Model Decision (e.g. set_ui_text(element_id="e2", text="hello"))
  ↓
Stale Element Validation & Execution (AT-SPI EditableText / Action)
  ↓
Settle & Re-Observe
  ↓
Done / Next Step
```

---

## Standalone Diagnostic Commands

Nido provides diagnostic commands to test each pipeline layer independently:

```bash
# 1. Desktop Accessibility Diagnostics
nido accessibility status       # Check AT-SPI2 bus, session (Wayland/X11), and active window
nido accessibility tree         # Print accessible element tree of active window
nido accessibility inspect      # Inspect detailed element roles, bounds, and actions

# 2. Test Multi-Step Desktop Agent directly (bypassing STT/mic)
nido test-desktop "open Kate and write hello"
nido test-desktop "open kcalc"
nido test-desktop "go to Downloads"

# 3. Test Persian Speech-to-Text on a WAV file
nido test-stt sample.wav

# 4. Test Persian -> English translation
nido test-translate "صدا رو بذار روی سی درصد"

# 5. Test Needle tool routing and execution
nido test-command "open firefox and set volume to 50"

# 6. List all registered tools and their argument schemas
nido tools list
```

---

## Example Spoken Commands

| Persian Spoken Command | English Translation | Executed Tool Call |
| :--- | :--- | :--- |
| `کروم رو باز کن` | Open Chrome | `open_app("google-chrome")` |
| `مرورگر را باز کن و برو یوتیوب` | Open browser and go to YouTube | `open_app("firefox")`, `open_url("https://youtube.com")` |
| `ترمینال رو باز کن` | Open terminal | `open_app("konsole")` |
| `صدا رو بذار روی سی درصد` | Set volume to 30 percent | `set_volume(30)` |
| `صدا را کم کن` | Decrease volume | `decrease_volume(5)` |
| `صدا رو قطع کن` | Mute volume | `mute_volume()` |
| `آهنگ بعدی` | Next track | `next_track()` |
| `پخش رو متوقف کن` | Pause playback | `play_pause()` |
| `عکس از صفحه بگیر` | Take screenshot | `take_screenshot()` |
| `صفحه را قفل کن` | Lock screen | `lock_screen()` |
| `پوشه دانلودها رو باز کن` | Open downloads folder | `open_folder("~/Downloads")` |

---

## Available Tools

All tools are strictly typed, isolated, and return structured JSON responses:

* **Applications (`apps`)**:
  * `open_app(app_name: str)`: Launches whitelisted desktop application.
  * `close_app(app_name: str)`: Terminates whitelisted application process.
* **Browser (`browser`)**:
  * `open_url(url: str)`: Opens HTTP/HTTPS URLs with default browser.
  * `search_web(query: str, engine: str = "google")`: Performs web search.
* **Audio (`audio`)**:
  * `set_volume(percent: int)`: Clamps volume to 0–100%.
  * `increase_volume(step: int = 5)`: Increments output volume.
  * `decrease_volume(step: int = 5)`: Decrements output volume.
  * `mute_volume()` / `unmute_volume()`: Toggles mute state.
* **Media (`media`)**:
  * `play_pause()`: Toggles active media playback.
  * `next_track()` / `previous_track()`: Skips tracks via MPRIS.
* **Files (`files`)**:
  * `open_file(path: str)`: Opens existing file with default app.
  * `open_folder(path: str)`: Opens folder in Dolphin file manager.
  * `find_files(query: str, search_dir: str = "~")`: Finds matching files.
* **System (`system`)**:
  * `take_screenshot()`: Captures desktop with Spectacle / Grim.
  * `show_system_info()`: Displays OS, kernel, and memory stats.
  * `lock_screen()`: Locks desktop session via `loginctl`.

---

## Configuration

Configuration is located at `~/.config/nido/config.toml`:

```toml
[assistant]
name = "Nido"
language = "fa"

[hotkey]
key = "KEY_F9"
device = ""  # Empty for auto-discovery

[audio]
device = ""  # Empty for default input
sample_rate = 16000
channels = 1
max_seconds = 30

[ui]
position = "top-right"  # "top-right", "top-left", "bottom-right", "bottom-left"
timeout_ms = 4000
compact = false

[stt]
model_dir = "~/.local/share/nido/models/shenava"
threads = 4

[translation]
model_dir = "~/.local/share/nido/models/translation-ct2"
compute_type = "int8"
beam_size = 1
max_tokens = 64

[needle]
max_steps = 8
max_new_tokens = 128

[desktop]
enabled = true
max_steps = 24
max_elements = 120
max_depth = 32
settle_delay_ms = 120
action_timeout_ms = 2000
snapshot_timeout_ms = 1500
include_invisible = false
include_offscreen = false
accessibility_backend = "auto"
input_backend = "auto"

[apps]
browser = "firefox"
chrome = "google-chrome"
terminal = "konsole"
editor = "code"
file_manager = "dolphin"
files = "dolphin"
calculator = "kcalc"
settings = "systemsettings"
music = "elisa"
telegram = "telegram-desktop"
discord = "discord"

[system]
auto_start = true
allow_shutdown = false
allow_reboot = false
```

---

## Startup Integration

### User Systemd Service

Install and enable Nido to automatically start when your desktop session begins:

```bash
python3 scripts/install_autostart.py --type systemd

systemctl --user enable nido.service
systemctl --user start nido.service
```

### KDE Desktop Autostart

Alternatively, install the `.desktop` autostart launcher:

```bash
python3 scripts/install_autostart.py --type desktop
```

---

## Privacy and Offline Guarantee

* **Zero Cloud Network Calls**: Once model assets are downloaded during setup, Nido functions entirely offline with internet disconnected.
* **Telemetry Disabled**:
  * `NEEDLE_TELEMETRY=0` is hard-enforced in code.
  * `DO_NOT_TRACK=1` is exported in all processes.
* **Audio Ephemerality**: Microphone input is captured to an in-memory buffer during the F9 keypress. Audio data is discarded immediately after inference and is never written to disk or logged.
* **Structured Logs**: Logs are kept locally in `~/.local/state/nido/nido.log` according to XDG specifications.

---

## Project Structure

```text
nido/
├── pyproject.toml              # Packaging and dependency declarations
├── README.md                   # Project documentation
├── LICENSE                     # MIT License
├── .gitignore
├── config/
│   └── config.example.toml     # Reference configuration template
├── scripts/
│   ├── setup_models.py         # Automated model downloader
│   ├── convert_translation_model.py # Marian -> CTranslate2 INT8 converter
│   └── install_autostart.py    # Systemd / desktop autostart installer
├── systemd/
│   └── nido.service            # Systemd user service unit
├── desktop/
│   └── nido.desktop            # XDG desktop autostart entry
├── src/
│   └── nido/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py              # CLI entry point and subcommands
│       ├── config.py           # TOML configuration loader & dataclasses
│       ├── logging.py          # Structured XDG logger
│       ├── events.py           # Pipeline event bus & stages
│       ├── pipeline.py         # Audio -> STT -> Trans -> Needle -> Execution
│       ├── audio/
│       │   ├── __init__.py
│       │   └── recorder.py     # Push-to-talk in-memory 16kHz recorder
│       ├── hotkey/
│       │   ├── __init__.py
│       │   ├── base.py         # HotkeyBackend Protocol
│       │   └── evdev_backend.py# Linux evdev F9 press/release backend
│       ├── stt/
│       │   ├── __init__.py
│       │   └── shenava.py      # Shenava Koochik sherpa-onnx + Persian ITN
│       ├── translation/
│       │   ├── __init__.py
│       │   └── marian_ct2.py   # Marian CTranslate2 INT8 + SentencePiece
│       ├── needle/
│       │   ├── __init__.py
│       │   └── agent.py        # Needle 3 router and tool planner
│       ├── tools/
│       │   ├── __init__.py     # Registry builder
│       │   ├── registry.py     # ToolRegistry and safe executor
│       │   ├── apps.py         # Whitelisted app launcher/terminator
│       │   ├── browser.py      # URL navigation and web search
│       │   ├── audio.py        # Volume control (pactl/amixer/pamixer)
│       │   ├── media.py        # MPRIS media playback control
│       │   ├── files.py        # Safe file / folder opening
│       │   └── system.py       # Screenshot, system info, session lock
│       ├── models/
│       │   ├── __init__.py
│       │   └── manager.py      # Download, status, and verification
│       └── ui/
│           ├── __init__.py
│           ├── overlay.py      # PySide6 frameless KDE overlay
│           └── worker.py       # Non-blocking QThread pipeline worker
└── tests/
    ├── run_tests.py            # Self-contained test runner
    ├── test_config.py          # Configuration loading tests
    ├── test_tools.py           # Tool whitelist & safety tests
    ├── test_translation.py     # Translation & entity preservation tests
    ├── test_stt.py             # Persian ITN and number tests
    ├── test_audio.py           # Audio resampling & recorder tests
    ├── test_hotkey.py          # Hotkey press/release/repeat tests
    ├── test_pipeline.py        # End-to-end pipeline orchestration tests
    └── test_cli.py             # CLI command tests
```

---

## Testing

Run the test suite:

```bash
# Run using the built-in self-check test runner:
python3 tests/run_tests.py

# Or run with pytest:
pytest tests/ -v
```

---

## Troubleshooting

### 1. `PermissionError: [Errno 13] Permission denied: '/dev/input/event*'`
Your user account does not have read access to `/dev/input/` devices. Run:
```bash
sudo usermod -aG input $USER
```
Then log out and log back into your desktop session.

### 2. No audio captured or microphone silent
* Check default input device in KDE System Settings -> Audio -> Recording Devices.
* Verify input device in terminal:
  ```bash
  python3 -c "import sounddevice as sd; print(sd.query_devices())"
  ```
* Set explicit device index in `~/.config/nido/config.toml` under `[audio] device`.

### 3. Models missing error on startup
Run the automated model setup:
```bash
nido models setup
```

---

## Attribution & Credits

* **Shenava Koochik v1.0**: [Reza2kn](https://huggingface.co/Reza2kn) ([Shenava Koochik sherpa-onnx](https://huggingface.co/Reza2kn/Shenava-Koochik-v1.0-sherpa-onnx)).
* **sherpa-onnx**: Next-gen Kaldi project by Daniel Povey & contributors.
* **HPLT Persian-English Translation**: [High Performance Language Technologies (HPLT)](https://hplt-project.org/).
* **CTranslate2**: OpenNMT project fast inference engine.
* **Needle 3**: [Cactus Compute](https://github.com/Cactus-Compute/needle3).

---

## License

Nido is licensed under the [MIT License](LICENSE).
