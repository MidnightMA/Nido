# Nido (نیدو)

> **Fully local, Persian-speaking, multi-step Linux desktop controller for KDE Plasma 6.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![Desktop: KDE Plasma 6](https://img.shields.io/badge/Desktop-KDE%20Plasma%206%20%2F%20Linux-blueviolet.svg)](https://kde.org/plasma-desktop/)
[![Privacy: 100% Offline](https://img.shields.io/badge/Privacy-100%25%20Offline-success.svg)](#privacy-and-offline-guarantee)

---

## What is Nido?

**Nido** is an offline, push-to-talk voice assistant and iterative desktop controller engineered specifically for **KDE Plasma 6** on Linux (Kubuntu, Ubuntu, Debian, Arch) on Wayland and X11.

With Nido, you hold down **F9**, speak your command in Persian (فارسی), and release the key.

Nido transcribes the Persian utterance locally using **Shenava STT**, passes the original Persian transcript directly into **Laya-MLX** (`aac6fef/laya-multilingual-mlx`), and enters a bounded multi-step agent loop (up to 24 steps) observing the desktop via **Linux AT-SPI accessibility** and executing safe, structured actions until completion.

**No Persian-to-English translation. No generative text LLMs. No cloud APIs. No OCR. Zero audio leaves your machine.**

---

## Primary Pipeline Architecture

```text
                    USER
                     │
                     ▼
               Persian Voice
                     │
                     ▼
                 Shenava STT
                     │
                     ▼
            Original Persian Text
                     │
                     ▼
              Unified DesktopAgent
                     │
              ┌──────┴──────┐
              │             │
              ▼             ▼
        Desktop State    Static Context
          via AT-SPI     / existing tools
              │             │
              └──────┬──────┘
                     ▼
              Candidate Builder
                     │
                     ▼
               Laya-MLX choice
                     │
                     ▼
              Selected Action
                     │
             ┌───────┴────────┐
             │                │
             ▼                ▼
       Static Nido Tool    AT-SPI action
             │                │
             └───────┬────────┘
                     ▼
                Linux Desktop
                     │
                     ▼
                New State
                     │
                     ▼
                Snapshot Diff
                     │
                     └──────────► Laya-MLX
```

---

## Key Features

* **True Push-to-Talk (F9)**: Microphone records strictly while F9 is held down via `evdev`. Release triggers instant local processing.
* **100% Local & Offline Inference**:
  * **Persian STT**: [Shenava Koochik v1.0](https://huggingface.co/Reza2kn/Shenava-Koochik-v1.0-sherpa-onnx) via `sherpa-onnx` (16 kHz float32 mono).
  * **Decision Model**: [Laya Multilingual MLX](https://huggingface.co/aac6fef/laya-multilingual-mlx) based on mmBERT-base (1024 token limit).
  * **Linux MLX Runtime**: Supports Linux CPU backend (`mlx[cpu]`) and NVIDIA GPU backend (`mlx[cuda]`) with automatic device probe.
* **Typed Decision Model (Not an LLM)**:
  * Laya produces structured choice probabilities without generating text tokens or hallucinations.
  * Every decision selects an opaque candidate ID (`A1`, `A2`, ...) that Nido resolves safely.
* **Preserves Exact Literal Text**:
  * When writing or typing (e.g. *"نوت را باز کن و بنویس سلام دنیا"*), Nido preserves the exact literal Persian text payload without translation.
* **Unified Multi-Step Agent Loop**:
  * Every command enters the same iterative controller (default `max_steps = 24`).
  * Simple requests complete after one useful action plus `DONE`. Complex GUI tasks execute step-by-step.
* **Accessibility-First & OCR-Free**:
  * Perceives the desktop exclusively via Linux AT-SPI2 accessibility trees.
  * No OCR, no computer vision, and no screen capture for UI understanding.
* **Stale Target Protection**:
  * If the UI changes between observation and execution, stale element references are rejected safely without misclicking.
* **PySide6 Desktop Overlay**:
  * Shows real-time Persian goal, active application, step counter (`X/24`), current action, and execution status.
* **Safe Static System Tools**:
  * Whitelisted applications, browser URLs, volume control, media playback, screenshots, screen lock.

---

## Model Stack

| Component | Checkpoint / Repo | Runtime Engine | Format | Local Storage Path |
| :--- | :--- | :--- | :--- | :--- |
| **Persian STT** | `Reza2kn/Shenava-Koochik-v1.0-sherpa-onnx` | `sherpa-onnx` | ONNX float32 | `~/.local/share/nido/models/shenava/` |
| **Decision Model** | `aac6fef/laya-multilingual-mlx` | `MLX` | MLX Multilingual Checkpoint | `~/.local/share/nido/models/laya-multilingual-mlx/` |

---

## Installation

### 1. Prerequisites (Kubuntu / Ubuntu / Debian)

```bash
sudo apt update
sudo apt install python3-pip python3-venv libasound2-dev pulseaudio-utils playerctl spectacle \
    python3-gi gir1.2-atspi-2.0 at-spi2-core libatk-adaptor wtype libcairo2-dev
```

### 2. Configure evdev Permissions (Push-to-Talk)

```bash
sudo usermod -aG input $USER
```
*(Log out and log back in for group membership to take effect)*

### 3. Clone and Setup Environment

```bash
git clone https://github.com/MidnightMA/Nido.git
cd Nido

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"
```

For Linux MLX backends:
```bash
# For Linux CPU systems
pip install "mlx[cpu]"

# Or for NVIDIA/CUDA systems
pip install "mlx[cuda]"
```

---

## Model Setup & Verification

Nido downloads models once during setup. After setup, runtime operates completely offline with zero network access:

```bash
# 1. Download Shenava STT and Laya Multilingual MLX checkpoints
nido models setup

# 2. Check local model status and device
nido models status

# 3. Verify local inference offline
nido models verify
```

`nido models status` reports:
```text
Nido Offline Models Status:
--------------------------------------------------
[✓] Shenava Koochik STT (sherpa-onnx)
    Directory: ~/.local/share/nido/models/shenava
    Status:    Ready (model.onnx, tokens.txt)

[✓] Laya Multilingual MLX Decision Model
    Checkpoint:    aac6fef/laya-multilingual-mlx
    Directory:     ~/.local/share/nido/models/laya-multilingual-mlx
    Present:       yes
    Size:          438.2 MB
    Runtime:       MLX
    Device:        CPU
    Offline-Ready: yes
--------------------------------------------------
```

---

## Running Nido

```bash
# Start Nido push-to-talk daemon
nido

# Start with debug logging
nido --debug

# Start with custom config file
nido --config ~/.config/nido/config.toml
```

### Push-to-Talk Usage:
1. Press and hold **F9**.
2. Speak your command in Persian (e.g., *"نوت را باز کن و بنویس سلام دنیا"*).
3. Release **F9**.
4. The PySide6 overlay displays live progress through the multi-step agent loop until completion.

---

## Standalone Diagnostic Commands

```bash
# 1. Check desktop accessibility status and active window
nido accessibility status
nido accessibility tree
nido accessibility inspect

# 2. Test Laya action candidate selection (mock desktop state)
nido test-laya "نوت را باز کن و بنویس سلام دنیا"
nido test-laya "صدا رو بذار روی پنجاه درصد"

# 3. Test Desktop Multi-Step Agent loop live on desktop
nido test-desktop "کیت را باز کن"
nido test-desktop "کیت را باز کن و بنویس سلام دنیا"
nido test-desktop "Dolphin را باز کن و Downloads را باز کن"

# 4. Test Persian STT on a WAV file
nido test-stt sample.wav

# 5. List all registered safe tools
nido tools list
```

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
device = ""
sample_rate = 16000
channels = 1
max_seconds = 30

[ui]
position = "top-right"
timeout_ms = 4000
compact = false

[stt]
model_dir = "~/.local/share/nido/models/shenava"
threads = 4

[laya]
enabled = true
model_dir = "~/.local/share/nido/models/laya-multilingual-mlx"
dtype = "float16"
device = "auto"  # "auto", "cpu", "gpu"
batch_size = 16
compile = true
cache_prompts = true
pad_to_multiple = 16
max_elements = 120
max_candidates = 24
shortlist_size = 20

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

## License

MIT License. See [LICENSE](LICENSE) for details.
