# Nido

> **Lightweight, fully offline English voice-controlled desktop assistant for KDE Plasma and Linux.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![Desktop: KDE Plasma 6](https://img.shields.io/badge/Desktop-KDE%20Plasma%206%20%2F%20Linux-blueviolet.svg)](https://kde.org/plasma-desktop/)
[![Privacy: 100% Offline](https://img.shields.io/badge/Privacy-100%25%20Offline-success.svg)](#privacy-and-offline-guarantee)

---

## What is Nido?

**Nido** is a lightweight, fully offline English voice assistant and iterative desktop controller engineered specifically for **Linux** and **KDE Plasma 6** on Wayland and X11.

Nido features a dual-mode global hotkey (**F9**):

1. **Push-to-Talk (Hold F9)**:
   - Hold **F9**, speak your command, and release the key.
   - Zero lost initial speech via an in-memory ring pre-buffer.
   - Nido finalizes transcription, sends it to Laya-MLX, and runs a bounded desktop task loop.

2. **Persistent Real-Time Voice Mode (Tap F9)**:
   - Single tap **F9** (< 300ms) to activate continuous real-time listening.
   - Microphone streams continuously in small PCM chunks.
   - Live partial hypotheses stream to the overlay.
   - Utterances are segmented via sherpa-onnx endpoint detection.
   - Finalized commands are enqueued into a serialized FIFO task queue and executed sequentially by the Laya desktop agent while the microphone continues listening for subsequent commands.
   - Single tap **F9** again to stop real-time mode.

**100% English-only. Local streaming Zipformer STT. Local Laya-MLX. No translation. No cloud APIs. No OCR. Zero audio leaves your machine.**

---

## Dual-Mode Interaction Architecture

```text
                                  F9 Hotkey
                                      │
                         ┌────────────┴────────────┐
                         ▼                         ▼
                    HOLD (>=300ms)             TAP (<300ms)
                     Push-To-Talk             Realtime Mode
                         │                         │
                         ▼                         ▼
                  Capture audio           Continuous Mic Capture
                  (with pre-buffer)                │
                         │                         ▼
                         ▼                  Streaming STT
                    Release F9      (Nemotron 0.6B Q8 GGUF)
                         │                         │
                         ▼                         ├───────────────┐
                   STT Finalize                    ▼               ▼
                         │                    Live Partial      Endpoint
                         ▼                     (to Overlay)    Final Utterance
                    Single Goal                                    │
                         │                                         ▼
                         │                                  FIFO Task Queue
                         │                                 (max 8 commands)
                         │                                         │
                         └─────────────────┬───────────────────────┘
                                           ▼
                                Serialized DesktopAgent
                                           │
                                  ┌────────┴────────┐
                                  ▼                 ▼
                            Desktop State     Static Tools
                             (via AT-SPI)
                                  │                 │
                                  └────────┬────────┘
                                           ▼
                                   Candidate Builder
                                           │
                                           ▼
                                    Laya-MLX Choice
                                           │
                                           ▼
                                   Execute & Settle
                                           │
                                  (up to 24 steps max)
                                           │
                                           ▼
                                          DONE
```

---

## Features

- **English-Only Streaming STT**: Uses NVIDIA Nemotron Speech Streaming EN 0.6B Q8 GGUF (`nemotron-speech-streaming-en-0.6b.q8_0.gguf`, ~700 MB) with `NeMo-Speech.cpp`. Optimized for low-latency real-time CPU streaming inference on Intel i5-7300U (AVX2, 4 threads). Avoids heavy 5.6 GB weights or PyTorch overhead entirely.
- **Dual-Mode F9 Hotkey**: Press-and-hold for push-to-talk; single press for continuous hands-free real-time listening.
- **Audio Pre-Buffer**: In-memory ring buffer (400ms) preserves the beginning of speech during press-vs-hold classification. Audio remains strictly in memory and is never written to disk.
- **Serialized Desktop Task Queue**: Finalized utterances are queued (up to 8 pending commands) and executed sequentially. Microphone capture and speech recognition never block while Laya is working.
- **Iterative 24-Step Agent Loop**: Each command is an independent goal executed through perception -> candidate generation -> Laya choice -> action -> observation (capped at 24 steps).
- **Exact Literal Text Preservation**: Commands like `"Open Notes and write Hello, world!"` preserve the exact string payload without alteration.
- **AT-SPI Semantic Perception**: Discovers UI elements directly via Linux accessibility; no screen capture, OCR, or vision model required.
- **Native Safe Static Tools**: Direct controls for volume, media playback, web URLs, searches, and screenshots.
- **PySide6 Desktop Overlay**: Non-intrusive HUD showing mode status, live partial transcript, executing task progress, and recent command history.

---

## Requirements

- **OS**: Linux (Kubuntu, Ubuntu, Debian, Arch, Fedora)
- **Desktop**: KDE Plasma 6 or 5 (Wayland or X11)
- **Python**: 3.10+
- **System Packages**:
  - `libpulse0` or `libasound2` (audio input)
  - `at-spi2-core` (accessibility)
  - User in `input` group for evdev global key monitoring (`sudo usermod -aG input $USER`)

---

## Installation

```bash
git clone https://github.com/mahdiahmadi87/Nido.git
cd Nido
pip install -e .
```

### Setup Models

Download the offline models explicitly (no runtime downloads during operation):

```bash
# Check status of models
nido models status

# Download Nemotron Speech Streaming (STT) and Laya-MLX
nido models setup

# Verify model inference offline
nido models verify
```

---

## Usage

### Run Daemon

```bash
nido
```

- **Hold F9**: Speak a command, release F9 to execute.
- **Tap F9**: Real-time mode starts. Speak commands naturally with pauses. Tap F9 again to stop.

### Diagnostic & Development Commands

```bash
# Test English Zipformer on a WAV audio file
nido test-stt sample.wav

# Test live streaming STT from microphone directly in terminal
nido test-stt-live

# Test realtime mode in terminal (mic -> STT -> queue -> desktop agent)
nido test-realtime

# Test Laya candidate decision on an English goal
nido test-laya "Open notes and write Hello world"

# Test multi-step desktop agent directly
nido test-desktop "Open dolphin and go to downloads"

# Inspect AT-SPI accessibility tree
nido accessibility tree
```

---

## Configuration

Configuration is stored in `~/.config/nido/config.toml`:

```toml
[assistant]
name = "Nido"
language = "en"

[hotkey]
key = "KEY_F9"
mode_threshold_ms = 300

[audio]
device = ""
sample_rate = 16000
prebuffer_ms = 400

[stt]
model_dir = "~/.local/share/nido/models/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17"
threads = 2
provider = "cpu"
decoding_method = "greedy_search"
enable_endpoint_detection = true
rule1_min_trailing_silence = 0.8
rule2_min_trailing_silence = 0.6
rule3_min_utterance_length = 20.0

[realtime]
enabled = true
max_pending_commands = 8
partial_update_interval_ms = 80
finalization_timeout_ms = 1500
show_transcript_history = true
transcript_history_size = 5
```

---

## Privacy and Offline Guarantee

Nido operates completely offline. Once models are installed:
- Zero network requests.
- Audio is processed entirely in memory.
- No disk writes for audio recordings.
- No remote telemetry or cloud APIs.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
