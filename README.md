#  Local Voice AI Receptionist (example use case- BrightSmile Clinic)

An end-to-end, ultra-low latency conversational Voice AI receptionist designed to run **100% locally** on Apple Silicon (M1/M2/M3) and modern CPUs with zero cloud API costs.

The agent handles real-time human interruptions (**barge-in**), checks clinic information, and triggers deterministic calendar appointment bookings via function calling.

---

#  Architecture & Tech Stack

<img width="833" height="580" alt="image" src="https://github.com/user-attachments/assets/d98df52c-e5ba-40a7-a942-658dd3714f45" />


```text
Microphone (16kHz)
│
▼
┌──────────────┐     Speech Detected      ┌───────────────────────────┐
│  Silero VAD  │ ───────────────────────► │ Faster-Whisper (base.en)  │
│     (v5)     │                          │     (int8 quantized)      │
└──────────────┘                          └───────────────────────────┘
│                                                 │
[Barge-in]                                         User Text
│                                                 ▼
│                                   ┌───────────────────────────┐
│                                   │    Ollama (Qwen2.5:3B)    │
│                                   │ └─ Deterministic Tools    │
│                                   └───────────────────────────┘
│                                                 │
▼                                            LLM Response
┌──────────────┐     Float32 PCM (24kHz)  ┌───────────────────────────┐
│ Audio Output │ ◄─────────────────────── │ Kokoro-82M (ONNX Engine)  │
│ (sounddevice)│                          └───────────────────────────┘
```

| Layer | Component | Runtime / Acceleration | Key Role |
|--------|------------|-----------------------|----------|
| **Audio I/O** | `sounddevice` (PortAudio) | Python 3.11 C-binding | 16kHz mic ingestion & non-blocking 24kHz playback |
| **VAD** | Silero VAD (v5) + RMS Gate | `onnxruntime` (CPU) | <1ms speech detection, endpointing & live barge-in |
| **STT** | Faster-Whisper (`base.en`) | `CTranslate2` (int8) | High-speed, streaming speech-to-text |
| **LLM** | Qwen2.5 (3B) | Ollama (`llama.cpp` / Apple Metal) | Natural reasoning & structured function calling |
| **TTS** | Kokoro-82M (v1.0) | `onnxruntime` (CPU) | Sub-200ms neural speech synthesis |

---

#  Getting Started Locally

## 1. Prerequisites

- **macOS** (Tested on Apple Silicon M1/M2/M3) or Linux
- **Python 3.10 or 3.11** (Avoid 3.12+ for C-extension binary stability)
- **Homebrew** installed

---

## 2. Clone the Repository

```bash
git clone https://github.com/Pranav080405/AI_receptionist-1.git
cd AI_receptionist-1
```

---

## 3. Set Up Python Environment

```bash
# Create and activate virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install sounddevice numpy onnxruntime faster-whisper ollama kokoro-onnx
```

> **Note for macOS Users**
>
> Ensure your Terminal or IDE (VS Code / iTerm) has microphone access enabled:
>
> `System Settings → Privacy & Security → Microphone`

---

## 4. Install & Launch Ollama

Ollama serves the local LLM reasoning engine using Apple Metal acceleration.

```bash
# Install Ollama
brew install ollama

# Start Ollama service
brew services start ollama

# Pull model weights (~1.9 GB)
ollama pull qwen2.5:3b
```

---

## 5. Download Model Weights

Create a `models/` directory and download the required ONNX models.

```bash
mkdir -p models

# Silero VAD v5 (~2 MB)
curl -L -o models/silero_vad.onnx \
https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx

# Kokoro-82M model (~310 MB)
curl -L -o models/kokoro-v1.0.onnx \
https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v1.0.onnx

# Voice embeddings (~29 MB)
curl -L -o models/voices-v1.0.bin \
https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices-v1.0.bin
```

---

## 6. Run the Voice Receptionist

For best results, use headphones to avoid acoustic feedback.

```bash
python main.py
```

---

# 🗣️ How to Test

Once the terminal prints:

```text
🤖 AI: Hello! Thanks for calling BrightSmile Clinic...
```

and then:

```text
🎧 Listening for your voice now...
```

try the following:

### 1. Clinic Hours & FAQ

> "Hi, what time are you open today?"

### 2. Deterministic Appointment Booking

> "Can I book an appointment for a cleaning at 2:00 PM?"

Watch for:

```text
🔧 Tool Triggered: book_appointment
```

### 3. Barge-In (Interruption)

Speak while the AI is talking.

The system will immediately stop playback:

```text
⚡ [Barge-In]
```

clear audio buffers and begin processing your new speech.

---

# ⚡ Key Features

- 🎙️ Real-time streaming voice conversations
- ⚡ Ultra-low latency local inference
- 🛑 Live barge-in interruption handling
- 🔒 100% offline operation (no cloud APIs)
- 🧠 Local LLM reasoning using Qwen2.5
- 📅 Deterministic appointment booking tools
- 🗣️ Natural neural speech synthesis
- 💰 Zero API costs after setup

---

# 📁 Project Structure

```text
AI_receptionist-1/
├── audio_io.py          # PortAudio duplex audio streaming & volume normalization
├── vad.py               # Silero VAD ONNX state machine + RMS energy gate
├── stt.py               # Faster-Whisper ASR inference wrapper
├── tts.py               # Kokoro-82M ONNX neural speech synthesizer
├── tools.py             # Deterministic clinic tool schemas & booking handlers
├── main.py              # Central async orchestration loop & barge-in logic
├── test_mic.py          # Audio device diagnostics and input RMS meter
├── models/              # Local model binaries (.onnx / .bin) [gitignored]
└── README.md
```

---
