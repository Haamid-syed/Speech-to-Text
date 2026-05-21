# Voice Assistant Project Context

This document tracks the entire state, changes, and future directions for the local macOS Voice Assistant project.

---

## 📌 Project Overview

A fully local, privacy-focused voice assistant built for macOS (Apple Silicon). It runs silently in the background, listening for global hotkeys, and directly pastes results into any active text field.

- **Speech-to-Text (STT):** `whisper.cpp` via `pywhispercpp` with Metal GPU acceleration.
- **Language Model (LLM):** `ollama` running `qwen2.5:1.5b`.
- **Core Libraries:** `sounddevice` (audio), `pynput` (hotkeys), `pyperclip` (clipboard), `pyobjc-framework-Quartz` (paste simulation).
- **Target Hardware:** macOS Apple Silicon (M1/M2/M3). Tested on M1 / 8GB RAM.

---

## 🛠️ Features & Implementation Details

### Two Modes of Operation

| Mode | Hotkey | Behavior |
|---|---|---|
| **Dictation** | `Cmd+Shift+0` | Transcribes speech → pastes raw Whisper text directly (no LLM) |
| **Answer** | `Cmd+Shift+9` | Transcribes speech → sends to LLM → pastes LLM's answer |

### Recording Behavior
- **Press-to-toggle**: Press hotkey once to start recording, press the same hotkey again to stop. No need to hold keys while speaking.
- **Hotkey debounce**: 500ms cooldown prevents rapid re-triggering of the same hotkey combo.
- **Minimum recording length**: 0.3s — shorter recordings are skipped.

### Pipeline Flow

```
Hotkey press → Start recording → Speak → Hotkey press again → Stop
    → Whisper transcribe → (Dictation: paste raw text) / (Answer: LLM → paste response)
```

### Key Technical Solutions

| Solution | Problem Solved |
|---|---|
| **Press-Once Toggle Recording** | Hotkeys toggle recording on/off — press once to start, press again to stop. No need to hold keys while speaking. |
| **Hotkey Debounce (500ms)** | Prevents rapid re-triggering of the same hotkey combo. |
| **macOS Key Normalization** | Overcomes macOS bug where holding `Cmd` suppresses the `char` attribute of keys inside `pynput`. Maps left/right modifier variants to generic forms. Reconstructs char from vk for printable keys. |
| **Quartz/CoreGraphics Paste** | Uses `CGEventCreateKeyboardEvent` + `CGEventPost(kCGHIDEventTap, ...)` to inject synthetic key events at the HID level. Most reliable paste method on macOS — doesn't steal focus, works in all apps. Falls back to AppleScript if Quartz unavailable. |
| **Modifier Wait Before Paste** | Polls `_pressed_keys` for modifier release before simulating paste. Prevents the "v" instead of paste bug caused by physical Cmd conflicting with synthetic Cmd. |
| **Clipboard Delay (0.1s)** | Buffer between `pyperclip.copy()` and paste simulation so the OS registers clipboard content. |
| **Whisper Hallucination Filtering** | Filters empty transcription artifacts (`[BLANK_AUDIO]`, `[music]`, `thank you.`, etc.) caused by background noise. |
| **Filler-Only Filtering** | Skips recordings that contain only filler words (um, uh, hmm, hm, ah). |
| **Clean Concurrency** | Uses `threading.Event` rather than polling or bare `time.sleep()`. Background listener consumes virtually no idle CPU. |
| **Pipeline Busy Guard** | Prevents starting a new recording while the previous pipeline is still processing. |

---

## 📁 Project Structure

```
STT/
├── voice_assistant.py    # Main script (567 lines)
├── requirements.txt      # Python dependencies
├── run.sh                # Shortcut: cd + activate venv + run
├── .venv/                # Python virtual environment
├── README.md             # User-facing documentation
└── CONTEXT.md            # This file — internal project context
```

---

## 🔧 Configuration

### Hotkeys

```python
HOTKEY_DICTATION  = {Key.cmd, Key.shift, KeyCode(char='0')}
HOTKEY_ANSWER     = {Key.cmd, Key.shift, KeyCode(char='9')}
```

**Hotkey evolution (why current combo was chosen):**
| Combo | Problem |
|---|---|
| `Cmd+Shift+.` / `Cmd+Shift+,` | Worked but user wanted different keys |
| `Option+/` / `Option+.` | User wanted Cmd-based |
| `Cmd+[` / `Cmd+]` | macOS system shortcuts — intercepted by Finder/browsers as "Go Back/Forward", paste failed |
| `Cmd+Shift+0` / `Cmd+Shift+9` | ✅ No system conflicts, works reliably across all apps |

### Constants

| Variable | Default | Description |
|---|---|---|
| `WHISPER_MODEL` | `small.en` | Whisper model size (~487 MB Metal GPU) |
| `SAMPLE_RATE` | `16000` | Audio sample rate (Hz) |
| `CHANNELS` | `1` | Mono audio |
| `MIN_RECORDING_S` | `0.3` | Minimum recording length before processing |
| `LLM_MODEL` | `qwen2.5:1.5b` | Ollama model name |
| `LLM_TIMEOUT_S` | `15` | Max seconds to wait for LLM response |
| `PASTE_DELAY_S` | `0.1` | Seconds between clipboard write and paste |
| `DEBUG_KEYS` | `False` | Log all keypresses for debugging |

### Prompts

**Dictation mode:** No LLM used. Raw Whisper transcription is pasted directly.

**Answer mode:**
```
You are a helpful assistant running locally on the user's computer.
Answer the user's question directly and concisely.
Do not include greetings, disclaimers, or unnecessary preamble.
If the question is simple, give a short answer.
If it requires explanation, be thorough but not verbose.
```

### Whisper Junk Filter

```python
WHISPER_JUNK = {
    "[BLANK_AUDIO]", "[silence]", "(silence)", "[inaudible]",
    "[music]", "(music)", "[applause]", "[laughter]",
    "you", "thank you.", "thanks.",
}
```

### Filler Words (skip silently)

```python
FILLER_ONLY = {"um", "uh", "hmm", "hm", "ah"}
```

---

## 🚀 How to Run

### Quick Launch
```bash
voice    # alias defined in ~/.zshrc
# or
./run.sh
```

### Manual Launch
```bash
ollama serve &
ollama pull qwen2.5:1.5b
source .venv/bin/activate
python voice_assistant.py
```

### macOS Permissions Required

| Permission | Why |
|---|---|
| **Microphone** | Recording audio |
| **Accessibility** | Simulating Cmd+V paste and reading global hotkeys |
| **Input Monitoring** | Detecting the hotkey combo |

> Your terminal app must appear in all three lists. Restart the terminal app after granting permissions.

---

## 💻 Memory & Performance (M1 / 8GB RAM)

### Measured Resource Usage

| Component | Physical RAM |
|---|---|
| Python (voice_assistant.py) | ~648 MB |
| ↳ Whisper `small.en` (Metal GPU) | ~487 MB |
| ↳ Python runtime + deps | ~161 MB |
| Ollama runner (qwen2.5:1.5b) | ~1090 MB |
| Ollama server | ~68 MB |
| **TOTAL PEAK** | **~1806 MB (1.76 GB)** |

### Performance Timings

| Step | Duration |
|---|---|
| Whisper load (cold) | 3-4 seconds |
| Whisper transcribe (3s audio) | ~2 seconds |
| LLM generation (short answer) | ~2-3 seconds |
| **Total pipeline** | **~5-7 seconds** |

### CPU Usage

| State | CPU |
|---|---|
| Idle (listening) | 0% |
| Whisper transcribing | 80-120% (M1 GPU) |
| LLM generating | 100-200% |

### Memory Budget on User's System

| | |
|---|---|
| Total RAM | 8 GB |
| System + apps (Brave, etc.) | ~4.5 GB |
| Voice assistant (peak) | ~1.8 GB |
| **Remaining free** | **~1.7 GB** |

> Comfortable for normal use. Tight with heavy multitasking (Docker, Xcode, 20+ browser tabs).

---

## 🐛 Known Issues & Resolutions

| Issue | Root Cause | Fix |
|---|---|---|
| **"v" pasted instead of Cmd+V** | Physical Cmd key still held when `pyautogui.hotkey()` fires, causing synthetic Cmd to conflict | Wait for all modifiers to release, then use explicit keyDown/keyUp with delays |
| **Paste goes to Terminal instead of target app** | `osascript` pastes to frontmost app at paste time (5-7s later), not at hotkey press time | Switched to Quartz/CoreGraphics HID-level event injection — no focus change needed |
| **`Cmd+[` / `Cmd+]` not working in other apps** | macOS system shortcuts intercepted by Finder/browsers as "Go Back/Forward" | Changed to `Cmd+Shift+0` / `Cmd+Shift+9` — no system conflicts |
| **Dictation mode was sending to LLM** | Pipeline routed both modes through LLM | Dictation now pastes raw Whisper transcription directly, bypassing LLM entirely |
| **macOS Cmd suppresses char attribute** | Holding Cmd causes pynput to receive `char=None` for printable keys | Key normalization reconstructs char from vk code for printable ASCII range |

---

## 💡 Future Feature Suggestions

### 1. Context-Aware Answering (Screen Context)
Use macOS AppleScript/Accessibility APIs to grab the currently highlighted text or the contents of the active window when the Answer hotkey is pressed.
* **Use Case:** Highlight a stack trace in VS Code, press the hotkey, and ask *"Why is this crashing?"* The LLM receives the highlighted text alongside your voice query.

### 2. Streaming Output (Typewriter Effect)
Instead of waiting 2-5 seconds for the entire LLM response to generate before pasting it all at once, stream the response token-by-token using Quartz key events.
* **Use Case:** Makes the assistant feel instantaneous, as you can watch the LLM "type" the answer live into your document.

### 3. Voice Feedback (Text-to-Speech)
In Answer mode, use the built-in macOS `say` command (or a local TTS model like Kokoro/Piper) to read the LLM's answer back to you aloud.
* **Use Case:** True hands-free assistance when you don't necessarily need the text pasted, but just want an answer to a question.

### 4. Smart Formatting Templates
Introduce lightweight command routing. If the user starts their dictation with specific trigger phrases, change the system prompt on the fly.
* **Use Case:** Say *"Write an email..."* to trigger an email-formatting prompt, or *"Write a script..."* to ensure the output is properly enclosed in markdown code blocks.

### 5. Clipboard History / Undo Buffer
Because the assistant forcefully overwrites the clipboard and pastes, it can sometimes overwrite important user text.
* **Use Case:** Save the user's previous clipboard state before the assistant runs, and implement a hotkey to instantly "undo" the assistant's paste and restore the old clipboard.

### 6. Smaller Whisper Model (`tiny.en`)
Switch from `small.en` (487 MB) to `tiny.en` (75 MB) to save ~300 MB RAM with minimal quality loss for clear dictation.
* **Use Case:** More breathing room on 8GB systems when running alongside heavy apps.
