# Voice Assistant Project Context

This document tracks the entire state, changes, and future directions for the local macOS Voice Assistant project.

## 📌 Project Overview
A fully local, privacy-focused voice assistant built for macOS (Apple Silicon). It runs silently in the background, listening for global hotkeys, and directly pastes results into any active text field. 
- **Speech-to-Text (STT):** `whisper.cpp` via `pywhispercpp` with Metal GPU acceleration.
- **Language Model (LLM):** `ollama` running `qwen2.5:1.5b`.
- **Core Library:** `sounddevice` (audio), `pynput` (hotkeys), `pyautogui` (paste simulation).

---

## 🛠️ Features & Implementation Details

The script currently features two distinct modes of operation, determined by the hotkey pressed:

1. **Dictation Mode (`Cmd+Shift+.`)**: 
   - Transcribes spoken audio and asks the LLM to clean up grammar, remove filler words, and add punctuation.
   - Designed to work like Wispr Flow, pasting clean, human-readable text directly into the cursor field.
   
2. **Answer Mode (`Cmd+Shift+,`)**:
   - Transcribes spoken audio, treats it as a question or command, and asks the LLM to directly answer it.
   - Pastes the LLM's answer into the active text field.

### Key Technical Solutions implemented:
- **macOS Key Normalization**: Overcame a macOS specific bug where holding `Cmd` suppresses the `char` attribute of the spacebar and other keys inside `pynput`.
- **Whisper Hallucination Filtering**: Automatically filters out empty transcription artifacts (e.g., `[BLANK_AUDIO]`, `[music]`, `thank you.`) caused by background noise.
- **Paste Reliability Delay**: Added a small buffer delay (`0.1s`) between writing to the clipboard and simulating `Cmd+V`, resolving intermittent auto-paste failures.
- **Clean Concurrency**: Uses `threading.Event` rather than polling or bare `time.sleep()`, ensuring the background listener consumes virtually no idle CPU.

---

## 🚀 How to Run

1. **Start Ollama**
   Ensure Ollama is running in the background and the model is pulled:
   ```bash
   ollama serve &
   ollama pull qwen2.5:1.5b
   ```

2. **Activate the Environment**
   ```bash
   source .venv/bin/activate
   ```

3. **Run the script**
   ```bash
   python voice_assistant.py
   ```

*(See `README.md` for launchd setup if you want it to run automatically on macOS login).*

---

## 💡 Future Feature Suggestions

Here are the advanced features we have brainstormed to take this tool to the next level:

### 1. Context-Aware Answering (Screen Context)
Use macOS AppleScript/Accessibility APIs to grab the currently highlighted text or the contents of the active window when the Answer hotkey is pressed.
* **Use Case:** Highlight a stack trace in VS Code, hold the hotkey, and ask *"Why is this crashing?"* The LLM receives the highlighted text alongside your voice query.

### 2. Streaming Output (Typewriter Effect)
Instead of waiting 2-5 seconds for the entire LLM response to generate before pasting it all at once, we can stream the response token-by-token using `pyautogui.typewrite`. 
* **Use Case:** Makes the assistant feel instantaneous, as you can watch the LLM "type" the answer live into your document.

### 3. Voice Feedback (Text-to-Speech)
In Answer mode, use the built-in macOS `say` command (or a local TTS model like Kokoro/Piper) to read the LLM's answer back to you aloud.
* **Use Case:** True hands-free assistance when you don't necessarily need the text pasted, but just want an answer to a question.

### 4. Smart Formatting Templates
Introduce lightweight command routing. If the user starts their dictation with specific trigger phrases, change the system prompt on the fly.
* **Use Case:** Say *"Write an email..."* to trigger an email-formatting prompt, or *"Write a script..."* to ensure the output is properly enclosed in markdown code blocks.

### 5. Multi-Model Architecture
Currently, both modes use `qwen2.5:1.5b`. We can optimize speed and quality by assigning different models to different tasks.
* **Use Case:** Use an ultra-fast model like `qwen2.5:0.5b` for Dictation mode (which only requires simple grammar checking), and a smarter, heavier model like `llama3.2:3b` for Answer mode (which requires reasoning and knowledge).

### 6. Clipboard History / Undo Buffer
Because the assistant forcefully overwrites the clipboard and pastes, it can sometimes overwrite important user text. 
* **Use Case:** Save the user's previous clipboard state before the assistant runs, and implement a hotkey to instantly "undo" the assistant's paste and restore the old clipboard.
