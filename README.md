# Voice Assistant — Local STT → LLM → Paste

A background voice assistant for macOS Apple Silicon. Two modes:

- **📝 Dictation** (`Cmd+Shift+.`) — speak → transcribe → LLM cleans up → paste refined text
- **💡 Answer** (`Cmd+Shift+,`) — speak a question → transcribe → LLM answers → paste response

**Everything runs locally. No internet required. No data leaves your machine.**

---

## Prerequisites

| Dependency | Purpose |
|---|---|
| Python 3.11+ | Runtime |
| [Homebrew](https://brew.sh) | Package manager |
| [Ollama](https://ollama.com) | Local LLM server |
| portaudio | Audio I/O (via Homebrew) |

---

## Install

### 1. System dependencies

```bash
brew install portaudio
brew install ollama
```

### 2. Pull the LLM model

```bash
ollama serve &        # start Ollama in background (or use the app)
ollama pull qwen2.5:1.5b
```

### 3. Python dependencies

```bash
cd /path/to/STT
python3 -m venv .venv
source .venv/bin/activate

# Install pywhispercpp with Metal GPU acceleration (Apple Silicon)
CMAKE_ARGS="-DWHISPER_METAL=ON" pip install pywhispercpp

# Install remaining dependencies
pip install -r requirements.txt
```

> **Note:** The first run downloads the `small.en` Whisper model automatically (~461 MB).

### 4. macOS Permissions

Go to **System Settings → Privacy & Security** and grant your terminal app:

| Permission | Why |
|---|---|
| **Microphone** | Recording audio |
| **Accessibility** | Simulating Cmd+V paste and reading global hotkeys |
| **Input Monitoring** | Detecting the hotkey combo |

> Your terminal app (Terminal.app, iTerm2, VS Code, etc.) must appear in all three lists.
> **Restart the terminal app** after granting permissions.

---

## Usage

```bash
source .venv/bin/activate
python voice_assistant.py
```

### Two Modes

| Hotkey | Mode | What it does |
|---|---|---|
| `Cmd+Shift+.` | **Dictation** | Cleans up your speech (fixes grammar, removes filler words) and pastes the refined text |
| `Cmd+Shift+,` | **Answer** | Sends your speech as a question to the LLM and pastes the answer |

1. **Hold** the hotkey → mic starts recording
2. **Speak** clearly
3. **Release** → transcription → LLM processing → auto-paste at cursor

Press `Ctrl+C` to quit.

---

## Changing the Hotkeys

Edit the `HOTKEY_DICTATION` and `HOTKEY_ANSWER` sets in the CONFIG block:

```python
# Default
HOTKEY_DICTATION = {Key.cmd, Key.shift, KeyCode(char='.')}
HOTKEY_ANSWER    = {Key.cmd, Key.shift, KeyCode(char=',')}

# Example: use Option instead of Shift
HOTKEY_DICTATION = {Key.cmd, Key.alt, KeyCode(char='.')}
HOTKEY_ANSWER    = {Key.cmd, Key.alt, KeyCode(char=',')}
```

> **Avoid** `Cmd+Shift+Space` — macOS intercepts it for Emoji & Symbols.

Available key names: [pynput docs](https://pynput.readthedocs.io/en/latest/keyboard.html#pynput.keyboard.Key).

---

## Auto-Start on Login (launchd)

### 1. Create the plist

Save to `~/Library/LaunchAgents/com.local.voiceassistant.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.local.voiceassistant</string>

    <key>ProgramArguments</key>
    <array>
        <string>/Users/YOUR_USERNAME/Documents/coding projects/STT/.venv/bin/python3</string>
        <string>/Users/YOUR_USERNAME/Documents/coding projects/STT/voice_assistant.py</string>
    </array>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>

    <key>StandardOutPath</key>
    <string>/tmp/voiceassistant.stdout.log</string>

    <key>StandardErrorPath</key>
    <string>/tmp/voiceassistant.stderr.log</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
    </dict>
</dict>
</plist>
```

### 2. Load / Manage

```bash
# Load (starts on login)
launchctl load ~/Library/LaunchAgents/com.local.voiceassistant.plist

# Check status
launchctl list | grep voiceassistant

# Unload
launchctl unload ~/Library/LaunchAgents/com.local.voiceassistant.plist

# View logs
tail -f /tmp/voiceassistant.stdout.log
```

---

## Troubleshooting

### 1. `❌ Microphone permission denied`
System Settings → Privacy & Security → Microphone → add your terminal app. **Restart the terminal.**

### 2. `❌ Ollama is not running`
```bash
ollama serve     # or open the Ollama desktop app
```

### 3. `❌ Model not available`
```bash
ollama pull qwen2.5:1.5b
```

### 4. `⚠️ LLM timed out after 15s`
- Close memory-heavy apps (Whisper + Ollama need ~4 GB combined)
- Increase `LLM_TIMEOUT_S` in the config
- Try a smaller model: `ollama pull qwen2.5:0.5b`

### 5. Hotkey not detected / paste not working
System Settings → Privacy & Security → **Accessibility** and **Input Monitoring** → add your terminal app. **Restart the terminal.**

---

## Config Reference

| Variable | Default | Description |
|---|---|---|
| `HOTKEY_DICTATION` | Cmd+Shift+. | Dictation mode hotkey |
| `HOTKEY_ANSWER` | Cmd+Shift+, | Answer mode hotkey |
| `WHISPER_MODEL` | `small.en` | Whisper model size |
| `SAMPLE_RATE` | `16000` | Audio sample rate (Hz) |
| `MIN_RECORDING_S` | `0.3` | Minimum recording length |
| `LLM_MODEL` | `qwen2.5:1.5b` | Ollama model name |
| `LLM_TIMEOUT_S` | `15` | Max seconds to wait for LLM |
| `PASTE_DELAY_S` | `0.1` | Delay before auto-paste |
| `DEBUG_KEYS` | `False` | Log all keypresses |

---

## License

MIT
