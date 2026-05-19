#!/usr/bin/env python3
"""
Local Voice Assistant for macOS (Apple Silicon)
Two modes:
  Cmd+Shift+.  →  Dictation (transcribe → LLM refine → paste clean text)
  Cmd+Shift+,  →  Answer    (transcribe → LLM answer → paste response)
"""

import signal
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Optional, Union

import numpy as np
import sounddevice as sd
from pynput.keyboard import Key, KeyCode, Listener

try:
    import ollama as _ollama
except ImportError:
    _ollama = None  # type: ignore[assignment]

try:
    from pywhispercpp.model import Model as WhisperModel
except ImportError:
    WhisperModel = None  # type: ignore[assignment,misc]

import pyperclip
import pyautogui

# ── CONFIG ────────────────────────────────────────────────────────────────────

# Mode 1: Dictation — refines your speech into clean text (like Wispr Flow)
HOTKEY_DICTATION  = {Key.cmd, Key.shift, KeyCode(char='.')}

# Mode 2: Answer — sends your speech as a question, pastes the LLM's answer
HOTKEY_ANSWER     = {Key.cmd, Key.shift, KeyCode(char=',')}

WHISPER_MODEL     = "small.en"
SAMPLE_RATE       = 16000
CHANNELS          = 1
MIN_RECORDING_S   = 0.3
LLM_MODEL         = "qwen2.5:1.5b"
LLM_TIMEOUT_S     = 15
PASTE_DELAY_S     = 0.1   # seconds between clipboard write and Cmd+V

DICTATION_PROMPT  = (
    "You are a dictation assistant. The user spoke the following text aloud. "
    "Clean it up: fix grammar, remove filler words (um, uh, like, you know), "
    "fix punctuation, and make it read naturally. "
    "Output ONLY the cleaned text — no quotes, no explanation, no preamble."
)
ANSWER_PROMPT     = (
    "You are a helpful assistant running locally on the user's computer. "
    "Answer the user's question directly and concisely. "
    "Do not include greetings, disclaimers, or unnecessary preamble. "
    "If the question is simple, give a short answer. "
    "If it requires explanation, be thorough but not verbose."
)

FILLER_ONLY       = {"um", "uh", "hmm", "hm", "ah"}

# Whisper artifacts to filter out
WHISPER_JUNK      = {
    "[BLANK_AUDIO]", "[silence]", "(silence)", "[inaudible]",
    "[music]", "(music)", "[applause]", "[laughter]",
    "you", "thank you.", "thanks.",  # common whisper hallucinations on silence
}

# Set to True to log every keypress (for debugging hotkey issues)
DEBUG_KEYS        = False

# Virtual keycode for Space on macOS (used when Cmd swallows the char)
_VK_SPACE = 49

# ── GLOBALS (minimal mutable state) ──────────────────────────────────────────
_is_recording = threading.Event()
_pipeline_busy = threading.Event()
_audio_frames: list[np.ndarray] = []
_frames_lock = threading.Lock()
_pressed_keys: set = set()
_active_mode: Optional[str] = None   # "dictation" or "answer"
_mode_lock = threading.Lock()
_shutdown = threading.Event()


# ── LOGGING HELPERS ──────────────────────────────────────────────────────────

def log_status(msg: str) -> None:
    """Print a status message to terminal."""
    print(msg, flush=True)


def log_warn(msg: str) -> None:
    """Print a warning message."""
    print(f"⚠️  {msg}", flush=True)


def log_error(msg: str) -> None:
    """Print an error message."""
    print(f"❌  {msg}", flush=True)


def log_debug(msg: str) -> None:
    """Print a debug message (only when DEBUG_KEYS is True)."""
    if DEBUG_KEYS:
        print(f"🔍  {msg}", flush=True)


# ── KEY NORMALISATION (macOS quirks) ─────────────────────────────────────────

def _normalize_key(key: Union[Key, KeyCode, None]) -> Union[Key, KeyCode, None]:
    """Normalise a key so left/right modifiers and vk-only space compare equal."""
    if key is None:
        return None

    # Map left/right modifier variants → generic form
    if key in (Key.cmd_l, Key.cmd_r):
        return Key.cmd
    if key in (Key.shift_l, Key.shift_r):
        return Key.shift
    if key in (Key.alt_l, Key.alt_r):
        return Key.alt
    if key in (Key.ctrl_l, Key.ctrl_r):
        return Key.ctrl

    # On macOS, Cmd suppresses `char` — space arrives as KeyCode(vk=49, char=None)
    if isinstance(key, KeyCode):
        vk = getattr(key, "vk", None)
        char = getattr(key, "char", None)

        if char == ' ' or vk == _VK_SPACE:
            return KeyCode(char=' ')

        # When Cmd is held, char may be None but vk is valid.
        # Reconstruct a char-based KeyCode for printable keys.
        if char is None and vk is not None:
            try:
                reconstructed = chr(vk) if 32 <= vk <= 126 else None
            except (ValueError, OverflowError):
                reconstructed = None
            if reconstructed:
                return KeyCode(char=reconstructed.lower())

    return key


def _combo_held(combo: set) -> bool:
    """Return True when all keys in the given combo are currently held."""
    return _pressed_keys.issuperset(combo)


# ── AUDIO ────────────────────────────────────────────────────────────────────

def validate_microphone() -> None:
    """Check that a microphone is available and accessible. Exits on failure."""
    try:
        devices = sd.query_devices()
        input_found = any(
            d.get("max_input_channels", 0) > 0 for d in devices  # type: ignore[union-attr]
        )
        if not input_found:
            log_error(
                "No input microphone found.\n"
                "   → Plug in a mic or check System Settings > Sound > Input."
            )
            sys.exit(1)

        # Quick probe to test permissions
        sd.check_input_settings(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16")

    except PermissionError:
        log_error(
            "Microphone permission denied.\n"
            "   → Grant access in: System Settings > Privacy & Security > Microphone\n"
            "   → Add your terminal app (Terminal / iTerm / VS Code) to the list."
        )
        sys.exit(1)
    except sd.PortAudioError as e:
        log_error(f"Audio device error: {e}\n   → Check System Settings > Sound > Input.")
        sys.exit(1)


def audio_callback(indata: np.ndarray, frames: int, time_info: object, status: object) -> None:
    """Sounddevice stream callback — appends frames while recording."""
    if _is_recording.is_set():
        with _frames_lock:
            _audio_frames.append(indata.copy())


def start_audio_stream() -> sd.InputStream:
    """Open and return a persistent InputStream."""
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        callback=audio_callback,
    )
    stream.start()
    return stream


# ── STT (whisper.cpp) ────────────────────────────────────────────────────────

def load_whisper_model() -> "WhisperModel":
    """Load the pywhispercpp model once. Exits on failure."""
    if WhisperModel is None:
        log_error(
            "pywhispercpp is not installed.\n"
            '   → Install with: CMAKE_ARGS="-DWHISPER_METAL=ON" pip install pywhispercpp'
        )
        sys.exit(1)

    try:
        model = WhisperModel(WHISPER_MODEL)
        log_status(f"✅  Whisper model '{WHISPER_MODEL}' loaded")
        return model
    except Exception as e:
        log_error(
            f"Failed to load Whisper model '{WHISPER_MODEL}': {e}\n"
            '   → Ensure model is available. Try:\n'
            f'     pip install pywhispercpp  (model "{WHISPER_MODEL}" downloads automatically)\n'
            '   → For Metal acceleration on Apple Silicon:\n'
            '     CMAKE_ARGS="-DWHISPER_METAL=ON" pip install pywhispercpp'
        )
        sys.exit(1)


def transcribe(model: "WhisperModel", audio: np.ndarray) -> Optional[str]:
    """Run whisper.cpp on int16 audio, return cleaned text or None."""
    try:
        # pywhispercpp expects float32 in [-1, 1]
        audio_f32 = audio.astype(np.float32) / 32768.0

        segments = model.transcribe(audio_f32)
        text = " ".join(seg.text.strip() for seg in segments).strip()

        if not text:
            log_warn("Nothing detected — empty transcription.")
            return None

        # Filter whisper hallucination artifacts
        if text.strip() in WHISPER_JUNK:
            log_warn("Nothing detected — whisper artifact.")
            return None

        # Filter filler-only utterances
        words = set(text.lower().replace(".", "").replace(",", "").split())
        if words and words.issubset(FILLER_ONLY):
            return None  # Skip silently

        return text

    except Exception as e:
        log_error(f"Transcription error: {e}")
        traceback.print_exc()
        return None


# ── LLM (Ollama) ─────────────────────────────────────────────────────────────

def query_llm(prompt: str, system_prompt: str) -> Optional[str]:
    """Send prompt to Ollama with the given system prompt, return response or None."""
    if _ollama is None:
        log_error("ollama Python client not installed.\n   → pip install ollama")
        return None

    def _call() -> str:
        response = _ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        return response["message"]["content"].strip()

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_call)
            result = future.result(timeout=LLM_TIMEOUT_S)

        if not result:
            log_warn("LLM returned an empty response.")
            return None

        return result

    except FuturesTimeoutError:
        log_warn(f"LLM timed out after {LLM_TIMEOUT_S}s.")
        return None
    except ConnectionError:
        log_error("Ollama is not running.\n   → Run: ollama serve")
        return None
    except Exception as e:
        err_str = str(e).lower()
        # Detect connection refused (may surface as various exception types)
        if "connection" in err_str and ("refused" in err_str or "error" in err_str):
            log_error("Ollama is not running.\n   → Run: ollama serve")
            return None
        # Detect model not found
        if "404" in err_str or "not found" in err_str or "model" in err_str:
            log_error(f"Model not available.\n   → Run: ollama pull {LLM_MODEL}")
            return None
        log_error(f"LLM error: {e}")
        traceback.print_exc()
        return None


# ── PASTE ─────────────────────────────────────────────────────────────────────

def paste_text(text: str) -> None:
    """Copy text to clipboard, wait briefly, then auto-paste via Cmd+V."""
    try:
        pyperclip.copy(text)
    except Exception as e:
        log_warn(f"Clipboard write failed: {e}")
        log_status(f"📋  LLM response:\n{text}")
        return

    # Brief pause so the OS registers the clipboard content before paste
    _paste_ready = threading.Event()
    _paste_ready.wait(timeout=PASTE_DELAY_S)

    try:
        pyautogui.hotkey("command", "v")
    except Exception as e:
        log_warn(f"Copied to clipboard (auto-paste failed: {e})")


# ── PIPELINE ──────────────────────────────────────────────────────────────────

def run_pipeline(model: "WhisperModel", audio: np.ndarray, mode: str) -> None:
    """Orchestrate: transcribe → LLM → paste. Mode determines behaviour."""
    try:
        # 1. STT
        log_status("⚙️  Transcribing...")
        text = transcribe(model, audio)
        if text is None:
            return

        log_status(f'📝  "{text}"')

        # 2. LLM — route by mode
        if mode == "dictation":
            log_status("✏️  Refining dictation...")
            system = DICTATION_PROMPT
        else:
            log_status("🤖  Querying LLM...")
            system = ANSWER_PROMPT

        response = query_llm(text, system)
        if response is None:
            return

        log_status(f'💬  "{response}"')

        # 3. Paste
        paste_text(response)
        log_status("✅  Done")

    except Exception as e:
        log_error(f"Pipeline error: {e}")
        traceback.print_exc()
    finally:
        _pipeline_busy.clear()


# ── HOTKEY HANDLING ───────────────────────────────────────────────────────────

def on_key_press(key: Union[Key, KeyCode, None], model: "WhisperModel") -> None:
    """Handle key press — detect which mode's combo is held."""
    global _active_mode
    normalized = _normalize_key(key)
    if normalized is None:
        return

    log_debug(f"PRESS   raw={key!r}  norm={normalized!r}  held={_pressed_keys}")
    _pressed_keys.add(normalized)

    # Already recording or processing — ignore
    if _is_recording.is_set():
        return
    if _pipeline_busy.is_set():
        log_warn("Still processing...")
        return

    # Check which combo is active (dictation takes priority if both match)
    mode = None
    if _combo_held(HOTKEY_DICTATION):
        mode = "dictation"
    elif _combo_held(HOTKEY_ANSWER):
        mode = "answer"

    if mode is not None:
        with _mode_lock:
            _active_mode = mode
        with _frames_lock:
            _audio_frames.clear()
        _is_recording.set()
        label = "📝  Dictation" if mode == "dictation" else "💡  Answer"
        log_status(f"🎙  Recording... ({label} mode)")


def on_key_release(key: Union[Key, KeyCode, None], model: "WhisperModel") -> None:
    """Handle key release — stops recording and kicks off pipeline."""
    global _active_mode
    normalized = _normalize_key(key)
    if normalized is None:
        return

    log_debug(f"RELEASE raw={key!r}  norm={normalized!r}  held={_pressed_keys}")
    _pressed_keys.discard(normalized)

    if not _is_recording.is_set():
        return

    # Check if the active mode's combo is still fully held
    with _mode_lock:
        mode = _active_mode

    if mode == "dictation" and _combo_held(HOTKEY_DICTATION):
        return  # Still holding — keep recording
    if mode == "answer" and _combo_held(HOTKEY_ANSWER):
        return  # Still holding — keep recording

    # Combo broken → stop recording
    _is_recording.clear()
    log_status("✋  Stopped")

    with _frames_lock:
        frames = list(_audio_frames)
        _audio_frames.clear()

    # No frames captured
    if not frames:
        return

    audio = np.concatenate(frames, axis=0).flatten()
    duration = len(audio) / SAMPLE_RATE

    if duration < MIN_RECORDING_S:
        log_warn("Too short, skipping.")
        return

    _pipeline_busy.set()
    thread = threading.Thread(
        target=run_pipeline, args=(model, audio, mode or "answer"), daemon=True
    )
    thread.start()


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point — validates environment, loads model, starts listener."""
    log_status("─── Voice Assistant ───")

    # Validate mic before anything else
    validate_microphone()

    # Load whisper model once
    model = load_whisper_model()

    # Open persistent audio stream
    stream = start_audio_stream()
    log_status("🎧  Audio stream open")

    # Graceful shutdown
    def shutdown(signum: int, frame: object) -> None:
        _shutdown.set()
        _is_recording.clear()
        stream.stop()
        stream.close()
        log_status("\n👋  Goodbye")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Start keyboard listener
    listener = Listener(
        on_press=lambda key: on_key_press(key, model),
        on_release=lambda key: on_key_release(key, model),
    )
    listener.start()

    log_status("🚀  Ready!")
    log_status("    📝  Cmd+Shift+.  →  Dictation (refines speech → pastes clean text)")
    log_status("    💡  Cmd+Shift+,  →  Answer    (asks LLM → pastes response)")
    if DEBUG_KEYS:
        log_status("    ⚡  DEBUG_KEYS is ON — all keypresses will be logged")
    log_status("    Press Ctrl+C to quit\n")

    # Block main thread until shutdown
    _shutdown.wait()


if __name__ == "__main__":
    main()
