import threading
import pyttsx3

# English (US) voice — Samantha is the standard macOS en_US voice.
VOICE_ID = "com.apple.voice.compact.en-US.Samantha"

# Shared engine for speak() — reused across calls (say+runAndWait works fine).
_engine = pyttsx3.init()
_engine.setProperty("rate", 165)   # words-per-minute (default ~200)
_engine.setProperty("voice", VOICE_ID)
_lock = threading.Lock()


def speak(text: str) -> None:
    """Synthesise text to speech using the English (US) Samantha voice."""
    with _lock:
        _engine.say(text)
        _engine.runAndWait()


def save_speech_to_file(text: str, path: str) -> None:
    """Save TTS output to a WAV file.

    A fresh engine instance is created each time because pyttsx3's nsss driver
    on macOS only writes audio correctly on the first save_to_file call per instance.
    """
    with _lock:
        tmp_engine = pyttsx3.init()
        tmp_engine.setProperty("rate", 165)
        tmp_engine.setProperty("voice", VOICE_ID)
        tmp_engine.save_to_file(text, path)
        tmp_engine.runAndWait()
        tmp_engine.stop()
