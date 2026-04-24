import os
import platform
import subprocess
import tempfile
import threading
import pyttsx3

from config.logger import get_logger

logger = get_logger(__name__)

# macOS uses the nsss driver with Apple's Samantha voice.
# Linux (Docker) uses the espeak-ng driver; no macOS voice ID is available there.
_IS_MACOS = platform.system() == "Darwin"
VOICE_ID = "com.apple.voice.compact.en-US.Samantha" if _IS_MACOS else None

# Shared engine for speak() — reused across calls (say+runAndWait works fine).
_engine = pyttsx3.init()
_engine.setProperty("rate", 165)   # words-per-minute (default ~200)
if VOICE_ID:
    _engine.setProperty("voice", VOICE_ID)
_lock = threading.Lock()


def speak(text: str) -> None:
    """Synthesise text to speech synchronously using the platform's default TTS engine."""
    with _lock:
        _engine.say(text)
        _engine.runAndWait()


def save_speech_to_file(text: str, path: str) -> None:
    """Save TTS output to a standard PCM WAV file playable by browsers.

    On macOS, the native ``say`` command is used to generate AIFF output which
    is then transcoded to PCM WAV by ffmpeg. This avoids pyttsx3 singleton
    conflicts when multiple engine instances exist in the same process.
    On Linux, pyttsx3 with the espeak-ng driver writes WAV directly.
    """
    if _IS_MACOS:
        # Use macOS `say` directly — reliable, no pyttsx3 singleton issues.
        with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as tmp:
            aiff_path = tmp.name
        try:
            logger.debug("TTS synthesising %d chars via say", len(text))
            subprocess.run(
                ["say", "-v", "Samantha", "-r", "165", "-o", aiff_path, text],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["ffmpeg", "-y", "-i", aiff_path, path],
                check=True,
                capture_output=True,
            )
            logger.debug("TTS WAV written: %s", path)
        except subprocess.CalledProcessError:
            logger.exception("TTS subprocess failed (say/ffmpeg)")
            raise
        finally:
            if os.path.exists(aiff_path):
                os.unlink(aiff_path)
    else:
        logger.debug("TTS synthesising %d chars via espeak", len(text))
        with _lock:
            _engine.save_to_file(text, path)
            _engine.runAndWait()
