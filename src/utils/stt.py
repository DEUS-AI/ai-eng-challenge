"""
Speech-to-text using OpenAI Whisper (local, free, no API key required).

The model is loaded once at import time and reused across calls.
Default model: 'small' — better accuracy for names, numbers and banking terms.
Override with the STT_MODEL env var (tiny | base | small | medium | large).
"""
import os
import threading

import whisper

_MODEL_NAME = os.environ.get("STT_MODEL", "small")
_model = None
_model_lock = threading.Lock()

# Initial prompt gives Whisper vocabulary context so it transcribes banking
# terms, proper names, IBANs, phone numbers and NIF codes more accurately.
_INITIAL_PROMPT = (
    "This is a customer support call for DEUS Bank. "
    "The customer may provide their name, NIF tax number, phone number, "
    "IBAN, account number, or a secret answer. "
    "Numbers are spoken digit by digit or in groups. "
    "Example: 'My IBAN is PT50 0002 1234 5678 9012 3 and my NIF is 123456789.'"
)


def _get_model() -> whisper.Whisper:
    """Lazy-load the Whisper model (downloaded on first use)."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = whisper.load_model(_MODEL_NAME)
    return _model


def transcribe(audio_path: str) -> str:
    """Transcribe an audio file to text using Whisper.

    Args:
        audio_path: Path to a WAV, MP3, M4A, OGG, or FLAC file.

    Returns:
        Transcribed text string (stripped).
    """
    model = _get_model()
    result = model.transcribe(audio_path, fp16=False, initial_prompt=_INITIAL_PROMPT)
    return result["text"].strip()
