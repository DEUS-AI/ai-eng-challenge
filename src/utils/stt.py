import os
import threading

import whisper

from config.logger import get_logger
from utils.get_prompts import get_prompt

logger = get_logger(__name__)

_MODEL_NAME = os.environ.get("STT_MODEL", "small")
_model = None
_model_lock = threading.Lock()

_INITIAL_PROMPT = get_prompt("STT_INITIAL_PROMPT")


def _get_model() -> whisper.Whisper:
    """Lazy-load the Whisper model (downloaded on first use)."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                logger.info("Loading Whisper model '%s'", _MODEL_NAME)
                _model = whisper.load_model(_MODEL_NAME)
                logger.info("Whisper model '%s' loaded", _MODEL_NAME)
    return _model


def transcribe(audio_path: str) -> str:
    """Transcribe an audio file to text using Whisper.

    Args:
        audio_path: Path to a WAV, MP3, M4A, OGG, or FLAC file.

    Returns:
        Transcribed text string (stripped).
    """
    logger.debug("Transcribing: %s", audio_path)
    try:
        model = _get_model()
        result = model.transcribe(audio_path, fp16=False, initial_prompt=_INITIAL_PROMPT)
        text = result["text"].strip()
    except Exception:
        logger.exception("Transcription failed for: %s", audio_path)
        raise
    logger.info("Transcription result: %r", text[:80])
    return text
