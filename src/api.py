import json
import os
import secrets
import tempfile
from datetime import datetime
from typing import Annotated
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel

from config.logger import get_logger
from config.models import MessageRequest, SessionResponse
from graph import compile_graph
from utils.stt import transcribe
from utils.tts import save_speech_to_file

logger = get_logger(__name__)

load_dotenv()

# ── App & shared graph ────────────────────────────────────────────────────────

app = FastAPI(title="DEUS Bank Customer Support API", version="1.0.0")

_raw_origins = os.environ.get("ALLOWED_ORIGINS", "*")
_allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
    expose_headers=["X-Session-Done", "X-Transcript", "X-Messages"],
)

_graph = compile_graph()
_session_shown: dict[str, int] = {}  # tracks messages already delivered per session
_session_log_path: dict[str, str] = {}  # log file path per session, set at session creation


# ── Auth ──────────────────────────────────────────────────────────────────────

_API_KEY = os.environ.get("API_KEY", "")


def _require_api_key(x_api_key: Annotated[str, Header()]) -> None:
    """Validate API key using constant-time comparison to prevent timing attacks."""
    if not _API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API_KEY not configured on the server.",
        )
    if not secrets.compare_digest(x_api_key.encode(), _API_KEY.encode()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _thread_config(session_id: str) -> dict:
    """Build the LangGraph thread config dict for the given session."""
    return {"configurable": {"thread_id": session_id}}


def _pop_new_messages(session_id: str) -> list[str]:
    """Return messages added since the last call for this session."""
    all_messages = _graph.get_state(_thread_config(session_id)).values.get("agent_messages", [])
    shown = _session_shown.get(session_id, 0)
    new = all_messages[shown:]
    _session_shown[session_id] = len(all_messages)
    return new


def _is_done(session_id: str) -> bool:
    """Return True when the graph has no pending nodes (conversation is fully closed)."""
    return not bool(_graph.get_state(_thread_config(session_id)).next)


def _save_log(session_id: str) -> None:
    """Persist the session log_lines to the session's log file (overwriting on each call).

    The log path is fixed at session creation so partial logs are preserved
    even if the client disconnects before the conversation ends.
    """
    log_lines = _graph.get_state(_thread_config(session_id)).values.get("log_lines", [])
    log_path = _session_log_path.get(session_id)
    if not log_path:
        return
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/health", status_code=status.HTTP_200_OK)
def health() -> dict:
    return {"status": "ok"}


@app.post(
    "/sessions",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(_require_api_key)],
)
def start_session() -> SessionResponse:
    """Start a new chat session. Returns the session ID and the first agent greeting."""
    session_id = str(uuid4())
    _session_shown[session_id] = 0

    now = datetime.now()
    log_dir = os.path.join(os.path.dirname(__file__), "../logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"chat_{now.strftime('%Y%m%d_%H%M%S')}_{session_id[:8]}.md")
    _session_log_path[session_id] = log_path

    initial_state = {
        "user_details": "",
        "secret_question": "",
        "matched_nif": "",
        "secret_answer": "",
        "identity_verified": False,
        "account_type": "",
        "user_request": "",
        "details_retry_count": 0,
        "agent_messages": [],
        "log_lines": [f"# Chat Session — {now.strftime('%Y-%m-%d %H:%M:%S')}\n"],
    }
    try:
        _graph.invoke(initial_state, config=_thread_config(session_id))
    except Exception:
        logger.exception("[%s] Failed to initialise session graph", session_id[:8])
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to start session.")

    logger.info("Session created: %s", session_id[:8])
    return SessionResponse(
        session_id=session_id,
        messages=_pop_new_messages(session_id),
        done=_is_done(session_id),
    )


@app.post(
    "/sessions/{session_id}/message",
    response_model=SessionResponse,
    dependencies=[Depends(_require_api_key)],
)
def send_message(session_id: str, body: MessageRequest) -> SessionResponse:
    """Send user input to an active session and receive the agent's reply."""
    if session_id not in _session_shown:
        logger.warning("send_message: session not found [%s]", session_id[:8])
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")

    if _is_done(session_id):
        logger.warning("send_message: session already closed [%s]", session_id[:8])
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="Session is already closed. Start a new session.",
        )

    logger.info("[%s] text message received (%d chars)", session_id[:8], len(body.message))
    try:
        _graph.invoke(Command(resume=body.message), config=_thread_config(session_id))
    except Exception:
        logger.exception("[%s] graph.invoke failed", session_id[:8])
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal error processing message.")

    done = _is_done(session_id)
    _save_log(session_id)
    if done:
        logger.info("[%s] session completed via text", session_id[:8])
    return SessionResponse(
        session_id=session_id,
        messages=_pop_new_messages(session_id),
        done=done,
    )


@app.post(
    "/sessions/{session_id}/message/audio",
    dependencies=[Depends(_require_api_key)],
    responses={200: {"content": {"audio/wav": {}}}},
)
def send_message_get_audio_reply(session_id: str, body: MessageRequest) -> StreamingResponse:
    """Send user input to an active session and receive the agent's reply as a WAV audio file.

    Response headers:
      X-Session-Done: 'true' when the conversation has ended, 'false' otherwise.
      X-Messages: JSON-encoded list of agent reply text (for display alongside audio).
    """
    if session_id not in _session_shown:
        logger.warning("send_message_audio: session not found [%s]", session_id[:8])
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")

    if _is_done(session_id):
        logger.warning("send_message_audio: session already closed [%s]", session_id[:8])
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="Session is already closed. Start a new session.",
        )

    logger.info("[%s] text→audio message received (%d chars)", session_id[:8], len(body.message))
    try:
        _graph.invoke(Command(resume=body.message), config=_thread_config(session_id))
    except Exception:
        logger.exception("[%s] graph.invoke failed (audio)", session_id[:8])
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal error processing message.")

    new_messages = _pop_new_messages(session_id)
    done = _is_done(session_id)
    _save_log(session_id)
    if done:
        logger.info("[%s] session completed via text→audio", session_id[:8])

    # Synthesise all new messages into a single WAV file
    combined_text = " ".join(new_messages)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        save_speech_to_file(combined_text, tmp_path)
        with open(tmp_path, "rb") as f:
            audio_bytes = f.read()
    finally:
        os.unlink(tmp_path)

    return StreamingResponse(
        iter([audio_bytes]),
        media_type="audio/wav",
        headers={
            "X-Session-Done": str(done).lower(),
            "X-Messages": json.dumps(new_messages),
        },
    )


@app.post(
    "/sessions/{session_id}/voice",
    dependencies=[Depends(_require_api_key)],
    responses={200: {"content": {"audio/wav": {}}}},
)
def voice_round_trip(
    session_id: str,
    audio: UploadFile = File(..., description="Audio file (WAV, MP3, M4A, OGG, FLAC)"),
) -> StreamingResponse:
    """Full voice round-trip: upload audio → STT transcription → agent → TTS reply (WAV).

    Accepts any audio format supported by Whisper (WAV, MP3, M4A, OGG, FLAC).

    Response headers:
      X-Session-Done:   'true' when the conversation has ended, 'false' otherwise.
      X-Transcript:     The text Whisper transcribed from the uploaded audio.
      X-Messages:       JSON-encoded list of agent reply text.
    """
    if session_id not in _session_shown:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")

    if _is_done(session_id):
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="Session is already closed. Start a new session.",
        )

    logger.info("[%s] voice request received (file: %s)", session_id[:8], audio.filename)
    # Save uploaded audio to a temp file for Whisper
    suffix = "." + (audio.filename.rsplit(".", 1)[-1] if audio.filename and "." in audio.filename else "wav")
    tmp_in = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp_in.write(audio.file.read())
        tmp_in.close()
        user_text = transcribe(tmp_in.name)
    except Exception:
        logger.exception("[%s] STT transcription failed", session_id[:8])
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Transcription failed.")
    finally:
        os.unlink(tmp_in.name)

    if not user_text:
        logger.warning("[%s] empty transcription result", session_id[:8])
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Could not transcribe audio. Please speak clearly and try again.",
        )

    logger.info("[%s] transcribed: %r", session_id[:8], user_text[:80])
    try:
        _graph.invoke(Command(resume=user_text), config=_thread_config(session_id))
    except Exception:
        logger.exception("[%s] graph.invoke failed (voice)", session_id[:8])
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal error processing message.")

    new_messages = _pop_new_messages(session_id)
    done = _is_done(session_id)
    _save_log(session_id)
    if done:
        logger.info("[%s] session completed via voice", session_id[:8])

    # Synthesise agent reply to WAV
    combined_text = " ".join(new_messages)
    tmp_out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_out_path = tmp_out.name
    tmp_out.close()
    try:
        save_speech_to_file(combined_text, tmp_out_path)
        with open(tmp_out_path, "rb") as f:
            audio_bytes = f.read()
    finally:
        os.unlink(tmp_out_path)

    return StreamingResponse(
        iter([audio_bytes]),
        media_type="audio/wav",
        headers={
            "X-Session-Done": str(done).lower(),
            "X-Transcript": user_text,
            "X-Messages": json.dumps(new_messages),
        },
    )
