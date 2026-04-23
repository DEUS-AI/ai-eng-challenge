"""
DEUS Bank Customer Support — FastAPI interface.

Endpoints:
  GET  /health                        — liveness check (no auth)
  POST /sessions                      — start a new chat session
  POST /sessions/{session_id}/message — send user input, receive agent replies

Authentication:
  All session endpoints require the header:
    X-API-Key: <value of API_KEY env var>

CORS:
  Set ALLOWED_ORIGINS in .env as a comma-separated list of allowed origins.
  Defaults to * (all origins) if not set — restrict this in production.
  Example: ALLOWED_ORIGINS=https://my-chatbot.com,http://localhost:3000
"""
import os
import secrets
from datetime import datetime
from typing import Annotated
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from langgraph.types import Command
from pydantic import BaseModel

from graph import compile_graph

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
)

_graph = compile_graph()
_session_shown: dict[str, int] = {}  # tracks messages already delivered per session


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


# ── Pydantic models ───────────────────────────────────────────────────────────


class MessageRequest(BaseModel):
    message: str


class SessionResponse(BaseModel):
    session_id: str
    messages: list[str]
    done: bool  # True when the conversation has ended and the session cannot accept more messages


# ── Helpers ───────────────────────────────────────────────────────────────────


def _config(session_id: str) -> dict:
    return {"configurable": {"thread_id": session_id}}


def _drain_new_messages(session_id: str) -> list[str]:
    """Return messages added since the last call for this session."""
    all_messages = _graph.get_state(_config(session_id)).values.get("agent_messages", [])
    shown = _session_shown.get(session_id, 0)
    new = all_messages[shown:]
    _session_shown[session_id] = len(all_messages)
    return new


def _is_done(session_id: str) -> bool:
    return not bool(_graph.get_state(_config(session_id)).next)


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

    initial_state = {
        "user_details": "",
        "secret_question": "",
        "matched_nif": "",
        "secret_answer": "",
        "identity_verified": False,
        "account_type": "",
        "user_request": "",
        "agent_messages": [],
        "log_lines": [f"# Chat Session — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"],
    }
    _graph.invoke(initial_state, config=_config(session_id))

    return SessionResponse(
        session_id=session_id,
        messages=_drain_new_messages(session_id),
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
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")

    if _is_done(session_id):
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="Session is already closed. Start a new session.",
        )

    _graph.invoke(Command(resume=body.message), config=_config(session_id))

    return SessionResponse(
        session_id=session_id,
        messages=_drain_new_messages(session_id),
        done=_is_done(session_id),
    )
