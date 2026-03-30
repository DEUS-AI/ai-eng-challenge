"""FastAPI application — routes and session management."""

from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from langgraph.checkpoint.sqlite import SqliteSaver

from src.database import init_db, seed_db
from src.graph import build_graph
from src.guardrails import check_input

CONVERSATIONS_DB = "conversations.db"

logger = logging.getLogger(__name__)

load_dotenv()

# Global state
_graph = None
_db_conn = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph, _db_conn
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is required")

    _db_conn = init_db()
    seed_db(_db_conn)

    # SqliteSaver persists conversation state across server restarts
    with SqliteSaver.from_conn_string(CONVERSATIONS_DB) as checkpointer:
        _graph = build_graph(db_conn=_db_conn, api_key=api_key, checkpointer=checkpointer)
        yield

    # Cleanup
    if _db_conn:
        _db_conn.close()


app = FastAPI(
    title="DEUS Bank AI Customer Support",
    description="Multi-agent customer support system powered by LangGraph and Gemini",
    version="0.1.0",
    lifespan=lifespan,
)


class StartResponse(BaseModel):
    session_id: str
    message: str


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    message: str
    phase: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat/start", response_model=StartResponse)
def start_chat() -> StartResponse:
    """Start a new conversation session."""
    if _graph is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    session_id = str(uuid.uuid4())

    # Run the graph with an initial greeting trigger
    config = {"configurable": {"thread_id": session_id}}
    initial_state = {
        "session_id": session_id,
        "messages": [HumanMessage(content="Hello, I need help.")],
        "phase": "greeting",
        "collected_name": None,
        "collected_phone": None,
        "collected_iban": None,
        "matched_customer_id": None,
        "identity_verified": False,
        "secret_attempts": 0,
        "customer_tier": None,
        "department": None,
        "support_number": None,
    }

    try:
        result = _graph.invoke(initial_state, config=config)
        last_ai_msg = _get_last_ai_message(result)
        return StartResponse(session_id=session_id, message=last_ai_msg)
    except Exception as e:
        logger.exception("Error starting conversation")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred starting the conversation: {type(e).__name__}",
        )


@app.post("/chat/message", response_model=ChatResponse)
def send_message(request: ChatRequest) -> ChatResponse:
    """Send a message in an existing conversation."""
    if _graph is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    config = {"configurable": {"thread_id": request.session_id}}

    # Check if session exists by trying to get state
    try:
        current_state = _graph.get_state(config)
        if not current_state.values:
            raise HTTPException(status_code=404, detail="Session not found")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=404, detail="Session not found")

    # Check if conversation is already complete
    phase = current_state.values.get("phase", "greeting")
    if phase == "complete":
        raise HTTPException(
            status_code=400,
            detail="This conversation has already ended. Please start a new session.",
        )

    # Run the graph with the new message
    try:
        result = _graph.invoke(
            {"messages": [HumanMessage(content=request.message)]},
            config=config,
        )
        last_ai_msg = _get_last_ai_message(result)
        current_phase = result.get("phase", phase)
        return ChatResponse(message=last_ai_msg, phase=current_phase)
    except Exception as e:
        logger.exception("Error processing message")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred processing your message: {type(e).__name__}",
        )


def _get_last_ai_message(state: dict) -> str:
    """Extract the last AI message from the state."""
    messages = state.get("messages", [])
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "ai":
            return msg.content
    return "Welcome to DEUS Bank. How can I help you today?"
