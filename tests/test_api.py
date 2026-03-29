"""Tests for FastAPI endpoints — no LLM calls required.

Uses a mock graph that returns predictable responses to test
the API layer: routing, session management, error handling, status codes.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

import src.app as app_module
from src.app import app


def _make_mock_graph():
    """Create a mock graph that simulates conversation state."""
    mock = MagicMock()
    _states: dict[str, dict] = {}

    def mock_invoke(input_state: dict, config: dict) -> dict:
        thread_id = config["configurable"]["thread_id"]

        if thread_id not in _states:
            # First invocation — start session
            _states[thread_id] = {
                "messages": input_state.get("messages", [])
                + [AIMessage(content="Welcome to DEUS Bank! How can I help you?")],
                "phase": "greeting",
                "session_id": thread_id,
            }
        else:
            # Follow-up — update state based on message
            state = _states[thread_id]
            new_msgs = input_state.get("messages", [])
            user_msg = new_msgs[0].content if new_msgs else ""

            if state["phase"] == "greeting":
                state["messages"].extend(new_msgs)
                state["messages"].append(
                    AIMessage(content="Thank you. I have your details on file.")
                )
                state["phase"] = "verification"
            elif state["phase"] == "verification":
                state["messages"].extend(new_msgs)
                state["messages"].append(
                    AIMessage(content="Your identity has been verified.")
                )
                state["phase"] = "routing"
            elif state["phase"] == "routing":
                state["messages"].extend(new_msgs)
                state["messages"].append(
                    AIMessage(
                        content="For insurance support, call +1999888999."
                    )
                )
                state["phase"] = "complete"

            _states[thread_id] = state

        return _states[thread_id]

    def mock_get_state(config: dict):
        thread_id = config["configurable"]["thread_id"]
        result = MagicMock()
        if thread_id in _states:
            result.values = _states[thread_id]
        else:
            result.values = {}
        return result

    mock.invoke = mock_invoke
    mock.get_state = mock_get_state
    mock._states = _states
    return mock


@pytest.fixture(autouse=True)
def setup_mock_graph():
    """Inject a mock graph into the app module before each test."""
    mock_graph = _make_mock_graph()
    app_module._graph = mock_graph
    yield mock_graph
    app_module._graph = None


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


class TestHealthEndpoint:
    def test_health_returns_ok(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestStartChat:
    def test_returns_session_id_and_greeting(self, client: TestClient) -> None:
        resp = client.post("/chat/start")
        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data
        assert len(data["session_id"]) == 36  # UUID format
        assert "message" in data
        assert len(data["message"]) > 0

    def test_each_start_gives_unique_session(self, client: TestClient) -> None:
        resp1 = client.post("/chat/start")
        resp2 = client.post("/chat/start")
        assert resp1.json()["session_id"] != resp2.json()["session_id"]

    def test_returns_503_when_not_initialized(self, client: TestClient) -> None:
        app_module._graph = None
        resp = client.post("/chat/start")
        assert resp.status_code == 503
        assert "not initialized" in resp.json()["detail"]


class TestSendMessage:
    def test_valid_session_returns_200(self, client: TestClient) -> None:
        start = client.post("/chat/start").json()
        resp = client.post(
            "/chat/message",
            json={"session_id": start["session_id"], "message": "My name is Lisa"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "message" in data
        assert "phase" in data

    def test_phase_transitions(self, client: TestClient) -> None:
        start = client.post("/chat/start").json()
        sid = start["session_id"]

        # greeting → verification
        resp = client.post(
            "/chat/message",
            json={"session_id": sid, "message": "My name is Lisa"},
        )
        assert resp.json()["phase"] == "verification"

        # verification → routing
        resp = client.post(
            "/chat/message",
            json={"session_id": sid, "message": "Yoda"},
        )
        assert resp.json()["phase"] == "routing"

        # routing → complete
        resp = client.post(
            "/chat/message",
            json={"session_id": sid, "message": "I need help with insurance"},
        )
        assert resp.json()["phase"] == "complete"

    def test_invalid_session_returns_404(self, client: TestClient) -> None:
        resp = client.post(
            "/chat/message",
            json={"session_id": "nonexistent-session-id", "message": "Hello"},
        )
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_completed_session_returns_400(self, client: TestClient) -> None:
        start = client.post("/chat/start").json()
        sid = start["session_id"]

        # Drive to completion
        client.post("/chat/message", json={"session_id": sid, "message": "identifiers"})
        client.post("/chat/message", json={"session_id": sid, "message": "secret"})
        client.post("/chat/message", json={"session_id": sid, "message": "request"})

        # Now session is complete — next message should fail
        resp = client.post(
            "/chat/message",
            json={"session_id": sid, "message": "one more thing"},
        )
        assert resp.status_code == 400
        assert "already ended" in resp.json()["detail"].lower()

    def test_returns_503_when_not_initialized(self, client: TestClient) -> None:
        app_module._graph = None
        resp = client.post(
            "/chat/message",
            json={"session_id": "any", "message": "Hello"},
        )
        assert resp.status_code == 503

    def test_missing_fields_returns_422(self, client: TestClient) -> None:
        resp = client.post("/chat/message", json={"message": "Hello"})
        assert resp.status_code == 422

        resp = client.post("/chat/message", json={"session_id": "abc"})
        assert resp.status_code == 422

    def test_empty_message_still_accepted(self, client: TestClient) -> None:
        start = client.post("/chat/start").json()
        resp = client.post(
            "/chat/message",
            json={"session_id": start["session_id"], "message": ""},
        )
        # Empty string is still a valid message — the agent handles it
        assert resp.status_code == 200
