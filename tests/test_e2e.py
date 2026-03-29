"""End-to-end tests that run full conversations through the graph with a real LLM.

Requires GEMINI_API_KEY in .env. Skipped if not available.
Saves conversation transcripts to tests/e2e_reports/.

IMPORTANT: These tests hit the real Gemini API with strict rate limits.
- gemini-2.5-flash free tier: 5 RPM, 20 requests/day
- Tests are ordered to minimize LLM calls and respect rate limits.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

import pytest
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from src.database import init_db, seed_db
from src.graph import build_graph

# Delay between turns to respect 5 RPM limit (12s minimum, 15s for safety)
TURN_DELAY = 15
# Extra delay between scenarios to let the rate window slide
SCENARIO_DELAY = 25

load_dotenv()

REPORTS_DIR = Path(__file__).parent / "e2e_reports"
REPORTS_DIR.mkdir(exist_ok=True)

API_KEY = os.getenv("GEMINI_API_KEY")
SKIP_REASON = "GEMINI_API_KEY not set — skipping E2E tests"


def _build_test_graph():
    """Build a graph with a fresh in-memory DB."""
    conn = init_db(":memory:")
    seed_db(conn)
    return build_graph(db_conn=conn, api_key=API_KEY)


def _get_last_ai_message(state: dict) -> str:
    """Extract the last AI message content."""
    for msg in reversed(state.get("messages", [])):
        if hasattr(msg, "type") and msg.type == "ai":
            return msg.content
    return ""


def _make_initial_state(thread_id: str, first_message: str) -> dict:
    return {
        "session_id": thread_id,
        "messages": [HumanMessage(content=first_message)],
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


def _run_conversation(
    graph, thread_id: str, first_message: str, followup_messages: list[str],
) -> list[dict]:
    """Run a multi-turn conversation, returning a transcript log."""
    config = {"configurable": {"thread_id": thread_id}}
    transcript = []

    # Turn 0: initial message
    initial_state = _make_initial_state(thread_id, first_message)
    result = graph.invoke(initial_state, config=config)
    ai_msg = _get_last_ai_message(result)
    transcript.append({
        "turn": 0,
        "user": first_message,
        "assistant": ai_msg,
        "phase": result.get("phase", "unknown"),
        "customer_tier": result.get("customer_tier"),
        "department": result.get("department"),
        "identity_verified": result.get("identity_verified"),
    })
    print(f"  Turn 0 [{result.get('phase')}]: {ai_msg[:120]}")

    # Follow-up turns with rate-limit delay
    for i, msg in enumerate(followup_messages, start=1):
        time.sleep(TURN_DELAY)
        result = graph.invoke(
            {"messages": [HumanMessage(content=msg)]},
            config=config,
        )
        ai_msg = _get_last_ai_message(result)
        transcript.append({
            "turn": i,
            "user": msg,
            "assistant": ai_msg,
            "phase": result.get("phase", "unknown"),
            "customer_tier": result.get("customer_tier"),
            "department": result.get("department"),
            "identity_verified": result.get("identity_verified"),
        })
        print(f"  Turn {i} [{result.get('phase')}]: {ai_msg[:120]}")

    return transcript


def _save_report(name: str, transcript: list[dict], passed: bool, notes: str = "") -> None:
    """Save a conversation transcript as a JSON report."""
    report = {
        "test": name,
        "timestamp": datetime.now().isoformat(),
        "model": "gemini-2.5-flash",
        "passed": passed,
        "notes": notes,
        "turns": len(transcript),
        "llm_calls_estimated": sum(
            1 for t in transcript
            if t.get("phase") not in ("complete",)
            or t["turn"] == 0
        ),
        "transcript": transcript,
    }
    filename = REPORTS_DIR / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filename.write_text(json.dumps(report, indent=2, default=str))

    # Print summary
    print(f"\n{'='*70}")
    print(f"  E2E REPORT: {name}")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    if notes:
        print(f"  Notes: {notes}")
    print(f"  Turns: {len(transcript)}")
    print(f"  Report: {filename}")
    print(f"{'='*70}")

    # Full transcript
    for turn in transcript:
        print(f"\n  [Turn {turn['turn']}] Phase={turn.get('phase')} | Tier={turn.get('customer_tier')} | Dept={turn.get('department')}")
        print(f"    USER: {turn['user'][:200]}")
        print(f"    BOT:  {turn['assistant'][:200]}")
    print()


# =============================================================================
# SCENARIO ORDER: non-client first (1 LLM call), then premium (4), then regular (4)
# Total: 9 LLM calls out of 20 daily limit
# =============================================================================


@pytest.mark.skipif(not API_KEY, reason=SKIP_REASON)
class TestE2ENonClient:
    """Scenario 1/3: Unknown person — bouncer rejects with no LLM call.
    Expected LLM calls: 1 (greeter only)
    """

    def test_non_client_rejection(self) -> None:
        print("\n\n>>> SCENARIO 1/3: Non-Client Rejection")
        graph = _build_test_graph()

        transcript = _run_conversation(
            graph,
            thread_id="e2e-nonclient-001",
            first_message="Hello, my name is Bob Nobody, my phone is +9999999999, my IBAN is XX00000000000000000",
            followup_messages=[
                "Yes, that is correct",  # → bouncer: no match → hardcoded rejection (0 LLM)
            ],
        )

        last_turn = transcript[-1]
        passed = True
        notes = []

        if last_turn.get("customer_tier") != "non_client":
            passed = False
            notes.append(f"Expected tier=non_client, got {last_turn.get('customer_tier')}")

        # Should NOT contain any support phone numbers
        for turn in transcript:
            if "+1999888999" in turn["assistant"] or "+1112112112" in turn["assistant"]:
                passed = False
                notes.append("Support numbers leaked to non-client")
                break

        _save_report("non_client_rejection", transcript, passed, "; ".join(notes))
        assert passed, f"E2E failed: {'; '.join(notes)}"


@pytest.mark.skipif(not API_KEY, reason=SKIP_REASON)
class TestE2EPremiumClient:
    """Scenario 2/3: Lisa (premium) — full verification + specialist routing.
    Expected LLM calls: 4 (greeter + bouncer ask + bouncer check + specialist)
    """

    def test_premium_happy_path(self) -> None:
        # Delay between scenarios to let rate window slide
        time.sleep(SCENARIO_DELAY)
        print("\n\n>>> SCENARIO 2/3: Premium Happy Path (Lisa)")
        graph = _build_test_graph()

        transcript = _run_conversation(
            graph,
            thread_id="e2e-premium-001",
            first_message="Hello, my name is Lisa, my phone number is +1122334455, and my IBAN is DE89370400440532013000",
            followup_messages=[
                "OK, go ahead with verification",  # → bouncer asks secret question (1 LLM)
                "Yoda",  # → bouncer checks answer (1 LLM) → verified, phase=routing
                "I need help with my insurance",  # → specialist routes (1 LLM)
            ],
        )

        last_turn = transcript[-1]
        passed = True
        notes = []

        if last_turn.get("customer_tier") != "premium":
            passed = False
            notes.append(f"Expected tier=premium, got {last_turn.get('customer_tier')}")

        if not last_turn.get("identity_verified"):
            # Check any turn for identity_verified (might be set earlier)
            verified = any(t.get("identity_verified") for t in transcript)
            if not verified:
                passed = False
                notes.append("Identity not verified")

        if "+1999888999" not in last_turn["assistant"]:
            passed = False
            notes.append("Premium support number +1999888999 not in final response")

        if last_turn.get("department") is None:
            passed = False
            notes.append("No department classified")

        _save_report("premium_happy_path", transcript, passed, "; ".join(notes))
        assert passed, f"E2E failed: {'; '.join(notes)}"


@pytest.mark.skipif(not API_KEY, reason=SKIP_REASON)
class TestE2ERegularClient:
    """Scenario 3/3: Anna Schmidt (regular) — full verification + specialist routing.
    Expected LLM calls: 4 (greeter + bouncer ask + bouncer check + specialist)
    """

    def test_regular_happy_path(self) -> None:
        time.sleep(SCENARIO_DELAY)
        print("\n\n>>> SCENARIO 3/3: Regular Happy Path (Anna Schmidt)")
        graph = _build_test_graph()

        transcript = _run_conversation(
            graph,
            thread_id="e2e-regular-001",
            first_message="Hi, I am Anna Schmidt, phone +4917612345678, IBAN DE44500105175407324931",
            followup_messages=[
                "Sure, go ahead",  # → bouncer asks secret question (1 LLM)
                "Blue",  # → bouncer checks answer (1 LLM) → verified, phase=routing
                "I need to check my account balance",  # → specialist routes (1 LLM)
            ],
        )

        last_turn = transcript[-1]
        passed = True
        notes = []

        if last_turn.get("customer_tier") != "regular":
            passed = False
            notes.append(f"Expected tier=regular, got {last_turn.get('customer_tier')}")

        if "+1112112112" not in last_turn["assistant"]:
            passed = False
            notes.append("Regular support number +1112112112 not in final response")

        if last_turn.get("department") is None:
            passed = False
            notes.append("No department classified")

        _save_report("regular_happy_path", transcript, passed, "; ".join(notes))
        assert passed, f"E2E failed: {'; '.join(notes)}"
