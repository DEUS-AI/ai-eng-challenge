"""Bouncer agent — verifies identity and classifies customer tier."""

from __future__ import annotations

import sqlite3
from typing import Literal

from langchain_core.messages import AIMessage, SystemMessage
from langgraph.types import Command
from pydantic import BaseModel

from src.database import Customer, find_customer_by_identifiers
from src.prompts import load_prompt
from src.schemas import ConversationState

MAX_SECRET_ATTEMPTS = 3

BOUNCER_VERIFY_PROMPT = load_prompt("bouncer_verify")


class BouncerOutput(BaseModel):
    """Structured output for verification decisions."""

    verified: bool = False
    tier: Literal["premium", "regular", "non_client"] | None = None
    secret_correct: bool | None = None
    message: str = ""


def bouncer_node(
    state: ConversationState, llm, db_conn: sqlite3.Connection,
) -> Command[Literal["output_policy"]]:
    """Verify customer identity and classify tier.

    Each invocation handles ONE turn — responds and ends.
    Phase and state fields route the next message back here if needed.
    """
    name = state.get("collected_name")
    phone = state.get("collected_phone")
    iban = state.get("collected_iban")
    secret_attempts = state.get("secret_attempts", 0)

    # Try to match customer in DB
    customer = find_customer_by_identifiers(db_conn, name=name, phone=phone, iban=iban)

    if customer is None:
        # No 2/3 match — non-client
        msg = (
            "Thank you for reaching out. It seems that you are not currently a client "
            "of DEUS Bank. I recommend that you contact your bank's support department "
            "directly for assistance with your inquiry."
        )
        return Command(
            update={
                "messages": [AIMessage(content=msg)],
                "customer_tier": "non_client",
                "identity_verified": False,
                "phase": "complete",
            },
            goto="output_policy",
        )

    # Customer found — check if we're verifying the secret answer
    if state.get("matched_customer_id") is not None:
        # We've already asked the secret question — check the answer
        return _check_secret_answer(state, customer, llm, secret_attempts)

    # First time seeing this customer: ask the secret question
    return _ask_secret_question(state, customer, llm)


def _ask_secret_question(
    state: ConversationState, customer: Customer, llm,
) -> Command[Literal["output_policy"]]:
    """Ask the customer their secret question. Ends the turn."""
    prompt = BOUNCER_VERIFY_PROMPT.format(
        name=state.get("collected_name", "N/A"),
        phone=state.get("collected_phone", "N/A"),
        iban=state.get("collected_iban", "N/A"),
        verification_result="Identity partially verified. Need to ask secret question.",
        secret_instruction=f'Ask the customer this security question: "{customer.secret_question}"',
    )

    messages = [SystemMessage(content=prompt)] + state["messages"]
    from src.guardrails import invoke_with_pii_guard
    result: BouncerOutput = invoke_with_pii_guard(llm, messages, BouncerOutput)

    return Command(
        update={
            "messages": [AIMessage(content=result.message)],
            "matched_customer_id": customer.id,
            "secret_attempts": 0,
            # Stay in verification phase — next message comes back here
        },
        goto="output_policy",
    )


def _check_secret_answer(
    state: ConversationState, customer: Customer, llm, attempts: int,
) -> Command[Literal["output_policy"]]:
    """Check the customer's answer to the secret question. Ends the turn."""
    last_user_msg = ""
    for msg in reversed(state["messages"]):
        if hasattr(msg, "type") and msg.type == "human":
            last_user_msg = msg.content
            break

    # Use LLM to check if answer is semantically correct
    system_prompt = load_prompt("bouncer_secret_check").format(
        secret_question=customer.secret_question,
        secret_answer=customer.secret_answer,
    )

    from langchain_core.messages import HumanMessage
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=f"The customer answered: {last_user_msg}")]
    structured_llm = llm.with_structured_output(BouncerOutput)
    result: BouncerOutput = structured_llm.invoke(messages)

    if result.secret_correct:
        tier = "premium" if customer.premium else "regular"
        return Command(
            update={
                "messages": [AIMessage(content=result.message)],
                "identity_verified": True,
                "customer_tier": tier,
                "phase": "routing",  # Next message goes to specialist
            },
            goto="output_policy",
        )

    new_attempts = attempts + 1
    if new_attempts >= MAX_SECRET_ATTEMPTS:
        msg = (
            "I'm sorry, but we were unable to verify your identity after multiple attempts. "
            "For your security, please visit your nearest DEUS Bank branch with a valid ID, "
            "or contact us again later."
        )
        return Command(
            update={
                "messages": [AIMessage(content=msg)],
                "customer_tier": "non_client",
                "identity_verified": False,
                "secret_attempts": new_attempts,
                "phase": "complete",
            },
            goto="output_policy",
        )

    remaining = MAX_SECRET_ATTEMPTS - new_attempts
    msg = (
        f"That doesn't seem to match our records. "
        f"You have {remaining} attempt(s) remaining. Please try again."
    )
    return Command(
        update={
            "messages": [AIMessage(content=msg)],
            "secret_attempts": new_attempts,
            # Stay in verification phase
        },
        goto="output_policy",
    )
