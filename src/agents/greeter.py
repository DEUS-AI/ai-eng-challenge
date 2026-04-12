"""Greeter agent — collects customer identifiers (name, phone, IBAN)."""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import AIMessage, SystemMessage
from langgraph.types import Command
from pydantic import BaseModel

from src.guardrails import invoke_with_pii_guard
from src.prompts import load_prompt
from src.schemas import ConversationState

GREETER_SYSTEM_PROMPT = load_prompt("greeter")


class GreeterOutput(BaseModel):
    """Structured output for identifier extraction."""

    name: str | None = None
    phone: str | None = None
    iban: str | None = None
    all_collected: bool = False
    message: str = ""


def greeter_node(state: ConversationState, llm) -> Command[Literal["output_policy"]]:
    """Collect customer identifiers through conversation.

    Each invocation handles ONE turn — responds and ends.
    The phase field routes the next message back here if still collecting.
    """
    messages = [SystemMessage(content=GREETER_SYSTEM_PROMPT)] + state["messages"]

    # Include context about what's already been collected
    collected = []
    if state.get("collected_name"):
        collected.append(f"Name: {state['collected_name']}")
    if state.get("collected_phone"):
        collected.append(f"Phone: {state['collected_phone']}")
    if state.get("collected_iban"):
        collected.append(f"IBAN: {state['collected_iban']}")

    if collected:
        context = "Already collected: " + ", ".join(collected)
        messages.insert(1, SystemMessage(content=context))

    result: GreeterOutput = invoke_with_pii_guard(llm, messages, GreeterOutput)

    # Merge newly extracted info with previously collected
    new_name = result.name or state.get("collected_name")
    new_phone = result.phone or state.get("collected_phone")
    new_iban = result.iban or state.get("collected_iban")

    update: dict = {
        "messages": [AIMessage(content=result.message)],
        "collected_name": new_name,
        "collected_phone": new_phone,
        "collected_iban": new_iban,
    }

    # If all collected, advance phase so next message goes to bouncer
    if result.all_collected:
        update["phase"] = "verification"

    # Always end the turn — next user message re-enters via input_guard
    return Command(update=update, goto="output_policy")
