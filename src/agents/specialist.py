"""Specialist agent — classifies request and routes to department."""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import AIMessage, SystemMessage
from langgraph.types import Command
from pydantic import BaseModel

from src.prompts import load_prompt
from src.schemas import ConversationState

DEPARTMENTS = {
    "loans": ["loan", "mortgage", "credit", "financing"],
    "cards": ["card", "visa", "mastercard", "debit"],
    "insurance": ["insurance", "coverage", "policy", "claim"],
    "general": ["account", "balance", "transfer", "statement"],
}

SUPPORT_NUMBERS = {
    "premium": "+1999888999",
    "regular": "+1112112112",
}

SPECIALIST_SYSTEM_PROMPT = load_prompt("specialist")


class SpecialistOutput(BaseModel):
    """Structured output for department routing."""

    department: Literal["loans", "cards", "insurance", "general"] = "general"
    support_number: str = ""
    message: str = ""


def specialist_node(state: ConversationState, llm) -> Command[Literal["output_policy"]]:
    """Route verified customer to the appropriate department."""
    tier = state.get("customer_tier", "regular")
    support_number = SUPPORT_NUMBERS.get(tier, SUPPORT_NUMBERS["regular"])

    prompt = SPECIALIST_SYSTEM_PROMPT.format(
        tier=tier,
        support_number=support_number,
    )

    messages = [SystemMessage(content=prompt)] + state["messages"]
    structured_llm = llm.with_structured_output(SpecialistOutput)
    result: SpecialistOutput = structured_llm.invoke(messages)

    return Command(
        update={
            "messages": [AIMessage(content=result.message)],
            "department": result.department,
            "support_number": result.support_number or support_number,
            "phase": "complete",
        },
        goto="output_policy",
    )
