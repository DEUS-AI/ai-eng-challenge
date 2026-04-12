"""Conversation state and shared schemas for the agent graph."""

from typing import Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from typing_extensions import Annotated


class ConversationState(TypedDict):
    session_id: str
    messages: Annotated[list[BaseMessage], add_messages]
    phase: Literal["greeting", "verification", "routing", "complete"]

    # Greeter outputs
    collected_name: str | None
    collected_phone: str | None
    collected_iban: str | None

    # Bouncer outputs
    matched_customer_id: int | None
    identity_verified: bool
    secret_attempts: int
    customer_tier: Literal["premium", "regular", "non_client"] | None

    # Specialist outputs
    department: str | None
    support_number: str | None

    # Guardrails state
    input_blocked: bool
    last_activity_ts: float | None
    injection_attempts: int
    lockout_until: float | None
