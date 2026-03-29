"""Input guard and output policy for the agent pipeline.

Three-tier PII protection for unverified users:
  1. Prompt enforcement — agent system prompts forbid echoing PII
  2. Retry — if the LLM still outputs PII, retry up to MAX_PII_RETRIES times
  3. Redact — after retries exhausted, silently strip PII from the response
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage
from pydantic import BaseModel

from src.schemas import ConversationState

# ---------------------------------------------------------------------------
# Input guard: prompt injection detection
# ---------------------------------------------------------------------------

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts)",
    r"ignore\s+your\s+(instructions|rules|system\s+prompt)",
    r"you\s+are\s+now\s+a",
    r"pretend\s+(to\s+be|you\s+are)",
    r"act\s+as\s+(if|though)",
    r"system\s*prompt",
    r"reveal\s+(your|the)\s+(instructions|prompt|system)",
    r"what\s+are\s+your\s+instructions",
    r"disregard\s+(all|your|previous)",
    r"override\s+(your|the|all)",
    r"jailbreak",
    r"DAN\s+mode",
]

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]

GUARD_REFUSAL = (
    "I appreciate your message, but I can only assist with DEUS Bank "
    "customer support inquiries. How can I help you with your banking needs today?"
)

POLICY_FALLBACK = (
    "I apologize, but I'm unable to process that request at this time. "
    "Please contact DEUS Bank support directly for further assistance."
)

# ---------------------------------------------------------------------------
# Output policy: PII detection
# ---------------------------------------------------------------------------

PHONE_PATTERN = re.compile(r"\+?\d[\d\s\-()]{7,}\d")
IBAN_PATTERN = re.compile(r"[A-Z]{2}\d{2}[A-Z0-9\s]{10,30}", re.IGNORECASE)

MAX_PII_RETRIES = 3

PII_RETRY_INSTRUCTION = (
    "Your previous response contained sensitive data (phone numbers or IBANs). "
    "This is NOT allowed. Rewrite your response WITHOUT including any phone numbers "
    "or IBAN values. Say 'I have your phone number on file' instead of repeating it."
)


def check_input(message: str) -> str | None:
    """Check user input for injection patterns.

    Returns a refusal message if blocked, None if the input is safe.
    """
    for pattern in _COMPILED_PATTERNS:
        if pattern.search(message):
            return GUARD_REFUSAL
    return None


def contains_pii(text: str) -> bool:
    """Check if text contains phone numbers or IBANs."""
    return bool(PHONE_PATTERN.search(text)) or bool(IBAN_PATTERN.search(text))


def redact_pii(text: str) -> str:
    """Replace phone numbers and IBANs with [REDACTED]."""
    cleaned = PHONE_PATTERN.sub("[REDACTED]", text)
    cleaned = IBAN_PATTERN.sub("[REDACTED]", cleaned)
    return cleaned


def invoke_with_pii_guard(
    llm: Any,
    messages: list,
    output_schema: type[BaseModel],
) -> BaseModel:
    """Call the LLM with PII retry protection.

    1. Invoke the LLM with structured output.
    2. If the response message contains PII, retry with an extra instruction
       telling the LLM to remove it (up to MAX_PII_RETRIES times).
    3. After retries exhausted, redact PII from the final response.
    """
    structured_llm = llm.with_structured_output(output_schema)
    result = structured_llm.invoke(messages)

    for _attempt in range(MAX_PII_RETRIES):
        if not contains_pii(result.message):
            return result
        # Retry: append correction instruction and re-invoke
        messages = messages + [
            AIMessage(content=result.message),
            SystemMessage(content=PII_RETRY_INSTRUCTION),
        ]
        result = structured_llm.invoke(messages)

    # All retries exhausted — redact as last resort
    if contains_pii(result.message):
        result.message = redact_pii(result.message)

    return result


def check_output(response: str, state: ConversationState) -> str:
    """Final safety net in the output policy node.

    For unverified users in later phases (routing/complete), replace
    the entire response if PII is found — this indicates a genuine
    data leak from the DB, not an echo of user input.
    """
    is_verified = state.get("identity_verified", False)
    if is_verified:
        return response

    if not contains_pii(response):
        return response

    phase = state.get("phase", "greeting")
    if phase in ("greeting", "verification"):
        # Should not happen if invoke_with_pii_guard is used, but
        # if it does, redact rather than replace the whole message.
        return redact_pii(response)

    # Later phases: full fallback (possible data leak from DB)
    return POLICY_FALLBACK


def input_guard_node(state: ConversationState) -> dict:
    """Graph node that checks input before passing to agents."""
    if not state["messages"]:
        return {}

    last_msg = state["messages"][-1]
    if hasattr(last_msg, "type") and last_msg.type == "human":
        refusal = check_input(last_msg.content)
        if refusal:
            return {
                "messages": [AIMessage(content=refusal)],
                "phase": state.get("phase", "greeting"),
            }

    return {}


def output_policy_node(state: ConversationState) -> dict:
    """Graph node that validates the last AI response."""
    if not state["messages"]:
        return {}

    last_msg = state["messages"][-1]
    if hasattr(last_msg, "type") and last_msg.type == "ai":
        checked = check_output(last_msg.content, state)
        if checked != last_msg.content:
            return {"messages": [AIMessage(content=checked)]}

    return {}
