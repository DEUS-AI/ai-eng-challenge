"""Input guard and output policy for the agent pipeline.

Guardrail layers (executed in order, cheapest first):
  1. Input length validation (<1ms)
  2. Unicode normalization + encoded payload detection (<1ms)
  3. Session timeout check (<1ms)
  4. Rate limit check (<1ms)
  5. Injection pattern detection (<1ms)
  6. Topic guardrail — routing phase only (<1ms)
  7. [Agent processes message]
  8. Output policy — PII detection + three-tier protection
"""

from __future__ import annotations

import base64
import re
import time
import unicodedata
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage
from pydantic import BaseModel

from src.prompts import load_prompt
from src.schemas import ConversationState

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_INPUT_LENGTH = 2000
SESSION_TIMEOUT_SECONDS = 900  # 15 minutes
INJECTION_LOCKOUT_THRESHOLD = 3
LOCKOUT_BASE_SECONDS = 30
LOCKOUT_MAX_SECONDS = 600

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
    r"repeat\s+(your|the)\s+(system|initial|first)\s+(prompt|instruction)",
    r"show\s+me\s+your\s+(rules|instructions|prompt)",
    r"forget\s+(everything|all|your)\s+(you|previous)",
    r"new\s+instruction[s]?\s*:",
    r"do\s+not\s+follow\s+(your|the|any)\s+(rules|instructions)",
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

LENGTH_REFUSAL = (
    "Your message is too long. Please keep messages under "
    f"{MAX_INPUT_LENGTH:,} characters."
)

TIMEOUT_REFUSAL = (
    "Your session has expired due to inactivity. "
    "Please start a new conversation."
)

TOPIC_REFUSAL = (
    "I'm here to help with your banking needs — accounts, cards, loans, "
    "transfers, and more. How can I assist you with your banking today?"
)

# ---------------------------------------------------------------------------
# Unicode normalization + encoded payload detection
# ---------------------------------------------------------------------------

# Zero-width and bidirectional override characters
_ZWC_PATTERN = re.compile(
    r"[\u200b\u200c\u200d\u200e\u200f\ufeff"
    r"\u202a\u202b\u202c\u202d\u202e"
    r"\u2066\u2067\u2068\u2069]"
)

# Base64-like substrings (20+ chars, valid base64 alphabet)
_BASE64_PATTERN = re.compile(r"[A-Za-z0-9+/]{20,}={0,2}")


def normalize_input(text: str) -> str:
    """Strip zero-width/bidi chars, normalize Unicode, collapse whitespace."""
    text = _ZWC_PATTERN.sub("", text)
    text = unicodedata.normalize("NFKD", text)
    text = unicodedata.normalize("NFC", text)  # Recompose for readable text
    text = re.sub(r"\s+", " ", text).strip()
    return text


def decode_encoded_payloads(text: str) -> list[str]:
    """Find and decode base64 substrings for secondary injection check.

    Only decodes substrings that look like base64 (20+ chars).
    Does NOT modify the original input.
    """
    decoded = []
    for match in _BASE64_PATTERN.finditer(text):
        try:
            raw = base64.b64decode(match.group(), validate=True)
            result = raw.decode("utf-8", errors="ignore")
            if result and len(result) > 5:  # Ignore very short/garbage decodes
                decoded.append(result)
        except Exception:
            pass
    return decoded


# ---------------------------------------------------------------------------
# Input validation checks
# ---------------------------------------------------------------------------

def check_input_length(message: str) -> str | None:
    """Reject messages over the length cap."""
    if len(message) > MAX_INPUT_LENGTH:
        return LENGTH_REFUSAL
    return None


def check_injection(message: str) -> str | None:
    """Check for injection patterns. Returns refusal if blocked, None if safe."""
    for pattern in _COMPILED_PATTERNS:
        if pattern.search(message):
            return GUARD_REFUSAL
    return None


def check_session_timeout(state: ConversationState) -> str | None:
    """Check if session has been idle beyond the timeout."""
    last_activity = state.get("last_activity_ts")
    if last_activity is not None and (time.time() - last_activity) > SESSION_TIMEOUT_SECONDS:
        return TIMEOUT_REFUSAL
    return None


def check_rate_limit(state: ConversationState) -> str | None:
    """Check if session is locked out or over the rate limit.

    Uses injection_attempts for progressive lockout.
    Uses last_activity_ts for basic RPM approximation.
    """
    now = time.time()

    # Check lockout
    lockout_until = state.get("lockout_until")
    if lockout_until is not None and now < lockout_until:
        remaining = int(lockout_until - now)
        return f"Too many policy violations. Please try again in {remaining} seconds."

    return None


def record_injection_attempt(state: ConversationState) -> dict:
    """Update state after an injection attempt is detected."""
    attempts = state.get("injection_attempts", 0) + 1
    update: dict = {"injection_attempts": attempts}

    if attempts >= INJECTION_LOCKOUT_THRESHOLD:
        duration = min(
            LOCKOUT_BASE_SECONDS * (2 ** (attempts - INJECTION_LOCKOUT_THRESHOLD)),
            LOCKOUT_MAX_SECONDS,
        )
        update["lockout_until"] = time.time() + duration

    return update


def check_topic(message: str, phase: str) -> str | None:
    """Check if message is on-topic. Only active during the routing phase.

    During greeting/verification, users send names, IBANs, secret answers
    that don't contain banking keywords — so topic check is skipped.
    """
    if phase != "routing":
        return None

    # Short messages or common confirmations always pass
    if len(message.strip()) < 20:
        return None

    banking_keywords = [
        "account", "balance", "transfer", "loan", "mortgage", "credit",
        "card", "debit", "insurance", "coverage", "policy", "claim",
        "deposit", "withdrawal", "fee", "branch", "atm", "payment",
        "statement", "wire", "savings", "checking", "interest", "rate",
        "bank", "money", "fund", "invest", "iban", "swift",
    ]

    message_lower = message.lower()
    if any(kw in message_lower for kw in banking_keywords):
        return None

    return TOPIC_REFUSAL


# ---------------------------------------------------------------------------
# Output policy: PII detection
# ---------------------------------------------------------------------------

PHONE_PATTERN = re.compile(r"\+?\d[\d\s\-()]{7,}\d")
IBAN_PATTERN = re.compile(r"[A-Z]{2}\d{2}[A-Z0-9\s]{10,30}", re.IGNORECASE)

MAX_PII_RETRIES = 3

PII_RETRY_INSTRUCTION = load_prompt("pii_retry")


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
        messages = messages + [
            AIMessage(content=result.message),
            SystemMessage(content=PII_RETRY_INSTRUCTION),
        ]
        result = structured_llm.invoke(messages)

    if contains_pii(result.message):
        result.message = redact_pii(result.message)

    return result


def check_output(response: str, state: ConversationState) -> str:
    """Final safety net in the output policy node."""
    is_verified = state.get("identity_verified", False)
    if is_verified:
        return response

    if not contains_pii(response):
        return response

    phase = state.get("phase", "greeting")
    if phase in ("greeting", "verification"):
        return redact_pii(response)

    return POLICY_FALLBACK


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def input_guard_node(state: ConversationState) -> dict:
    """Graph node: validate input through all guardrail layers.

    Order: length → normalize → timeout → rate limit → injection → topic
    Sets input_blocked=True if any check fails, so the router can short-circuit.
    """
    if not state["messages"]:
        return {"input_blocked": False, "last_activity_ts": time.time()}

    last_msg = state["messages"][-1]
    if not (hasattr(last_msg, "type") and last_msg.type == "human"):
        return {"input_blocked": False, "last_activity_ts": time.time()}

    raw_message = last_msg.content

    # 1. Length check
    refusal = check_input_length(raw_message)
    if refusal:
        return {
            "messages": [AIMessage(content=refusal)],
            "input_blocked": True,
            "last_activity_ts": time.time(),
        }

    # 2. Normalize
    normalized = normalize_input(raw_message)

    # 3. Session timeout
    refusal = check_session_timeout(state)
    if refusal:
        return {
            "messages": [AIMessage(content=refusal)],
            "input_blocked": True,
            "phase": "complete",
        }

    # 4. Rate limit / lockout
    refusal = check_rate_limit(state)
    if refusal:
        return {
            "messages": [AIMessage(content=refusal)],
            "input_blocked": True,
            "last_activity_ts": time.time(),
        }

    # 5. Injection detection (on normalized + decoded payloads)
    refusal = check_injection(normalized)
    if not refusal:
        for decoded in decode_encoded_payloads(normalized):
            refusal = check_injection(decoded)
            if refusal:
                break

    if refusal:
        lockout_update = record_injection_attempt(state)
        return {
            "messages": [AIMessage(content=refusal)],
            "input_blocked": True,
            "last_activity_ts": time.time(),
            **lockout_update,
        }

    # 6. Topic guardrail
    phase = state.get("phase", "greeting")
    refusal = check_topic(normalized, phase)
    if refusal:
        return {
            "messages": [AIMessage(content=refusal)],
            "input_blocked": True,
            "last_activity_ts": time.time(),
        }

    # All checks passed
    return {"input_blocked": False, "last_activity_ts": time.time()}


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
