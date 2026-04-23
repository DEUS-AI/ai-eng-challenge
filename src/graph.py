"""
StateGraph definition for DEUS Bank customer support.

Flow:
  START → greeting → details_input*
                        └─ greeter_verify [verify_identity 2-of-3]
                                ├─ no match → reject → END
                                └─ 2/3 match → secret_question_node → secret_input*
                                                                          ├─ wrong answer → reject → END
                                                                          └─ verified → bouncer [verify_account_type]
                                                                                          ├─ not found → reject → END
                                                                                          └─ found → request_welcome → request_input*
                                                                                                        ├─ exit → farewell → END
                                                                                                        └─ request → specialist → request_input* (loop)
* = interrupt (waits for human input)
"""
import json
import asyncio
import operator
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langgraph.checkpoint.memory import InMemorySaver

from agents.greeter import greeter_agent
from agents.bouncer import bouncer_agent
from agents.specialist import specialist_agent
from ai.llm import call_google_generative_ai_model
from config.models import Skill, IdentityResult, AccountResult, SpecialistDecision
from utils.get_prompts import get_prompt
from utils.database_queries import verify_secret

_llm = call_google_generative_ai_model()

def _llm_respond(prompt_key: str) -> str:
    prompt = get_prompt(prompt_key)
    result = _llm.invoke(prompt)
    return result.content.strip()

# ── Shared state ─────────────────────────────────────────────────────────────

class ChatState(TypedDict):
    user_details: str
    secret_question: str
    matched_nif: str
    secret_answer: str
    identity_verified: bool
    account_type: str
    user_request: str
    details_retry_count: int
    agent_messages: Annotated[list[str], operator.add]
    log_lines: Annotated[list[str], operator.add]


# ── Helper ────────────────────────────────────────────────────────────────────

def _call_agent(agent, message: str) -> str:
    result = agent.invoke({"messages": [{"role": "user", "content": message}]})
    for msg in reversed(result["messages"]):
        if getattr(msg, "content", None):
            return msg.content
    return ""


# ── Nodes ─────────────────────────────────────────────────────────────────────

def greeting(state: ChatState) -> dict:
    response = _call_agent(greeter_agent, "Greet the customer and ask for their authentication details.")
    return {
        "agent_messages": [response],
        "log_lines": [f"\n**Greeter Agent**: {response}"],
    }


def details_input(state: ChatState) -> dict:
    user_input = interrupt("Waiting for customer details")
    return {
        "user_details": user_input.strip(),
        "log_lines": [f"\n**You**: {user_input}"],
    }


def greeter_verify(state: ChatState) -> dict:
    """Runs greeter_agent with user details, parses tool response to extract secret question."""
    result = greeter_agent.invoke({
        "messages": [{"role": "user", "content": state["user_details"]}]
    })
    tool_content = next(
        (m.content for m in result["messages"] if type(m).__name__ == "ToolMessage"),
        "",
    )
    try:
        data = json.loads(tool_content)
    except (json.JSONDecodeError, TypeError):
        data = {}
    if data.get("status") == "match":
        parsed = IdentityResult(
            verified=False,
            secret_question=data.get("secret_question", ""),
            matched_nif=data.get("nif", ""),
        )
    else:
        parsed = IdentityResult(verified=False)
    return {
        "identity_verified": parsed.verified,
        "secret_question": parsed.secret_question,
        "matched_nif": parsed.matched_nif,
        "log_lines": [f"\n**[internal greeter_verify]**: {parsed.model_dump()}"],
    }


_MAX_DETAILS_RETRIES = 2


def retry_details(state: ChatState) -> dict:
    attempt = state.get("details_retry_count", 0) + 1
    msg = (
        "I wasn't able to find your details. "
        "Please provide at least two of the following: full name, phone number, or IBAN."
    )
    return {
        "agent_messages": [msg],
        "details_retry_count": attempt,
        "log_lines": [f"\n**Greeter Agent**: {msg}"],
    }


def secret_question_node(state: ChatState) -> dict:
    msg = f"Security question: {state['secret_question']}"
    return {
        "agent_messages": [msg],
        "log_lines": [f"\n**Greeter Agent**: {msg}"],
    }


def secret_input(state: ChatState) -> dict:
    answer = interrupt("Waiting for secret answer")
    # Strip whitespace and trailing punctuation added by STT (e.g. "Testing." → "Testing")
    cleaned = answer.strip().rstrip(".,!?;:")
    return {
        "secret_answer": cleaned,
        "log_lines": [f"\n**You**: {answer}"],
    }


def secret_verify(state: ChatState) -> dict:
    """Verifies the secret answer directly against the database."""
    verified = asyncio.run(verify_secret(state["matched_nif"], state["secret_answer"]))
    return {
        "identity_verified": verified,
        "log_lines": [f"\n**[internal secret_verify]**: {{'verified': {verified}}}"],
    }


def bouncer(state: ChatState) -> dict:
    """Runs bouncer_agent, parses ToolMessage into AccountResult."""
    result = bouncer_agent.invoke({
        "messages": [{"role": "user", "content": state["matched_nif"]}]
    })
    tool_content = next(
        (m.content for m in result["messages"] if type(m).__name__ == "ToolMessage"),
        "",
    )
    parsed = AccountResult(
        account_type=tool_content,
        found="not found" not in tool_content.lower(),
    )
    return {
        "account_type": parsed.account_type,
        "log_lines": [f"\n**[internal bouncer]**: {parsed.model_dump()}"],
    }


def reject(state: ChatState) -> dict:
    msg = _llm_respond("REJECT_PROMPT")
    return {
        "agent_messages": [msg],
        "log_lines": [f"\n**Bouncer Agent**: {msg}"],
    }


def reject_secret(state: ChatState) -> dict:
    msg = _llm_respond("SECRET_REJECT_PROMPT")
    return {
        "agent_messages": [msg],
        "log_lines": [f"\n**Bouncer Agent**: {msg}"],
    }


def request_welcome(state: ChatState) -> dict:
    msg = _llm_respond("REQUEST_WELCOME_PROMPT")
    return {
        "agent_messages": [msg],
        "log_lines": [f"\n**Specialist Agent**: {msg}"],
    }


def request_input(state: ChatState) -> dict:
    user_request = interrupt("Waiting for service request")
    return {
        "user_request": user_request.strip(),
        "log_lines": [f"\n**You**: {user_request}"],
    }


# Patterns that indicate an attempt to inject system-level instructions into user input.
_INJECTION_PATTERNS = (
    "[system",
    "ignore all",
    "ignore previous",
    "new rule:",
    "override:",
    "context update",
    "you are now",
    "act as",
)


def _sanitize_user_line(line: str) -> str | None:
    """Return the line if it looks safe, or None to drop it from history."""
    lower = line.lower()
    if any(pat in lower for pat in _INJECTION_PATTERNS):
        return None
    return line


def specialist(state: ChatState) -> dict:
    """Runs specialist_agent with conversation history for contextual memory."""
    log_lines = state.get("log_lines", [])
    nif = state["matched_nif"]
    account_type = state["account_type"]

    # Build multi-turn chat history from log_lines so the model has proper
    # conversational context (assistant/user turns, not a flat string block).
    turns: list[dict] = []
    for line in log_lines:
        stripped = line.strip()
        if stripped.startswith("**Specialist Agent**:"):
            content = stripped[len("**Specialist Agent**: "):]
            turns.append({"role": "assistant", "content": content})
        elif stripped.startswith("**You**:"):
            safe = _sanitize_user_line(stripped)
            if safe:
                content = safe[len("**You**: "):]
                turns.append({"role": "user", "content": content})

    # System message is just the prompt (no history block needed)
    history: list[dict] = []

    # Replay previous turns so the model understands the conversational context
    history.extend(turns)

    # Append the current request as the final user turn with auth context
    history.append({
        "role": "user",
        "content": (
            f"[Customer NIF: {nif} | Account type: {account_type}]\n"
            f"{state['user_request']}"
        ),
    })
    result = specialist_agent.invoke({"messages": history})
    # Identify which tool (if any) was called in this turn
    ai_msg_with_tool = next(
        (m for m in result["messages"]
         if type(m).__name__ == "AIMessage" and getattr(m, "tool_calls", [])),
        None,
    )
    tool_name = ai_msg_with_tool.tool_calls[0]["name"] if ai_msg_with_tool else None
    tool_msg = next(
        (m for m in result["messages"] if type(m).__name__ == "ToolMessage"),
        None,
    )
    # Always prefer a follow-up AIMessage composed by the LLM after the tool call
    final_ai_msg = next(
        (m.content for m in reversed(result["messages"])
         if type(m).__name__ == "AIMessage"
         and getattr(m, "content", None)
         and not getattr(m, "tool_calls", [])),
        None,
    )
    if tool_msg and tool_name == "delegate_hitl":
        # Specialist delegation — build response from employee name
        skill_value = ai_msg_with_tool.tool_calls[0]["args"].get("skill") if ai_msg_with_tool else None
        try:
            skill = Skill(skill_value)
        except (ValueError, TypeError):
            skill = None
        parsed = SpecialistDecision(in_scope=True, skill=skill)
        employee_name = tool_msg.content
        account_type = state.get("account_type", "regular").lower()
        if final_ai_msg:
            response = final_ai_msg
        elif employee_name == "no_employee_available":
            response = "No specialist is currently available. Your request has been escalated."
        elif account_type == "premium":
            response = (
                f"Thank you for reaching out. As a premium client, we value your experience. "
                f"{employee_name} will attend your request. "
                f"For immediate support, you can also contact our dedicated support line at +1999888999."
            )
        else:
            response = (
                f"{employee_name} will attend your request. "
                f"For assistance, you can also call our support department at +1112112112."
            )
        return {
            "agent_messages": [response],
            "log_lines": [f"\n**Specialist Agent**: {response}"],
        }
    elif tool_msg and final_ai_msg:
        # Another tool was called (e.g. get_account_summary, log_complaint) —
        # the LLM already composed a natural-language response after seeing the tool result
        return {
            "agent_messages": [final_ai_msg],
            "log_lines": [f"\n**Specialist Agent**: {final_ai_msg}"],
        }
    elif final_ai_msg:
        # No tool, LLM replied directly (e.g. conversation recall, refusal)
        parsed = SpecialistDecision(in_scope=False, refusal_message=final_ai_msg)
        return {
            "agent_messages": [parsed.refusal_message],
            "log_lines": [f"\n**Specialist Agent**: {parsed.refusal_message}"],
        }
    # Fallback
    refusal = "I can only assist you with your own banking services."
    parsed = SpecialistDecision(in_scope=False, refusal_message=refusal)
    return {
        "agent_messages": [parsed.refusal_message],
        "log_lines": [f"\n**Specialist Agent**: {parsed.refusal_message}"],
    }


def farewell(state: ChatState) -> dict:
    msg = "Thank you for contacting DEUS Bank. Have a great day! Goodbye!"
    return {
        "agent_messages": [msg],
        "log_lines": [f"\n**Greeter Agent**: {msg}"],
    }


# ── Routing ───────────────────────────────────────────────────────────────────

def route_after_greeter(state: ChatState) -> str:
    if state.get("secret_question"):
        return "secret_question_node"
    if state.get("details_retry_count", 0) < _MAX_DETAILS_RETRIES:
        return "retry_details"
    return "reject_details"


def route_after_secret_verify(state: ChatState) -> str:
    if state.get("identity_verified"):
        return "bouncer"
    return "reject_secret"


def route_after_bouncer(state: ChatState) -> str:
    low = state.get("account_type", "").lower()
    if "not found" in low or "not a customer" in low:
        return "reject"
    return "request_welcome"


def route_after_request_input(state: ChatState) -> str:
    req = state.get("user_request", "").lstrip("\\").lower()
    if req in {"exit", "quit", "sair", "bye", "tchau", "goodbye"}:
        return "farewell"
    return "specialist"


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    builder = StateGraph(ChatState)

    builder.add_node("greeting", greeting)
    builder.add_node("details_input", details_input)
    builder.add_node("greeter_verify", greeter_verify)
    builder.add_node("retry_details", retry_details)
    builder.add_node("secret_question_node", secret_question_node)
    builder.add_node("secret_input", secret_input)
    builder.add_node("secret_verify", secret_verify)
    builder.add_node("bouncer", bouncer)
    builder.add_node("reject", reject)
    builder.add_node("reject_secret", reject_secret)
    builder.add_node("request_welcome", request_welcome)
    builder.add_node("request_input", request_input)
    builder.add_node("specialist", specialist)
    builder.add_node("farewell", farewell)

    builder.add_edge(START, "greeting")
    builder.add_edge("greeting", "details_input")
    builder.add_edge("details_input", "greeter_verify")
    builder.add_conditional_edges(
        "greeter_verify",
        route_after_greeter,
        {"secret_question_node": "secret_question_node", "retry_details": "retry_details", "reject_details": "reject"},
    )
    builder.add_edge("retry_details", "details_input")
    builder.add_edge("secret_question_node", "secret_input")
    builder.add_edge("secret_input", "secret_verify")
    builder.add_conditional_edges(
        "secret_verify",
        route_after_secret_verify,
        {"bouncer": "bouncer", "reject_secret": "reject_secret"},
    )
    builder.add_conditional_edges(
        "bouncer",
        route_after_bouncer,
        {"reject": "reject", "request_welcome": "request_welcome"},
    )
    builder.add_edge("reject", END)
    builder.add_edge("reject_secret", END)
    builder.add_edge("request_welcome", "request_input")
    builder.add_conditional_edges(
        "request_input",
        route_after_request_input,
        {"farewell": "farewell", "specialist": "specialist"},
    )
    builder.add_edge("specialist", "request_input")
    builder.add_edge("farewell", END)

    return builder


def compile_graph(checkpointer=None):
    if checkpointer is None:
        checkpointer = InMemorySaver()
    return build_graph().compile(checkpointer=checkpointer)
