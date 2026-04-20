"""
StateGraph definition for DEUS Bank customer support.

Flow:
  START → greeting → nif_input*
                        ├─ invalid format  → nif_invalid → nif_input* (loop)
                        └─ 9-digit NIF    → greeter_verify [verify_identity]
                                                ├─ not found → reject → END
                                                └─ verified  → bouncer [verify_account_type]
                                                                  ├─ not found → reject → END
                                                                  └─ found     → request_welcome → request_input*
                                                                                                    ├─ exit    → farewell → END
                                                                                                    └─ request → specialist → request_input* (loop)
* = interrupt (waits for human input)
"""
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

_llm = call_google_generative_ai_model()

def _llm_respond(prompt_key: str) -> str:
    prompt = get_prompt(prompt_key)
    result = _llm.invoke(prompt)
    return result.content.strip()

# ── Shared state ─────────────────────────────────────────────────────────────

class ChatState(TypedDict):
    nif: str
    identity_verified: bool
    account_type: str
    user_request: str
    agent_messages: Annotated[list[str], operator.add]   # every agent response, accumulated
    log_lines: Annotated[list[str], operator.add]        # full conversation log


# ── Helper ────────────────────────────────────────────────────────────────────

def _call_agent(agent, message: str) -> str:
    result = agent.invoke({"messages": [{"role": "user", "content": message}]})
    for msg in reversed(result["messages"]):
        if getattr(msg, "content", None):
            return msg.content
    return ""


# ── Nodes ─────────────────────────────────────────────────────────────────────

def greeting(state: ChatState) -> dict:
    response = _call_agent(greeter_agent, "Greet the customer and ask for their NIF.")
    return {
        "agent_messages": [response],
        "log_lines": [f"\n**Greeter Agent**: {response}"],
    }


def nif_input(state: ChatState) -> dict:
    nif = interrupt("Waiting for NIF")
    return {
        "nif": nif.strip(),
        "log_lines": [f"\n**You**: {nif}"],
    }


def nif_invalid(state: ChatState) -> dict:
    msg = _llm_respond("NIF_INVALID_PROMPT")
    return {
        "nif": "",
        "agent_messages": [msg],
        "log_lines": [f"\n**Greeter Agent**: {msg}"],
    }


def greeter_verify(state: ChatState) -> dict:
    """Runs greeter_agent, parses ToolMessage into IdentityResult."""
    result = greeter_agent.invoke({
        "messages": [{
            "role": "user",
            "content": f"The customer provided NIF {state['nif']}. Please verify their identity.",
        }]
    })
    tool_content = next(
        (m.content for m in result["messages"] if type(m).__name__ == "ToolMessage"),
        "",
    )
    parsed = IdentityResult(verified=tool_content == "Identity verified successfully")
    return {
        "identity_verified": parsed.verified,
        "log_lines": [f"\n**[internal greeter_verify]**: {parsed.model_dump()}"],
    }


def bouncer(state: ChatState) -> dict:
    """Runs bouncer_agent, parses ToolMessage into AccountResult."""
    result = bouncer_agent.invoke({
        "messages": [{"role": "user", "content": state["nif"]}]
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


def specialist(state: ChatState) -> dict:
    """Runs specialist_agent, parses ToolMessage into SpecialistDecision."""
    result = specialist_agent.invoke({
        "messages": [{"role": "user", "content": state["user_request"]}]
    })
    tool_msg = next(
        (m for m in result["messages"] if type(m).__name__ == "ToolMessage"),
        None,
    )
    if tool_msg:
        # Agent called delegate_hitl — extract skill from the preceding AIMessage tool_call
        ai_msg = next(
            (m for m in result["messages"]
             if type(m).__name__ == "AIMessage" and getattr(m, "tool_calls", [])),
            None,
        )
        skill_value = ai_msg.tool_calls[0]["args"].get("skill") if ai_msg else None
        try:
            skill = Skill(skill_value)
        except (ValueError, TypeError):
            skill = None
        parsed = SpecialistDecision(in_scope=True, skill=skill)
        employee_name = tool_msg.content
        if employee_name == "no_employee_available":
            response = "No specialist is currently available. Your request has been escalated."
        else:
            response = f"{employee_name} will attend your request, I will pass the request."
        return {
            "agent_messages": [response],
            "log_lines": [f"\n**Specialist Agent**: {response}"],
        }
    # No tool called — agent refused the request
    refusal = next(
        (m.content for m in reversed(result["messages"])
         if getattr(m, "content", None)),
        "I can only assist you with your own banking services.",
    )
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

def route_after_nif_input(state: ChatState) -> str:
    nif = state.get("nif", "")
    if nif.isdigit() and len(nif) == 9:
        return "greeter_verify"
    return "nif_invalid"


def route_after_greeter(state: ChatState) -> str:
    if state.get("identity_verified"):
        return "bouncer"
    return "reject"


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
    builder.add_node("nif_input", nif_input)
    builder.add_node("nif_invalid", nif_invalid)
    builder.add_node("greeter_verify", greeter_verify)
    builder.add_node("bouncer", bouncer)
    builder.add_node("reject", reject)
    builder.add_node("request_welcome", request_welcome)
    builder.add_node("request_input", request_input)
    builder.add_node("specialist", specialist)
    builder.add_node("farewell", farewell)

    builder.add_edge(START, "greeting")
    builder.add_edge("greeting", "nif_input")
    builder.add_conditional_edges(
        "nif_input",
        route_after_nif_input,
        {"greeter_verify": "greeter_verify", "nif_invalid": "nif_invalid"},
    )
    builder.add_conditional_edges(
        "greeter_verify",
        route_after_greeter,
        {"bouncer": "bouncer", "reject": "reject"},
    )
    builder.add_edge("nif_invalid", "nif_input")
    builder.add_conditional_edges(
        "bouncer",
        route_after_bouncer,
        {"reject": "reject", "request_welcome": "request_welcome"},
    )
    builder.add_edge("reject", END)
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
    return build_graph().compile(checkpointer=checkpointer or InMemorySaver())
