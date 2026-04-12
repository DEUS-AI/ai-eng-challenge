"""LangGraph builder — wires agents, guardrails, and edges together."""

from __future__ import annotations

import sqlite3
from functools import partial
from typing import Literal

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.agents.bouncer import bouncer_node
from src.agents.greeter import greeter_node
from src.agents.specialist import specialist_node
from src.guardrails import input_guard_node, output_policy_node
from src.schemas import ConversationState


def _route_after_input_guard(state: ConversationState) -> Literal["greeter", "bouncer", "specialist", "__end__"]:
    """Route based on whether the input guard blocked the message and current phase."""
    if state.get("input_blocked", False):
        return "__end__"

    phase = state.get("phase", "greeting")
    if phase == "greeting":
        return "greeter"
    elif phase == "verification":
        return "bouncer"
    elif phase == "routing":
        return "specialist"
    else:
        return "__end__"


def build_graph(
    db_conn: sqlite3.Connection,
    model_name: str = "gemini-2.5-flash",
    api_key: str | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> StateGraph:
    """Build and compile the agent graph."""
    llm_kwargs = {"model": model_name, "temperature": 0}
    if api_key:
        llm_kwargs["google_api_key"] = api_key
    llm = ChatGoogleGenerativeAI(**llm_kwargs)

    # Bind dependencies to node functions
    greeter = partial(greeter_node, llm=llm)
    bouncer = partial(bouncer_node, llm=llm, db_conn=db_conn)
    specialist = partial(specialist_node, llm=llm)

    builder = StateGraph(ConversationState)

    # Add nodes
    builder.add_node("input_guard", input_guard_node)
    builder.add_node("greeter", greeter)
    builder.add_node("bouncer", bouncer)
    builder.add_node("specialist", specialist)
    builder.add_node("output_policy", output_policy_node)

    # Entry: every message goes through input guard first
    builder.add_edge(START, "input_guard")
    builder.add_conditional_edges("input_guard", _route_after_input_guard)

    # Exit: output policy always leads to end
    # (Agents route to output_policy via Command before finishing)
    builder.add_edge("output_policy", END)

    # Greeter, bouncer, specialist route themselves via Command:
    # - greeter -> bouncer (when all collected) or greeter (loop)
    # - bouncer -> specialist (verified) or output_policy (non-client/failed) or bouncer (retry)
    # - specialist -> output_policy (always)

    # Compile with checkpointer for session persistence
    # Defaults to in-memory; use SqliteSaver for persistence across restarts
    return builder.compile(checkpointer=checkpointer or MemorySaver())
