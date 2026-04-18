"""
State machine tests using LangGraph StateGraph.
Simulates 3 conversation scenarios with predefined inputs via Command(resume=...).
"""
from langgraph.types import Command

from graph import compile_graph


def run_scenario(name: str, nif: str, requests: list[str]) -> None:
    print(f"\n{'=' * 60}")
    print(f"SCENARIO: {name}")
    print(f"{'=' * 60}")

    graph = compile_graph()
    config = {"configurable": {"thread_id": name}}
    shown = 0  # index of last displayed agent_message

    initial_state = {
        "nif": "",
        "identity_verified": False,
        "account_type": "",
        "user_request": "",
        "agent_messages": [],
        "log_lines": [],
    }

    def invoke_and_print(inputs, user_label: str | None = None) -> None:
        nonlocal shown
        if user_label is not None:
            print(f"  You: {user_label}")
        graph.invoke(inputs, config=config)
        messages = graph.get_state(config).values.get("agent_messages", [])
        for msg in messages[shown:]:
            print(f"  Agent: {msg}")
        shown = len(messages)

    # Start — runs greeting, pauses at nif_input interrupt
    invoke_and_print(initial_state)

    # Provide NIF
    if graph.get_state(config).next:
        invoke_and_print(Command(resume=nif), user_label=nif)

    # Provide service requests
    for req in requests:
        if not graph.get_state(config).next:
            break
        invoke_and_print(Command(resume=req), user_label=req)

    # End session if still running
    if graph.get_state(config).next:
        invoke_and_print(Command(resume="exit"), user_label="exit")

    print(f"\n--- End of scenario: {name} ---")


if __name__ == "__main__":
    # 1. Regular customer asks about her account
    run_scenario(
        name="scenario-lisa-accounts",
        nif="123456789",
        requests=["Quero saber o saldo da minha conta"],
    )

    # 2. Premium customer asks about investments
    run_scenario(
        name="scenario-carlos-investments",
        nif="234567890",
        requests=["Gostava de falar sobre os meus investimentos em ações"],
    )

    # 3. Non-customer — rejected after NIF
    run_scenario(
        name="scenario-john-not-customer",
        nif="345678901",
        requests=[],
    )

    # 4. Invalid NIF format — should loop back and ask again
    run_scenario(
        name="scenario-invalid-nif",
        nif="olá",
        requests=["123456789", "Quero saber o saldo da minha conta"],
    )

    print(f"\n{'=' * 60}")
    print("All scenarios completed.")
    print(f"{'=' * 60}\n")
