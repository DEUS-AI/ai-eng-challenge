"""
State machine tests using LangGraph StateGraph.
Simulates conversation scenarios with predefined inputs via Command(resume=...).

Auth flow: provide 2-of-3 details (name, phone, iban) → secret question → answer → bouncer
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from datetime import datetime
from langgraph.types import Command

from graph import compile_graph


def run_scenario(name: str, auth_details: str, secret_answer: str, requests: list[str]) -> list[str]:
    print(f"\n{'=' * 60}")
    print(f"SCENARIO: {name}")
    print(f"{'=' * 60}")

    graph = compile_graph()
    config = {"configurable": {"thread_id": name}}
    shown = 0
    transcript: list[str] = [f"### {name}\n\n"]

    initial_state = {
        "user_details": "",
        "secret_question": "",
        "matched_nif": "",
        "secret_answer": "",
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
            transcript.append(f"**You**: {user_label}\n\n")
        graph.invoke(inputs, config=config)
        messages = graph.get_state(config).values.get("agent_messages", [])
        for msg in messages[shown:]:
            print(f"  Agent: {msg}")
            transcript.append(f"**Agent**: {msg}\n\n")
        shown = len(messages)

    invoke_and_print(initial_state)

    if graph.get_state(config).next:
        invoke_and_print(Command(resume=auth_details), user_label=auth_details)

    if graph.get_state(config).next:
        next_node = graph.get_state(config).next[0]
        if next_node == "secret_input":
            invoke_and_print(Command(resume=secret_answer), user_label=secret_answer or "(empty)")

    for req in requests:
        if not graph.get_state(config).next:
            break
        invoke_and_print(Command(resume=req), user_label=req)

    if graph.get_state(config).next:
        invoke_and_print(Command(resume="exit"), user_label="exit")

    print(f"\n--- End of scenario: {name} ---")
    transcript.append("\n---\n\n")
    return transcript


def save_report(all_lines: list[str]) -> None:
    log_dir = os.path.join(os.path.dirname(__file__), "../logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(log_dir, f"test_run_{timestamp}.md")
    header = [
        f"# Test Run — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n",
        "Auth flow: 2-of-3 details (name, phone, IBAN) → secret question → service request\n\n",
        "---\n\n",
    ]
    with open(path, "w") as f:
        f.writelines(header + all_lines)
    print(f"\nReport saved to {path}")


if __name__ == "__main__":
    all_lines: list[str] = []

    all_lines += run_scenario(
        name="scenario-lisa-name-phone",
        auth_details="Name: Lisa, Phone: +1122334455",
        secret_answer="Yoda",
        requests=["I want to check my account balance"],
    )

    all_lines += run_scenario(
        name="scenario-carlos-name-iban",
        auth_details="Name: Carlos, IBAN: PT50000201231234567890154",
        secret_answer="Silva",
        requests=["I'd like to talk about my investment portfolio"],
    )

    all_lines += run_scenario(
        name="scenario-john-no-account",
        auth_details="Name: John, Phone: +447911123456, IBAN: GB29NWBK60161331926819",
        secret_answer="Greenwood",
        requests=[],
    )

    all_lines += run_scenario(
        name="scenario-wrong-details",
        auth_details="Name: Unknown, Phone: +0000000000",
        secret_answer="",
        requests=[],
    )

    all_lines += run_scenario(
        name="scenario-lisa-wrong-secret",
        auth_details="Name: Lisa, IBAN: DE89370400440532013000",
        secret_answer="WrongAnswer",
        requests=[],
    )

    # Account issue — premium client (Carlos)
    all_lines += run_scenario(
        name="scenario-carlos-account-issue",
        auth_details="Name: Carlos, IBAN: PT50000201231234567890154",
        secret_answer="Silva",
        requests=["I'm having an issue with my account"],
    )

    # Account issue — regular client (Lisa)
    all_lines += run_scenario(
        name="scenario-lisa-account-issue",
        auth_details="Name: Lisa, Phone: +1122334455",
        secret_answer="Yoda",
        requests=["I'm having an issue with my account"],
    )

    # Account issue — non-client (John, no account)
    all_lines += run_scenario(
        name="scenario-john-account-issue",
        auth_details="Name: John, Phone: +447911123456, IBAN: GB29NWBK60161331926819",
        secret_answer="Greenwood",
        requests=["I'm having an issue with my account"],
    )

    # Memory: ask about previous request in same session
    all_lines += run_scenario(
        name="scenario-memory-what-did-i-ask",
        auth_details="Name: Lisa, Phone: +1122334455",
        secret_answer="Yoda",
        requests=[
            "I want to check my account balance",
            "What was my previous request?",
        ],
    )

    # Memory: ask about the agent response
    all_lines += run_scenario(
        name="scenario-memory-what-did-agent-say",
        auth_details="Name: Carlos, IBAN: PT50000201231234567890154",
        secret_answer="Silva",
        requests=[
            "I'd like to talk about my investment portfolio",
            "What did you tell me just now?",
        ],
    )

    save_report(all_lines)

    print(f"\n{'=' * 60}")
    print("All scenarios completed.")
    print(f"{'=' * 60}\n")
