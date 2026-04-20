import os
from datetime import datetime
from uuid import uuid4

from langgraph.types import Command

from graph import compile_graph


def save_log(log_lines: list[str]) -> None:
    log_dir = os.path.join(os.path.dirname(__file__), "../logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))
    print(f"\nConversation saved to {log_path}")


def chat() -> None:
    graph = compile_graph()
    config = {"configurable": {"thread_id": str(uuid4())}}
    shown = 0

    print("\n" + "=" * 60)
    print("Welcome to DEUS Bank Customer Support")
    print("Type 'exit' to quit at any time.")
    print("=" * 60 + "\n")

    initial_state = {
        "user_details": "",
        "secret_question": "",
        "matched_nif": "",
        "secret_answer": "",
        "identity_verified": False,
        "account_type": "",
        "user_request": "",
        "agent_messages": [],
        "log_lines": [f"# Chat Session — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"],
    }

    def invoke_and_display(inputs) -> None:
        nonlocal shown
        graph.invoke(inputs, config=config)
        messages = graph.get_state(config).values.get("agent_messages", [])
        for msg in messages[shown:]:
            print(f"Agent: {msg}\n")
        shown = len(messages)

    try:
        invoke_and_display(initial_state)

        while graph.get_state(config).next:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            invoke_and_display(Command(resume=user_input))

    except KeyboardInterrupt:
        print("\nSession interrupted.")

    finally:
        state = graph.get_state(config).values
        save_log(state.get("log_lines", []))


if __name__ == "__main__":
    chat()
