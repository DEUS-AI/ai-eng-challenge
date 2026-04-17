import getpass
import os
from langchain.tools import tool
from langchain.agents import create_agent

from agents.bouncer import calendar_agent
from agents.greeter import greeter_agent
from ai.llm import call_google_generative_ai_model


os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")


@tool
def schedule_event(request: str) -> str:
    """Schedule calendar events using natural language.

    Use this when the user wants to create, modify, or check calendar appointments.
    Handles date/time parsing, availability checking, and event creation.

    Input: Natural language scheduling request (e.g., 'meeting with design team
    next Tuesday at 2pm')
    """
    result = calendar_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].text


@tool
def manage_email(request: str) -> str:
    """Send emails using natural language.

    Use this when the user wants to send notifications, reminders, or any email
    communication. Handles recipient extraction, subject generation, and email
    composition.

    Input: Natural language email request (e.g., 'send them a reminder about
    the meeting')
    """
    result = email_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].text


@tool
def manage_greetings(request: str) -> str:
    """Generate greetings using natural language.

    Use this when the user wants to greet customers or clients. Handles greeting
    composition and identity verification.

    Input: Natural language greeting request (e.g., 'greet the customer and ask for
    their ID')
    """
    result = greeter_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].text

SUPERVISOR_PROMPT = (
    "You are a supervisor that routes customer requests to specialized agents. "
    "You MUST always delegate to the appropriate agent — never respond directly to the user. "
    "For any greeting, introduction, or identity-related message, you MUST call manage_greetings. "
    "Do not generate any response yourself."
)

model = call_google_generative_ai_model()

supervisor_agent = create_agent(
    model,
    tools=[manage_greetings],
    system_prompt=SUPERVISOR_PROMPT,
)

scenarios = [
    {
        "label": "Scenario 1 - Existing NIF (Lisa, regular customer)",
        "messages": [
            {"role": "user", "content": "Hello, I need some help."},
            {"role": "user", "content": "My NIF is 123456789."},
        ],
    },
    {
        "label": "Scenario 2 - Non-existing NIF (John, not a customer)",
        "messages": [
            {"role": "user", "content": "Hello, I need some help."},
            {"role": "user", "content": "My NIF is 345678901."},
        ],
    },
]

output_lines = []

for scenario in scenarios:
    header = f"\n{'='*60}\n{scenario['label']}\n{'='*60}"
    print(header)
    output_lines.append(header)

    history = []
    for user_msg in scenario["messages"]:
        history.append(user_msg)
        turn_label = f"\n> User: {user_msg['content']}"
        print(turn_label)
        output_lines.append(turn_label)

        for step in supervisor_agent.stream({"messages": history}):
            for update in step.values():
                for message in update.get("messages", []):
                    message.pretty_print()
                    output_lines.append(f"\n**{message.type}**: {message.content}")
                    history.append(message)

output_path = os.path.join(os.path.dirname(__file__), "../../output.md")
with open(output_path, "w") as f:
    f.write("# Supervisor Agent Output\n")
    f.write("\n".join(output_lines))