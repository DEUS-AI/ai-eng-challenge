import getpass
import os
from langchain.tools import tool
from langchain.agents import create_agent

from agents.bouncer import bouncer_agent
from agents.greeter import greeter_agent
from agents.specialist import specialist_agent
from ai.llm import call_google_generative_ai_model
from utils.get_prompts import get_prompt


os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")



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


@tool 
def manage_bouncer(request:str) -> str:
    """Handle account type verifications.

    Use this to validate de account type of the user. 
    After the greetings and user validation, validade the account type.

    Input: Natural language conversation and user asking for something. 
    """
    result = bouncer_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].text

@tool
def manage_specialist(request: str) -> str:
    """Handle specialized customer requests.

    Use this to route requests to the appropriate specialist based on the required skill.

    Input: Natural language conversation and user asking for something.
    """
    result = specialist_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].text


SUPERVISOR_PROMPT = get_prompt("SUPERVISOR_PROMPT")

model = call_google_generative_ai_model()

supervisor_agent = create_agent(
    model,
    tools=[manage_greetings, manage_bouncer, manage_specialist],
    system_prompt=SUPERVISOR_PROMPT,
)

if __name__ == "__main__":
    scenarios = [
        {
            "label": "Scenario 1 - Regular customer (Lisa) asking about her account",
            "messages": [
                {"role": "user", "content": "Hello, I need some help."},
                {"role": "user", "content": "My NIF is 123456789."},
                {"role": "user", "content": "I want to check my account balance and recent transactions."},
            ],
        },
        {
            "label": "Scenario 2 - Premium customer (Carlos) asking about investments",
            "messages": [
                {"role": "user", "content": "Hi there, I need assistance."},
                {"role": "user", "content": "My NIF is 234567890."},
                {"role": "user", "content": "I'd like to review my investment portfolio and explore new fund options."},
            ],
        },
        {
            "label": "Scenario 3 - Not a customer (John) asking about insurance",
            "messages": [
                {"role": "user", "content": "Good morning, I need help with something."},
                {"role": "user", "content": "My NIF is 345678901."},
            ],
        },
        {
            "label": "Scenario 4 - Impatient customer (Lisa) demanding help immediately",
            "messages": [
                {"role": "user", "content": "I need to know my account balance RIGHT NOW. My NIF is 123456789. I've been waiting too long, just tell me!"},
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
                        if hasattr(message, "tool_calls") and message.tool_calls:
                            tools_called = ", ".join([tc["name"] for tc in message.tool_calls])
                            output_lines.append(f"\n**supervisor** → calls: `{tools_called}`")
                        elif message.type == "tool":
                            tool_name = getattr(message, "name", "tool")
                            output_lines.append(f"\n**{tool_name}** response: {message.content}")
                        else:
                            output_lines.append(f"\n**{message.type}**: {message.content}")
                        history.append(message)

    output_path = os.path.join(os.path.dirname(__file__), "../../output.md")
    with open(output_path, "w") as f:
        f.write("# Supervisor Agent Output\n")
        f.write("\n".join(output_lines))