from langchain.agents import create_agent
from agents.tools import verify_identity
from ai.llm import call_google_generative_ai_model
from utils.get_prompts import get_prompt



GREETER_AGENT_PROMPT = get_prompt("GREETER_AGENT_PROMPT")


model = call_google_generative_ai_model()

greeter_agent = create_agent(
    model,
    tools=[verify_identity],
    system_prompt=GREETER_AGENT_PROMPT,
)

if __name__ == "__main__":
    query = "Hello, I need some help with my account."

    for step in greeter_agent.stream(
        {"messages": [{"role": "user", "content": query}]}
    ):
        for update in step.values():
            for message in update.get("messages", []):
                message.pretty_print()