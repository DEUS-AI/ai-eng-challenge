from langchain.agents import create_agent
from agents.tools import verify_account_type
from ai.llm import call_google_generative_ai_model
from utils.get_prompts import get_prompt

BOUNCER_AGENT_PROMPT = get_prompt("BOUNCER_AGENT_PROMPT")

model = call_google_generative_ai_model()


bouncer_agent = create_agent(
    model,
    tools=[verify_account_type],
    system_prompt=BOUNCER_AGENT_PROMPT,
)


if __name__ == "__main__":
    scenarios = [
        {
            "label": "Scenario 1 - Premium customer (Carlos, NIF 234567890)",
            "message": "Hi, I'd like to know what services are available to me. My NIF is 234567890.",
        },
        {
            "label": "Scenario 2 - Regular customer (Lisa, NIF 123456789)",
            "message": "Hello, I want to check my account details. My NIF is 123456789.",
        },
        {
            "label": "Scenario 3 - Not a customer (John, NIF 345678901)",
            "message": "Hello, I want to check my account. My NIF is 345678901.",
        },
    ]

    for scenario in scenarios:
        print(f"\n{'='*60}\n{scenario['label']}\n{'='*60}")
        for step in bouncer_agent.stream(
            {"messages": [{"role": "user", "content": scenario["message"]}]}
        ):
            for update in step.values():
                for message in update.get("messages", []):
                    message.pretty_print()

