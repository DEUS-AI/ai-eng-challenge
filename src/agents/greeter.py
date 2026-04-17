from langchain.agents import create_agent
from agents.tools import verify_identity
from ai.llm import call_google_generative_ai_model
import yaml

with open("src/ai/prompt.yaml", "r") as f:
    prompts = yaml.safe_load(f)

GREETER_AGENT_PROMPT = prompts["GREETER_AGENT_PROMPT"]


model = call_google_generative_ai_model()

greeter_agent = create_agent(
    model,
    tools=[verify_identity],
    system_prompt=GREETER_AGENT_PROMPT,
)

query = "Send the design team a reminder about reviewing the new mockups"

for step in greeter_agent.stream(
    {"messages": [{"role": "user", "content": query}]}
):
    for update in step.values():
        for message in update.get("messages", []):
            message.pretty_print()