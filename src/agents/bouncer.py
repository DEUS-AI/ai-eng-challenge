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

