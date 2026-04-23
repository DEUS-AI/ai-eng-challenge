from langchain.agents import create_agent
from agents.tools import delegate_hitl, get_account_field, get_available_specialists, log_complaint
from ai.llm import call_google_generative_ai_model
from utils.get_prompts import get_prompt
from config.models import Skill, SKILL_METADATA


skills_detail = "\n".join([
    f"- {s.value}: {SKILL_METADATA[s]['description']} Related requests: {', '.join(SKILL_METADATA[s]['related_requests'])}."
    for s in Skill
])
SPECIALIST_AGENT_PROMPT = get_prompt("SPECIALIST_AGENT_PROMPT").format(skills=skills_detail)


model = call_google_generative_ai_model()

specialist_agent = create_agent(
    model,
    tools=[delegate_hitl, get_account_field, get_available_specialists, log_complaint],
    system_prompt=SPECIALIST_AGENT_PROMPT,
)

if __name__ == "__main__":
    scenarios = [
        "I need help with my insurance policy.",
        "I have questions about my investment portfolio.",
        "I need assistance with my account.",
    ]

    for query in scenarios:
        print(f"\n{'='*60}\nQuery: {query}\n{'='*60}")
        for step in specialist_agent.stream(
            {"messages": [{"role": "user", "content": query}]}
        ):
            for update in step.values():
                for message in update.get("messages", []):
                    message.pretty_print()