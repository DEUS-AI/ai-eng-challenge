from pathlib import Path
import yaml

_PROMPT_FILE = Path(__file__).parent.parent / "ai" / "prompt.yaml"


def get_prompt(agent: str):
    with open(_PROMPT_FILE, "r") as f:
        prompts = yaml.safe_load(f)
    return prompts.get(agent)