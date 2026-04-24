from pathlib import Path
import yaml

_PROMPT_FILE = Path(__file__).parent.parent / "ai" / "prompt.yaml"


def get_prompt(agent: str):
    """Load prompt.yaml and return the named prompt string, or None if the key is not found."""
    with open(_PROMPT_FILE, "r") as f:
        prompts = yaml.safe_load(f)
    return prompts.get(agent)