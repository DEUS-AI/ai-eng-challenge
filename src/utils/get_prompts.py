import yaml


def get_prompt(agent:str):
    with open("src/ai/prompt.yaml", "r") as f:
        prompts = yaml.safe_load(f)
    return prompts.get(agent)