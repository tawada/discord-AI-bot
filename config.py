import dataclasses
import os


@dataclasses.dataclass
class ConfigTemplate:
    discord_api_key: str
    openai_api_key: str
    gemini_api_key: str
    anthropic_api_key: str
    target_channel_ids: list[int]
    role_prompt: str
    role_name: str


def load_config():
    return ConfigTemplate(
        # API keys
        discord_api_key=os.environ["DISCORD_API_KEY"],
        openai_api_key=os.environ["OPENAI_API_KEY"],
        gemini_api_key=os.environ["GEMINI_API_KEY"],
        anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
        # Target channel IDs
        target_channel_ids=list(map(int, os.environ["CHANNEL_IDS"].split(","))),
        # Roleplay settings
        role_prompt=os.environ["ROLE_PROMPT"],
        role_name=os.environ["ROLE_NAME"],
    )
