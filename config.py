from pydantic import Field
from pydantic_settings import BaseSettings
import os


class Settings(BaseSettings):
    discord_api_key: str
    openai_api_key: str
    gemini_api_key: str
    anthropic_api_key: str
    text_model: str | None = None
    target_channel_ids: list[int] = Field(
        ..., alias="CHANNEL_IDS", description="Comma-separated list of channel IDs"
    )
    role_prompt: str
    role_name: str

    class Config:
        env_file = ".env"
        extra = "ignore"


def load_config():
    return Settings()
