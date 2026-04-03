import app

from pydantic import ConfigDict, Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    discord_api_key: str
    target_channel_ids: str = ""
    model_config = ConfigDict(
        env_file=".env",
        extra="ignore",
    )

    def get_target_channel_ids(self) -> set[int]:
        if not self.target_channel_ids:
            return set()
        return {int(ch.strip()) for ch in self.target_channel_ids.split(",") if ch.strip()}


if __name__ == "__main__":
    settings = Settings()
    app.caller.run(
        api_key=settings.discord_api_key,
        target_channels=settings.get_target_channel_ids(),
    )
