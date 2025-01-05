import logging

import openai

from config import load_config

logger = logging.getLogger(__name__)


class HybridAIClient:
    """OpenAI と Gemini の両方を使うクライアント

    openai_client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
    ).choices[0].message.content
    のmodelによってopenai_clientとgemini_clientを使い分ける
    """

    def __init__(self, openai_client, gemini_client):
        self.openai_client = openai_client
        self.gemini_client = gemini_client
        self.openai_models = ["gpt-4o"]
        self.chat = self
        self.completions = self

    def create(self, model, messages):
        if model in self.openai_models:
            return self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
            )
        else:
            try:
                return self.gemini_client.chat.completions.create(
                    model=model,
                    messages=messages,
                )
            except Exception as e:
                logger.error(f"gemini_client error: {e}")
            return self.openai_client.chat.completions.create(
                # Gemini がうまくいかないので一旦gpt-4oにしている
                model=self.openai_models[0],
                messages=messages,
            )


def load_ai_client():
    config = load_config()
    gemini_client = openai.OpenAI(
        api_key=config.gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    openai_client = openai.OpenAI(api_key=config.openai_api_key)
    return HybridAIClient(openai_client, gemini_client)
