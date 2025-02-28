import logging
from typing import Any, Dict, List

import openai
from anthropic import Anthropic

from config import load_config

logger = logging.getLogger(__name__)


class HybridAIClient:
    """OpenAI、Gemini、Claude-3を使うクライアント

    openai_client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
    ).choices[0].message.content
    のmodelによってopenai_client、gemini_client、anthropic_clientを使い分ける
    """

    def __init__(self, openai_client, gemini_client, anthropic_client=None):
        self.openai_client = openai_client
        self.gemini_client = gemini_client
        self.anthropic_client = anthropic_client
        self.openai_models = ["gpt-4o"]
        self.anthropic_models = ["claude-3-sonnet-20240229"]
        self.chat = self
        self.completions = self

    def _convert_messages_for_anthropic(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Convert OpenAI message format to Anthropic format"""
        role_mapping = {
            "system": "system",
            "user": "user",
            "assistant": "assistant"
        }
        return [
            {"role": role_mapping[msg["role"]], "content": msg["content"]}
            for msg in messages
        ]

    def create(self, model: str, messages: List[Dict[str, str]]) -> Any:
        if model in self.openai_models:
            return self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
            )
        elif model in self.anthropic_models and self.anthropic_client:
            try:
                anthropic_messages = self._convert_messages_for_anthropic(messages)
                system_message = next((msg["content"] for msg in messages if msg["role"] == "system"), None)
                response = self.anthropic_client.messages.create(
                    model=model,
                    max_tokens=1024,
                    messages=anthropic_messages,
                    system=system_message
                )
                # Convert Anthropic response to OpenAI format
                return type('AnthropicResponse', (), {
                    'choices': [{
                        'message': type('Message', (), {
                            'content': response.content[0].text,
                            'role': 'assistant'
                        })
                    }]
                })
            except Exception as err:
                logger.error(f"anthropic_client error: {err}")
                return self.openai_client.chat.completions.create(
                    model=self.openai_models[0],
                    messages=messages,
                )
        else:
            try:
                return self.gemini_client.chat.completions.create(
                    model=model,
                    messages=messages,
                )
            except Exception as err:
                logger.error(f"gemini_client error: {err}")
                return self.openai_client.chat.completions.create(
                    model=self.openai_models[0],
                    messages=messages,
                )

    def is_knowledge_insufficient(self, model, messages):
        try:
            response = self.create(model, messages)
            # Assuming the model returns a confidence score or similar metric
            confidence_score = getattr(response.choices[0].message, 'confidence', 1.0)
            return confidence_score < 0.5  # Threshold for insufficient knowledge
        except Exception as e:
            logger.exception(e)
            return True


def load_ai_client():
    config = load_config()
    gemini_client = openai.OpenAI(
        api_key=config.gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    openai_client = openai.OpenAI(api_key=config.openai_api_key)
    anthropic_client = Anthropic(api_key=config.anthropic_api_key) if config.anthropic_api_key else None
    return HybridAIClient(openai_client, gemini_client, anthropic_client)
