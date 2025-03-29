from typing import Any, Callable, Dict, List, Tuple

import openai
from anthropic import Anthropic
from loguru import logger

from config import load_config


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

    def _convert_messages_for_anthropic(
        self, messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Convert OpenAI message format to Anthropic format.
        Excludes system messages as they are handled separately."""
        role_mapping = {"user": "user", "assistant": "assistant"}
        return [
            {"role": role_mapping[msg["role"]], "content": msg["content"]}
            for msg in messages
            if msg["role"] != "system"
        ]

    def _create_with_openai(self, model: str, messages: List[Dict[str, str]]) -> Any:
        """OpenAIのモデルを使用してレスポンスを生成"""
        return self.openai_client.chat.completions.create(
            model=model,
            messages=messages,
        )

    def _create_with_anthropic(self, model: str, messages: List[Dict[str, str]]) -> Any:
        """Anthropicのモデルを使用してレスポンスを生成"""
        anthropic_messages = self._convert_messages_for_anthropic(messages)
        system_message = next(
            (msg["content"] for msg in messages if msg["role"] == "system"), None
        )
        response = self.anthropic_client.messages.create(
            model=model,
            max_tokens=1024,
            messages=anthropic_messages,
            system=system_message,
        )
        return self._convert_anthropic_response(response)

    def _create_with_gemini(self, model: str, messages: List[Dict[str, str]]) -> Any:
        """Geminiのモデルを使用してレスポンスを生成"""
        return self.gemini_client.chat.completions.create(
            model=model,
            messages=messages,
        )

    def _convert_anthropic_response(self, response: Any) -> Any:
        """AnthropicのレスポンスをOpenAI形式に変換"""
        message = type(
            "Message", (), {"content": response.content[0].text, "role": "assistant"}
        )
        choice = type("Choice", (), {"message": message})
        return type("AnthropicResponse", (), {"choices": [choice]})

    def _get_fallback_response(self, messages: List[Dict[str, str]]) -> Any:
        """フォールバック用のレスポンスを生成"""
        return self._create_with_openai(self.openai_models[0], messages)

    def _select_model_handler(
        self, model: str
    ) -> Tuple[Callable[[str, List[Dict[str, str]]], Any], str]:
        """モデルに応じたハンドラーを選択"""
        if model in self.openai_models:
            return self._create_with_openai, "openai"
        if model in self.anthropic_models and self.anthropic_client:
            return self._create_with_anthropic, "anthropic"
        return self._create_with_gemini, "gemini"

    def _execute_with_fallback(
        self,
        handler: Callable[[str, List[Dict[str, str]]], Any],
        model: str,
        messages: List[Dict[str, str]],
        provider: str,
    ) -> Any:
        """ハンドラーを実行し、失敗時はフォールバック"""
        try:
            return handler(model, messages)
        except Exception as err:
            logger.error(f"{provider}_client error: {err}")
            return self._get_fallback_response(messages)

    def create(self, model: str, messages: List[Dict[str, str]]) -> Any:
        """指定されたモデルを使用してレスポンスを生成

        各モデルでエラーが発生した場合は、OpenAIのモデルにフォールバックする
        """
        try:
            LOG_LEN = 20
            for message in messages:
                log_msg = message["content"][:LOG_LEN]
                if isinstance(log_msg, str):
                    log_msg = log_msg.replace("\n", " ")
                logger.info(f"LLM IN: {log_msg}")
            handler, provider = self._select_model_handler(model)
            response = self._execute_with_fallback(handler, model, messages, provider)
            # subscriptableか判定
            if response.choices[0].message.content is not None:
                logger.info(f"LLM OUT: {response.choices[0].message.content[:LOG_LEN]}")
            return response
        except Exception as err:
            logger.error(f"Unexpected error in create: {err}")
            return self._get_fallback_response(messages)

    def is_knowledge_insufficient(self, model, messages):
        try:
            response = self.create(model, messages)
            # Assuming the model returns a confidence score or similar metric
            confidence_score = getattr(response.choices[0].message, "confidence", 1.0)
            return confidence_score < 0.5  # Threshold for insufficient knowledge
        except Exception as e:
            logger.exception(e)
            return True


def load_ai_client():
    config = load_config()
    gemini_client = openai.OpenAI(
        api_key=config.gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    openai_client = openai.OpenAI(api_key=config.openai_api_key)
    anthropic_client = (
        Anthropic(api_key=config.anthropic_api_key)
        if config.anthropic_api_key
        else None
    )
    return HybridAIClient(openai_client, gemini_client, anthropic_client)
