from typing import Any, Dict, List
from loguru import logger
from config import load_config

from langchain_openai import ChatOpenAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic

class HybridAIClient:
    """
    OpenAI, Gemini, Claude-3 すべてをLangChain経由で利用するクライアント
    モデル名に応じて適切なLangChain LLMを選択し、OpenAI互換のレスポンスを返す
    """
    def __init__(self):
        config = load_config()
        self.openai_api_key = config.openai_api_key
        self.gemini_api_key = config.gemini_api_key
        self.anthropic_api_key = config.anthropic_api_key
        self.openai_models = ["gpt-4o"]
        self.gemini_models = ["gemini-2.0-flash", "gemini-1.5-flash"]
        self.anthropic_models = ["claude-3-sonnet-20240229"]
        self.LOG_LEN = 40
        self.llms = {}
        if self.openai_api_key:
            self.llms["openai"] = ChatOpenAI(openai_api_key=self.openai_api_key)
        if self.gemini_api_key:
            self.llms["gemini"] = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=self.gemini_api_key)
        if self.anthropic_api_key:
            try:
                self.llms["anthropic"] = ChatAnthropic(
                    anthropic_api_key=self.anthropic_api_key,
                    model_name="claude-3-sonnet-20240229"
                )
            except ImportError:
                logger.warning("anthropicパッケージが見つからないためAnthropic LLMは無効化されます")
        self.chat = self
        self.completions = self

    def _select_llm(self, model: str):
        if model in self.openai_models:
            return self.llms["openai"], "openai"
        if model in self.gemini_models:
            return self.llms["gemini"], "gemini"
        if model in self.anthropic_models:
            return self.llms["anthropic"], "anthropic"
        raise ValueError(f"Unknown model: {model}")

    def create(self, model: str, messages: List[Dict[str, str]], **kwargs) -> Any:
        try:
            llm, provider = self._select_llm(model)
            for message in messages:
                log_msg = message.get("content", "")
                if isinstance(log_msg, str):
                    log_msg = log_msg[:self.LOG_LEN].replace("\n", " ")
                logger.info(f"LLM IN: {log_msg}")
            result = llm.invoke(messages)
            class Message:
                def __init__(self, content):
                    self.content = content
                    self.role = "assistant"
            class Choice:
                def __init__(self, message):
                    self.message = message
            class Response:
                def __init__(self, choices):
                    self.choices = choices
            content = result.content if hasattr(result, "content") else str(result)
            logger.info(f"LLM OUT: {content[:self.LOG_LEN]}")
            return Response([Choice(Message(content))])
        except Exception as err:
            logger.error(f"LangChain client error: {err}")
            raise

    def is_knowledge_insufficient(self, model: str, messages: List[Dict[str, str]]) -> bool:
        try:
            knowledge_check_messages = messages.copy()
            knowledge_check_messages.append({
                "role": "system",
                "content": (
                    "ユーザーの質問に対して、あなたの知識が不足していると感じる場合は、"
                    "'KNOWLEDGE_INSUFFICIENT'という単語を含めてください。"
                    "知識が十分な場合は通常通り回答してください。"
                )
            })
            response = self.create(model, knowledge_check_messages)
            content = response.choices[0].message.content.lower()
            has_insufficient_marker = "knowledge_insufficient" in content
            uncertain_words = ["わかりません", "知りません", "不明です", "情報がありません"]
            has_uncertain_phrases = any(word in content for word in uncertain_words)
            return has_insufficient_marker or has_uncertain_phrases
        except Exception as e:
            logger.exception(f"Error checking knowledge sufficiency: {e}")
            return True

def load_ai_client():
    return HybridAIClient()
