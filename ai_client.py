from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod

import openai
from anthropic import Anthropic
from loguru import logger

from config import load_config


class BaseAIClient(ABC):
    """AI APIクライアントの基底クラス"""
    
    @abstractmethod
    def create(self, model: str, messages: List[Dict[str, str]]) -> Any:
        """指定モデルを使用してレスポンスを生成"""
        pass


class OpenAIClient(BaseAIClient):
    """OpenAIのAPIを使うクライアント"""
    
    def __init__(self, client):
        self.client = client
        self.models = ["gpt-4o"]
        
    def create(self, model: str, messages: List[Dict[str, str]]) -> Any:
        """OpenAIのモデルを使用してレスポンスを生成"""
        try:
            return self.client.chat.completions.create(
                model=model,
                messages=messages,
            )
        except Exception as err:
            logger.error(f"OpenAI client error: {err}")
            raise


class AnthropicClient(BaseAIClient):
    """AnthropicのAPIを使うクライアント"""
    
    def __init__(self, client):
        self.client = client
        self.models = ["claude-3-sonnet-20240229"]
        
    def _convert_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """OpenAI形式のメッセージをAnthropic形式に変換"""
        role_mapping = {"user": "user", "assistant": "assistant"}
        return [
            {"role": role_mapping[msg["role"]], "content": msg["content"]}
            for msg in messages
            if msg["role"] != "system"
        ]
    
    def _convert_response(self, response: Any) -> Any:
        """AnthropicのレスポンスをOpenAI形式に変換"""
        message = type(
            "Message", (), {"content": response.content[0].text, "role": "assistant"}
        )
        choice = type("Choice", (), {"message": message})
        return type("AnthropicResponse", (), {"choices": [choice]})
    
    def create(self, model: str, messages: List[Dict[str, str]]) -> Any:
        """Anthropicのモデルを使用してレスポンスを生成"""
        try:
            anthropic_messages = self._convert_messages(messages)
            system_message = next(
                (msg["content"] for msg in messages if msg["role"] == "system"), None
            )
            response = self.client.messages.create(
                model=model,
                max_tokens=1024,
                messages=anthropic_messages,
                system=system_message,
            )
            return self._convert_response(response)
        except Exception as err:
            logger.error(f"Anthropic client error: {err}")
            raise


class GeminiClient(BaseAIClient):
    """GoogleのGemini APIを使うクライアント"""
    
    def __init__(self, client):
        self.client = client
        
    def create(self, model: str, messages: List[Dict[str, str]]) -> Any:
        """Geminiのモデルを使用してレスポンスを生成"""
        try:
            return self.client.chat.completions.create(
                model=model,
                messages=messages,
            )
        except Exception as err:
            logger.error(f"Gemini client error: {err}")
            raise


class HybridAIClient:
    """OpenAI、Gemini、Claude-3を使うハイブリッドクライアント
    
    各モデルの特性を活かし、相互にフォールバックする柔軟な利用を実現
    """

    def __init__(self, openai_client, gemini_client, anthropic_client=None):
        # 内部クライアントをラップ
        self.openai = OpenAIClient(openai_client)
        self.gemini = GeminiClient(gemini_client)
        self.anthropic = AnthropicClient(anthropic_client) if anthropic_client else None
        
        # OpenAI互換インターフェースのためのプロパティ
        self.chat = self
        self.completions = self
        
        # 各プロバイダーのモデル一覧
        self.openai_models = self.openai.models
        self.anthropic_models = self.anthropic.models if self.anthropic else []
        
        # ログ出力用の設定
        self.LOG_LEN = 20

    def _get_fallback_response(self, messages: List[Dict[str, str]]) -> Any:
        """フォールバック用のレスポンスを生成"""
        try:
            return self.openai.create(self.openai_models[0], messages)
        except Exception as err:
            logger.error(f"Fallback error: {err}")
            # 最終フォールバックが失敗した場合はエラーを投げる
            raise RuntimeError("All AI providers failed to respond") from err

    def _select_client(self, model: str) -> Tuple[BaseAIClient, str]:
        """モデル名からAPIクライアントを選択"""
        if model in self.openai_models:
            return self.openai, "openai"
        if model in self.anthropic_models and self.anthropic:
            return self.anthropic, "anthropic"
        return self.gemini, "gemini"

    def create(self, model: str, messages: List[Dict[str, str]]) -> Any:
        """指定されたモデルを使用してレスポンスを生成
        
        Args:
            model: 使用するモデル名
            messages: メッセージ履歴
            
        Returns:
            生成されたレスポンス
            
        Note:
            各モデルでエラーが発生した場合は、OpenAIのモデルにフォールバックする
        """
        try:
            # 入力ログ
            for message in messages:
                log_msg = message.get("content", "")
                if isinstance(log_msg, str):
                    log_msg = log_msg[:self.LOG_LEN].replace("\n", " ")
                logger.info(f"LLM IN: {log_msg}")
            
            # モデルに対応するクライアントを選択
            client, provider = self._select_client(model)
            
            try:
                # 選択したクライアントで実行
                response = client.create(model, messages)
                
                # レスポンスログ
                content = response.choices[0].message.content
                if content is not None:
                    logger.info(f"LLM OUT: {content[:self.LOG_LEN]}")
                return response
            except Exception as err:
                # エラー時はフォールバック
                logger.error(f"{provider} client error: {err}")
                return self._get_fallback_response(messages)
                
        except Exception as err:
            # 予期せぬエラー時もフォールバック
            logger.error(f"Unexpected error in create: {err}")
            return self._get_fallback_response(messages)

    def is_knowledge_insufficient(self, model: str, messages: List[Dict[str, str]]) -> bool:
        """LLMの知識が不足しているかを判断
        
        Args:
            model: 使用するモデル名
            messages: メッセージ履歴
            
        Returns:
            bool: 知識が不足している場合はTrue
        """
        try:
            # システムメッセージを追加して知識不足を検出しやすくする
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
            
            # 信頼度が低い、または明示的に知識不足と応答した場合はTrue
            confidence_score = getattr(response.choices[0].message, "confidence", 1.0)
            has_insufficient_marker = "knowledge_insufficient" in content
            uncertain_words = ["わかりません", "知りません", "不明です", "情報がありません"]
            has_uncertain_phrases = any(word in content for word in uncertain_words)
            
            return confidence_score < 0.5 or has_insufficient_marker or has_uncertain_phrases
        except Exception as e:
            logger.exception(f"Error checking knowledge sufficiency: {e}")
            return True  # エラー時は安全のためTrue


def load_ai_client():
    """設定からAIクライアントをロードして初期化"""
    config = load_config()
    
    # Geminiクライアント初期化
    gemini_client = openai.OpenAI(
        api_key=config.gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    
    # OpenAIクライアント初期化
    openai_client = openai.OpenAI(api_key=config.openai_api_key)
    
    # Anthropicクライアント初期化（キーがあれば）
    anthropic_client = (
        Anthropic(api_key=config.anthropic_api_key)
        if config.anthropic_api_key
        else None
    )
    
    return HybridAIClient(openai_client, gemini_client, anthropic_client)