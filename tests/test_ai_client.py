from unittest.mock import MagicMock, patch

import pytest
from anthropic import Anthropic

from ai_client import HybridAIClient


@pytest.fixture
def mock_openai_client():
    client = MagicMock()
    return client


@pytest.fixture
def mock_gemini_client():
    client = MagicMock()
    return client


@pytest.fixture
def mock_anthropic_client():
    client = MagicMock()
    # Mock the response format
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Hello from Claude")]
    client.messages.create.return_value = mock_response
    return client


def test_hybrid_client_openai_model(mock_openai_client, mock_gemini_client):
    hybrid = HybridAIClient(
        openai_client=mock_openai_client, gemini_client=mock_gemini_client
    )

    messages = [{"role": "user", "content": "Hello from openai."}]
    hybrid.create("gpt-4o", messages)

    # openai_client が呼ばれ、gemini_client は呼ばれていないこと
    mock_openai_client.chat.completions.create.assert_called_once_with(
        model="gpt-4o",
        messages=messages,
    )
    mock_gemini_client.chat.completions.create.assert_not_called()


def test_hybrid_client_gemini_model_fallback_success(
    mock_openai_client, mock_gemini_client
):
    hybrid = HybridAIClient(
        openai_client=mock_openai_client, gemini_client=mock_gemini_client
    )

    messages = [{"role": "user", "content": "Hello from gemini."}]
    hybrid.create("gemini-1.5-flash", messages)

    # gemini_client が呼ばれる
    mock_gemini_client.chat.completions.create.assert_called_once_with(
        model="gemini-1.5-flash",
        messages=messages,
    )
    mock_openai_client.chat.completions.create.assert_not_called()


def test_hybrid_client_gemini_model_fallback_error(
    mock_openai_client, mock_gemini_client
):
    """Gemini が失敗した時に fallback で openai_client.gpt-4o を呼び出す"""
    mock_gemini_client.chat.completions.create.side_effect = Exception("Gemini error")

    hybrid = HybridAIClient(
        openai_client=mock_openai_client, gemini_client=mock_gemini_client
    )

    messages = [{"role": "user", "content": "Hello from gemini fallback."}]
    hybrid.create("gemini-1.5-flash", messages)

    # gemini_client が一度失敗したのち、openai_client が呼ばれる
    mock_gemini_client.chat.completions.create.assert_called_once()
    mock_openai_client.chat.completions.create.assert_called_once()
    # モデルは "gpt-4o" にフォールバックされる
    called_args, called_kwargs = mock_openai_client.chat.completions.create.call_args
    assert called_kwargs["model"] == "gpt-4o"


def test_hybrid_client_claude_model_success(
    mock_openai_client, mock_gemini_client, mock_anthropic_client
):
    """Claude-3-Sonnetモデルの正常系テスト"""
    hybrid = HybridAIClient(
        openai_client=mock_openai_client,
        gemini_client=mock_gemini_client,
        anthropic_client=mock_anthropic_client,
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello from Claude."},
    ]
    response = hybrid.create("claude-3-sonnet-20240229", messages)

    # Anthropicクライアントが正しく呼び出されること
    mock_anthropic_client.messages.create.assert_called_once()
    mock_gemini_client.chat.completions.create.assert_not_called()
    mock_openai_client.chat.completions.create.assert_not_called()

    # レスポンスが正しい形式に変換されていること
    assert response.choices[0].message.content == "Hello from Claude"
    assert response.choices[0].message.role == "assistant"

    # システムメッセージとユーザーメッセージが正しく渡されていること
    called_args = mock_anthropic_client.messages.create.call_args.kwargs
    assert called_args["system"] == "You are a helpful assistant."
    assert len(called_args["messages"]) == 1
    assert called_args["messages"][0]["role"] == "user"
    assert called_args["messages"][0]["content"] == "Hello from Claude."


def test_hybrid_client_claude_model_fallback(
    mock_openai_client, mock_gemini_client, mock_anthropic_client
):
    """Claudeが失敗した場合のフォールバックテスト"""
    mock_anthropic_client.messages.create.side_effect = Exception("Claude error")

    hybrid = HybridAIClient(
        openai_client=mock_openai_client,
        gemini_client=mock_gemini_client,
        anthropic_client=mock_anthropic_client,
    )

    messages = [{"role": "user", "content": "Hello from Claude fallback."}]
    hybrid.create("claude-3-sonnet-20240229", messages)

    # Claudeが失敗してGPT-4にフォールバックすること
    mock_anthropic_client.messages.create.assert_called_once()
    mock_openai_client.chat.completions.create.assert_called_once()
    called_args = mock_openai_client.chat.completions.create.call_args.kwargs
    assert called_args["model"] == "gpt-4o"


def test_hybrid_client_claude_model_not_configured(
    mock_openai_client, mock_gemini_client
):
    """Claudeクライアントが設定されていない場合のテスト"""
    hybrid = HybridAIClient(
        openai_client=mock_openai_client,
        gemini_client=mock_gemini_client,
        anthropic_client=None,
    )

    messages = [{"role": "user", "content": "Hello without Claude."}]
    hybrid.create("claude-3-sonnet-20240229", messages)

    # Claudeクライアントがない場合はGeminiを試す
    mock_gemini_client.chat.completions.create.assert_called_once_with(
        model="claude-3-sonnet-20240229",
        messages=messages,
    )


def test_message_format_conversion():
    """メッセージフォーマット変換のテスト
    システムメッセージは別途処理されるため、変換結果には含まれない"""
    hybrid = HybridAIClient(
        openai_client=MagicMock(),
        gemini_client=MagicMock(),
        anthropic_client=MagicMock(),
    )

    messages = [
        {"role": "system", "content": "System message"},
        {"role": "user", "content": "User message"},
        {"role": "assistant", "content": "Assistant message"},
    ]

    converted = hybrid._convert_messages_for_anthropic(messages)

    assert len(converted) == 2  # システムメッセージは除外される
    assert converted[0]["role"] == "user"
    assert converted[0]["content"] == "User message"
    assert converted[1]["role"] == "assistant"
    assert converted[1]["content"] == "Assistant message"
