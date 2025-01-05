from unittest.mock import MagicMock

import pytest

from ai_client import HybridAIClient


@pytest.fixture
def mock_openai_client():
    client = MagicMock()
    return client


@pytest.fixture
def mock_gemini_client():
    client = MagicMock()
    return client


def test_hybrid_client_openai_model(mock_openai_client, mock_gemini_client):
    hybrid = HybridAIClient(openai_client=mock_openai_client, gemini_client=mock_gemini_client)

    messages = [{"role": "user", "content": "Hello from openai."}]
    hybrid.create("gpt-4o", messages)

    # openai_client が呼ばれ、gemini_client は呼ばれていないこと
    mock_openai_client.chat.completions.create.assert_called_once_with(
        model="gpt-4o",
        messages=messages,
    )
    mock_gemini_client.chat.completions.create.assert_not_called()


def test_hybrid_client_gemini_model_fallback_success(mock_openai_client, mock_gemini_client):
    hybrid = HybridAIClient(openai_client=mock_openai_client, gemini_client=mock_gemini_client)

    messages = [{"role": "user", "content": "Hello from gemini."}]
    hybrid.create("gemini-1.5-flash", messages)

    # gemini_client が呼ばれる
    mock_gemini_client.chat.completions.create.assert_called_once_with(
        model="gemini-1.5-flash",
        messages=messages,
    )
    mock_openai_client.chat.completions.create.assert_not_called()


def test_hybrid_client_gemini_model_fallback_error(mock_openai_client, mock_gemini_client):
    """Gemini が失敗した時に fallback で openai_client.gpt-4o を呼び出す"""
    mock_gemini_client.chat.completions.create.side_effect = Exception("Gemini error")

    hybrid = HybridAIClient(openai_client=mock_openai_client, gemini_client=mock_gemini_client)

    messages = [{"role": "user", "content": "Hello from gemini fallback."}]
    hybrid.create("gemini-1.5-flash", messages)

    # gemini_client が一度失敗したのち、openai_client が呼ばれる
    mock_gemini_client.chat.completions.create.assert_called_once()
    mock_openai_client.chat.completions.create.assert_called_once()
    # モデルは "gpt-4o" にフォールバックされる
    called_args, called_kwargs = mock_openai_client.chat.completions.create.call_args
    assert called_kwargs["model"] == "gpt-4o"
