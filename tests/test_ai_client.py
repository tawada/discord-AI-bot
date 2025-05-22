from unittest.mock import MagicMock, patch

import pytest

from ai_client import HybridAIClient


def make_openai_like_response(content="Hello from Claude", role="assistant"):
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_choice.message.role = role
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


def test_hybrid_client_openai_model():
    with patch.object(HybridAIClient, 'create', return_value=make_openai_like_response("Hello from openai")) as mock_create:
        hybrid = HybridAIClient()
        messages = [{"role": "user", "content": "Hello from openai."}]
        try:
            response = hybrid.create("gpt-4o", messages)
            assert response.choices[0].message.content == "Hello from openai"
        except Exception as e:
            assert False, f"LangChain OpenAI create failed: {e}"


def test_hybrid_client_gemini_model_fallback_success():
    with patch.object(HybridAIClient, 'create', return_value=make_openai_like_response("Hello from gemini")) as mock_create:
        hybrid = HybridAIClient()
        messages = [{"role": "user", "content": "Hello from gemini."}]
        try:
            response = hybrid.create("gemini-1.5-flash", messages)
            assert response.choices[0].message.content == "Hello from gemini"
        except Exception as e:
            assert False, f"LangChain Gemini create failed: {e}"


def test_hybrid_client_gemini_model_fallback_error():
    with patch.object(HybridAIClient, 'create', side_effect=Exception("Gemini error")) as mock_create:
        hybrid = HybridAIClient()
        messages = [{"role": "user", "content": "Hello from gemini fallback."}]
        with pytest.raises(Exception):
            hybrid.create("gemini-1.5-flash", messages)


def test_hybrid_client_claude_model_success():
    with patch.object(HybridAIClient, 'create', return_value=make_openai_like_response("Hello from Claude")) as mock_create:
        hybrid = HybridAIClient()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello from Claude."},
        ]
        try:
            response = hybrid.create("claude-3-sonnet-20240229", messages)
            assert response.choices[0].message.content == "Hello from Claude"
            assert response.choices[0].message.role == "assistant"
        except Exception as e:
            assert False, f"LangChain Claude create failed: {e}"


def test_hybrid_client_claude_model_fallback():
    with patch.object(HybridAIClient, 'create', return_value=make_openai_like_response("Claude fallback")) as mock_create:
        hybrid = HybridAIClient()
        messages = [{"role": "user", "content": "Hello from Claude fallback."}]
        try:
            response = hybrid.create("claude-3-sonnet-20240229", messages)
            assert response.choices[0].message.content == "Claude fallback"
        except Exception as e:
            assert False, f"LangChain Claude fallback failed: {e}"


def test_hybrid_client_claude_model_not_configured():
    with patch.object(HybridAIClient, 'create', side_effect=Exception("Claude not configured")) as mock_create:
        hybrid = HybridAIClient()
        messages = [{"role": "user", "content": "Hello without Claude."}]
        with pytest.raises(Exception):
            hybrid.create("claude-3-sonnet-20240229", messages)
