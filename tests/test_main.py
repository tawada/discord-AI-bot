import pytest
from unittest.mock import patch, MagicMock
import dataclasses

import main
from main import GPTMessage, History, search_and_summarize


@pytest.fixture
def mock_ai_client():
    """ai_client.chat.completions.create をモックする"""
    with patch("main.ai_client.chat.completions.create") as mock_create:
        yield mock_create


@pytest.fixture
def mock_ddgs():
    """DuckDuckGo の検索をモックする"""
    with patch("main.DDGS") as mock_ddgs_class:
        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.text.return_value = [
            "Result 1: This is the first snippet.",
            "Result 2: Another snippet.",
        ]
        mock_ddgs_class.return_value.__enter__.return_value = mock_ddgs_instance
        yield mock_ddgs_class


def test_history_add_and_get_messages():
    history = History(num_output=3)
    m1 = GPTMessage(role="user", content="Hello")
    m2 = GPTMessage(role="assistant", content="Hi")
    m3 = GPTMessage(role="system", content="System message")
    m4 = GPTMessage(role="user", content="Another user message")

    history.add(m1)
    history.add(m2)
    history.add(m3)
    history.add(m4)

    messages = history.get_messages()
    assert len(messages) == 3  # 最新3つのみ取得
    # dataclasses.asdict() で取得される辞書に role, content が含まれるかを確認
    assert messages[0]["role"] == "assistant"
    assert messages[1]["role"] == "system"
    assert messages[2]["role"] == "user"


def test_search_and_summarize_success(mock_ai_client, mock_ddgs):
    """
    search_and_summarize が正常に動作して DuckDuckGo と ai_client の結果を用いて
    要約を返すかをテスト
    """
    # ai_client から返ってくるダミーのレスポンスを用意
    mock_ai_client.return_value.choices = [
        MagicMock(message=MagicMock(content="Dummy AI summary"))
    ]

    result = search_and_summarize("Python の概要を教えて")
    assert "Dummy AI summary" in result  # 要約結果が返ってくる

    # DuckDuckGo の検索が呼ばれているかを確認
    mock_ddgs.assert_called_once()


def test_search_and_summarize_no_results(mock_ai_client):
    """
    DuckDuckGo で結果が一つも得られなかった場合にメッセージを返すか
    """
    with patch("main.DDGS") as mock_ddgs_class:
        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.text.return_value = []
        mock_ddgs_class.return_value.__enter__.return_value = mock_ddgs_instance

        mock_ai_client.return_value.choices = [
            MagicMock(message=MagicMock(content="No query found"))
        ]

        result = search_and_summarize("何もヒットしないテスト")
        assert "検索結果が見つかりませんでした" in result
