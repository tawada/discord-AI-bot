import dataclasses
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import discord_client
from message_history import GPTMessage, History
from search_handler import search_and_summarize
from ai_client import HybridAIClient
from config import load_config

def make_openai_like_response(content="テスト応答", role="assistant"):
    class Message:
        def __init__(self, content):
            self.content = content
            self.role = role
    class Choice:
        def __init__(self, message):
            self.message = message
    class Response:
        def __init__(self, choices):
            self.choices = choices
    return Response([Choice(Message(content))])

@pytest.fixture(autouse=True)
def patch_discord_ai_client(monkeypatch):
    ai_client = HybridAIClient()
    monkeypatch.setattr(discord_client, "ai_client", ai_client)
    with patch.object(ai_client, "create", side_effect=lambda *a, **kw: make_openai_like_response()):
        yield

@pytest.fixture(autouse=True)
def patch_llmchain():
    with patch("langchain.chains.LLMChain") as mock_llmchain:
        mock_chain = MagicMock()
        mock_chain.run.return_value = "Dummy AI summary"
        mock_llmchain.return_value = mock_chain
        yield

@pytest.fixture
def mock_ddgs():
    """DuckDuckGo の検索をモックする"""
    with patch("search_handler.DDGS") as mock_ddgs_class:
        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.text.return_value = [
            {"title": "Result 1", "body": "This is the first snippet.", "href": "https://example.com/1"},
            {"title": "Result 2", "body": "Another snippet.", "href": "https://example.com/2"},
        ]
        mock_ddgs_class.return_value.__enter__.return_value = mock_ddgs_instance
        yield mock_ddgs_class

@pytest.fixture(autouse=True)
def patch_ddgs():
    with patch("search_handler.DDGS") as mock_ddgs_class:
        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.text.return_value = [
            {"title": "Result 1", "body": "This is the first snippet.", "href": "https://example.com/1"},
            {"title": "Result 2", "body": "Another snippet.", "href": "https://example.com/2"},
        ]
        mock_ddgs_class.return_value.__enter__.return_value = mock_ddgs_instance
        yield

@pytest.fixture(autouse=True)
def patch_discord_config(monkeypatch):
    monkeypatch.setattr(discord_client, "config", load_config())
    yield

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


def test_search_and_summarize_success(mock_ddgs):
    """
    search_and_summarize が正常に動作して DuckDuckGo と ai_client の結果を用いて
    要約を返すかをテスト
    """
    # langchainのChatOpenAIとLLMChainをモック
    with patch("langchain.chat_models.ChatOpenAI") as mock_chat_openai:
        mock_chain = MagicMock()
        mock_chain.run.return_value = "Dummy AI summary"
        mock_chat_openai.return_value = MagicMock()
        with patch("langchain.chains.LLMChain") as mock_llmchain:
            mock_llmchain.return_value = mock_chain
            result = search_and_summarize(
                "Python の概要を教えて", discord_client.ai_client, discord_client.text_model
            )
            assert "Dummy AI summary" in result  # 要約結果が返ってくる
            mock_ddgs.assert_called_once()


def test_search_and_summarize_no_results():
    """
    DuckDuckGo で結果が一つも得られなかった場合にメッセージを返すか
    """
    with patch("search_handler.DDGS") as mock_ddgs_class:
        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.text.return_value = []
        mock_ddgs_class.return_value.__enter__.return_value = mock_ddgs_instance

        result = search_and_summarize(
            "何もヒットしないテスト",
            discord_client.ai_client,
            discord_client.text_model,
        )
        assert "検索結果が見つかりませんでした" in result


@pytest.mark.asyncio
async def test_get_reply_message():
    """get_reply_message 関数のテスト"""
    mock_message = MagicMock()
    mock_message.content = "テストメッセージ"
    mock_message.author.name = "テストユーザー"

    result = await discord_client.process_message(
        mock_message,
        discord_client.history,
        discord_client.ai_client,
        discord_client.text_model,
        discord_client.config,
    )
    assert "テスト応答" in result


@pytest.mark.asyncio
async def test_get_reply_message_with_insufficient_knowledge(mock_ddgs):
    """Test get_reply_message when LLM knowledge is insufficient"""
    mock_message = MagicMock()
    mock_message.content = "テストメッセージ"
    mock_message.author.name = "テストユーザー"

    # Mock the knowledge insufficiency check to return True
    with patch("discord_client.ai_client.is_knowledge_insufficient", return_value=True):
        # Mock the search and summarize function
        with patch(
            "message_handler.search_and_summarize", return_value="検索結果の要約"
        ) as mock_search:
            result = await discord_client.process_message(
                mock_message,
                discord_client.history,
                discord_client.ai_client,
                discord_client.text_model,
                discord_client.config,
            )
            assert "テスト応答" in result
            mock_search.assert_called_once_with(
                "テストメッセージ", discord_client.ai_client, discord_client.text_model
            )


@pytest.mark.asyncio
async def test_send_messages():
    """send_messages 関数のテスト - 2000文字以上のメッセージの分割送信"""
    mock_channel = MagicMock()
    # send メソッドを非同期関数としてモック化
    mock_channel.send = AsyncMock()
    long_message = "a" * 2500  # 2000文字を超えるメッセージ

    await discord_client.send_messages(mock_channel, long_message)

    # 2回に分けて送信されることを確認
    assert mock_channel.send.call_count == 2
    # 最初の送信が2000文字
    assert len(mock_channel.send.call_args_list[0][0][0]) == 2000
    # 2回目の送信が残りの文字数
    assert len(mock_channel.send.call_args_list[1][0][0]) == 500


@pytest.mark.asyncio  # 非同期テストを実行するためのマーカー
async def test_on_message_ignore_bot():
    """ボット自身のメッセージは無視されることを確認"""
    mock_message = MagicMock()
    mock_message.author = discord_client.client.user

    with patch("message_handler.process_message") as mock_process:
        await discord_client.on_message(mock_message)
        mock_process.assert_not_called()
