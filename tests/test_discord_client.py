import dataclasses
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

import discord_client
from discord_client import GPTMessage, History, search_and_summarize


@pytest.fixture(autouse=True)
def mock_discord_setup():
    """
    テスト実行前に、discord_client 内の config/ai_client を
    すべてモック化するフィクスチャ。
    """
    with patch("discord_client.config", MagicMock(
        # テスト中に参照される設定だけ用意すればOK
        role_prompt="Mocked role prompt",
        role_name="Mocked role name",
        target_channnel_ids=[12345],
        discord_api_key="dummy"
    )):
        with patch("discord_client.ai_client", MagicMock()) as mock_ai:
            yield mock_ai


@pytest.fixture
def mock_ai_client():
    """ai_client.chat.completions.create をモックする"""
    with patch("discord_client.ai_client.chat.completions.create") as mock_create:
        yield mock_create


@pytest.fixture
def mock_ddgs():
    """DuckDuckGo の検索をモックする"""
    with patch("discord_client.DDGS") as mock_ddgs_class:
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
    with patch("discord_client.DDGS") as mock_ddgs_class:
        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.text.return_value = []
        mock_ddgs_class.return_value.__enter__.return_value = mock_ddgs_instance

        mock_ai_client.return_value.choices = [
            MagicMock(message=MagicMock(content="No query found"))
        ]

        result = search_and_summarize("何もヒットしないテスト")
        assert "検索結果が見つかりませんでした" in result


@pytest.mark.asyncio
async def test_get_reply_message():
    """get_reply_message 関数のテスト"""
    mock_message = MagicMock()
    mock_message.content = "テストメッセージ"
    mock_message.author.name = "テストユーザー"

    with patch("discord_client.ai_client.chat.completions.create") as mock_create:
        mock_create.return_value.choices = [
            MagicMock(message=MagicMock(content="テスト応答"))
        ]
        
        result = await discord_client.get_reply_message(mock_message)
        assert "テスト応答" in result


@pytest.mark.asyncio
async def test_get_reply_message_with_insufficient_knowledge(mock_ai_client, mock_ddgs):
    """Test get_reply_message when LLM knowledge is insufficient"""
    mock_message = MagicMock()
    mock_message.content = "テストメッセージ"
    mock_message.author.name = "テストユーザー"

    # Mock the knowledge insufficiency check to return True
    with patch("discord_client.ai_client.is_knowledge_insufficient", return_value=True):
        # Mock the AI client response
        mock_ai_client.return_value.choices = [
            MagicMock(message=MagicMock(content="テスト応答"))
        ]

        # Mock the search and summarize function
        with patch("discord_client.search_and_summarize", return_value="検索結果の要約") as mock_search:
            result = await discord_client.get_reply_message(mock_message)
            assert "テスト応答" in result
            mock_search.assert_called_once_with("テストメッセージ")

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
    
    with patch("discord_client.get_reply_message") as mock_get_reply:
        await discord_client.on_message(mock_message)
        mock_get_reply.assert_not_called()
