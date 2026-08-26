import os
from collections import deque

import discord
from loguru import logger

from ai_client import load_ai_client
from config import load_config
from message_handler import process_message, send_messages
from message_history import History

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

config = None
ai_client = None

# 利用可能なモデルのリスト
AVAILABLE_MODELS = [
    "gemini-3.6-flash",  # Gemini
    "gpt-4o",  # OpenAI
    "claude-3-sonnet-20240229",  # Anthropic
]


def validate_model(model: str) -> str:
    """モデル名を検証し、有効なモデル名を返す

    Args:
        model: 検証するモデル名

    Returns:
        検証済みのモデル名

    Raises:
        ValueError: モデル名が無効な場合
    """
    if model not in AVAILABLE_MODELS:
        raise ValueError(
            f"Invalid model: {model}. "
            f"Available models are: {', '.join(AVAILABLE_MODELS)}"
        )
    return model


# デフォルトのテキストモデル（初期化時に検証）
text_model = validate_model("gemini-3.6-flash")

history = History()

# 直近に処理したメッセージIDを保持し、Gateway再接続時のイベント再配信による
# 二重処理（二重返信）を防ぐ。maxlenで自動的に古いIDが破棄される。
_recent_message_ids: deque = deque(maxlen=1000)


def is_duplicate_message(message: discord.Message) -> bool:
    """同一メッセージが再配信された場合にTrueを返し、処理済みとして記録する"""
    if message.id in _recent_message_ids:
        return True
    _recent_message_ids.append(message.id)
    return False


def ignore_message(message: discord.Message) -> bool:
    """ボット/Webhook/自分自身のメッセージを無視（自己返信ループの防止）"""
    is_self = client.user is not None and message.author.id == client.user.id
    is_bot = bool(getattr(message.author, "bot", False))
    is_webhook = message.webhook_id is not None
    ignore = is_self or is_bot or is_webhook
    if ignore:
        logger.info(
            f"ignore_message=True self={is_self} bot={is_bot} "
            f"webhook={is_webhook} author={message.author}"
        )
    return ignore


def check_if_channel_is_target(message: discord.Message) -> bool:
    """対象チャンネルかどうかを確認"""
    return message.channel.id in config.target_channel_ids


@client.event
async def on_ready():
    """ボット起動時の処理"""
    logger.info("We have logged in as {0.user} (pid={1})".format(client, os.getpid()))


@client.event
async def on_message(message: discord.Message):
    """メッセージ受信時の処理"""
    # 二重返信の切り分け用: 受信した全メッセージのIDとPIDを最初に記録
    logger.info(
        f"on_message ENTER pid={os.getpid()} id={message.id} "
        f"author={message.author} content={message.content[:20]!r}"
    )

    if ignore_message(message):
        logger.info("ignore message")
        return

    if not check_if_channel_is_target(message):
        logger.info("not target channel")
        return

    # 再配信された同一メッセージは無視（二重返信の防止）
    if is_duplicate_message(message):
        logger.info(f"duplicate message ignored: {message.id}")
        return

    logger.info(f"pid:{os.getpid()} channel_id:{message.channel.id}")
    logger.info(f"name:{message.author.name}")
    logger.info(f"message:{message.content[:50]}")

    async with message.channel.typing():
        bot_reply_message = await process_message(
            message, history, ai_client, text_model, config
        )

    if bot_reply_message:
        logger.info(
            f"on_message SEND pid={os.getpid()} id={message.id} "
            f"reply={bot_reply_message[:30]!r}"
        )
        await send_messages(message.channel, bot_reply_message)


def set_text_model(model: str) -> None:
    """テキストモデルを設定する

    Args:
        model: 設定するモデル名

    Raises:
        ValueError: モデル名が無効な場合
    """
    global text_model
    text_model = validate_model(model)


def run():
    """ボットの起動"""
    global config
    global ai_client
    config = load_config()
    ai_client = load_ai_client()

    # 環境変数でモデルを上書き可能に
    if config.text_model:
        try:
            set_text_model(config.text_model)
            logger.info(f"Using text model: {text_model}")
        except ValueError as e:
            logger.error(f"Invalid TEXT_MODEL environment variable: {e}")

    client.run(config.discord_api_key)
