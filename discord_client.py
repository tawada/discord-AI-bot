import logging
import os

import discord

from ai_client import load_ai_client
from config import load_config
from message_handler import process_message, send_messages
from message_history import History

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

config = None
ai_client = None

logger = logging.getLogger(__name__)

# 利用可能なモデルのリスト
AVAILABLE_MODELS = [
    "gemini-2.0-flash",  # Gemini
    "gpt-4o",            # OpenAI
    "claude-3-sonnet-20240229"  # Anthropic
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
text_model = validate_model("gemini-2.0-flash")

history = History()


def ignore_message(message: discord.Message) -> bool:
    """ボットのメッセージを無視"""
    return message.author == client.user or message.author.bot

def check_if_channel_is_target(message: discord.Message) -> bool:
    """対象チャンネルかどうかを確認"""
    return message.channel.id in config.target_channnel_ids

@client.event
async def on_ready():
    """ボット起動時の処理"""
    logger.info("We have logged in as {0.user}".format(client))

@client.event
async def on_message(message: discord.Message):
    """メッセージ受信時の処理"""
    if ignore_message(message):
        logger.info("ignore message")
        return

    if not check_if_channel_is_target(message):
        logger.info("not target channel")
        return

    logger.info(f"channel_id:{message.channel.id}")
    logger.info(f"name:{message.author.name}")
    logger.info(f"message:{message.content[:50]}")

    async with message.channel.typing():
        bot_reply_message = await process_message(
            message,
            history,
            ai_client,
            text_model,
            config
        )

    if bot_reply_message:
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
    if "TEXT_MODEL" in os.environ:
        try:
            set_text_model(os.environ["TEXT_MODEL"])
            logger.info(f"Using text model: {text_model}")
        except ValueError as e:
            logger.error(f"Invalid TEXT_MODEL environment variable: {e}")
    
    client.run(config.discord_api_key)
