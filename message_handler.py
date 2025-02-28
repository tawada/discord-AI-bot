import logging
from typing import List, Dict, Any

import discord

import functions
import summarizer
from message_history import GPTMessage, History
from search_handler import search_and_summarize, is_search_needed

logger = logging.getLogger(__name__)


async def get_reply_message(
    message: discord.Message,
    history: History,
    ai_client: Any,
    text_model: str,
    config: Any,
    optional_messages: List[Dict[str, str]] = []
) -> str:
    """ユーザーのメッセージに対して、LLMによる返信を取得する"""
    user_message = message.content
    user_name = message.author.name

    messages = history.get_messages()

    # ロールプロンプトを system として追加
    messages.append({"role": "system", "content": config.role_prompt})
    # ユーザーからのメッセージ
    messages.append({"role": "user", "content": user_name + ":\n" + user_message})
    # LLMの知識不足を判定
    if ai_client.is_knowledge_insufficient(text_model, messages) and not optional_messages:
        logger.info("LLMの知識が不足しています。外部情報を検索します。")
        summary = search_and_summarize(user_message, ai_client, text_model)
        optional_messages.append(
            {
                "role": "system",
                "content": f"「{user_message}」の検索結果要約:\n{summary}"
            }
        )

    # optional_messages があれば追記
    if optional_messages:
        messages.extend(optional_messages)
    # アシスタント応答
    messages.append({"role": "assistant", "content": config.role_name + ":\n"})

    try:
        response = ai_client.chat.completions.create(
            model=text_model,
            messages=messages,
        )
        bot_reply_message = response.choices[0].message.content
        bot_reply_message = bot_reply_message.replace(config.role_name + ":", "").strip()
    except Exception as err:
        logger.exception(err)
        bot_reply_message = "Error: LLM API failed"

    # 履歴に追加
    history.add(GPTMessage("user", user_name + ":\n" + user_message))
    for optional_message in optional_messages:
        history.add(GPTMessage(optional_message["role"], optional_message["content"]))
    history.add(GPTMessage("assistant", config.role_name + ":\n" + bot_reply_message))

    return bot_reply_message


async def process_message(
    message: discord.Message,
    history: History,
    ai_client: Any,
    text_model: str,
    config: Any
) -> str:
    """メッセージを処理し、必要な情報を収集して返信を生成する"""
    optional_messages = []

    # 画像の処理
    for attachment in message.attachments:
        if not functions.is_image_attachment(attachment):
            continue
        try:
            summarized_text = summarizer.summarize_image(attachment.url, ai_client)
            logger.info(summarized_text[:50])
            optional_messages.append(
                {
                    "role": "system",
                    "content": "画像の要約:\n" + attachment.filename + "\n" + summarized_text
                }
            )
        except RuntimeError:
            pass

    # URLの処理
    if functions.contains_url(message.content):
        urls = functions.extract_urls(message.content)
        try:
            url = urls[0]
            summarized_text = summarizer.summarize_webpage(url, ai_client)
            logger.info(summarized_text[:50])
            optional_messages.append(
                {
                    "role": "system",
                    "content": "ページの要約:\n" + url + "\n" + summarized_text,
                }
            )
        except RuntimeError:
            pass

    # 検索が必要な場合の処理
    if is_search_needed(message.content, ai_client, text_model):
        summary = search_and_summarize(message.content, ai_client, text_model)
        optional_messages.append(
            {
                "role": "system",
                "content": f"「{message.content}」の検索結果要約:\n{summary}"
            }
        )

    # 本文も添付もなく、かつ検索にもヒットしないなら返信なし
    if not message.content and not optional_messages:
        return ""

    return await get_reply_message(
        message,
        history,
        ai_client,
        text_model,
        config,
        optional_messages
    )


async def send_messages(channel: discord.TextChannel, message: str) -> None:
    """Discordの制限にあわせて2000文字ごとに分割して送信"""
    num_limit = 2000
    short_messages = []
    while len(message) > num_limit:
        short_messages.append(message[:num_limit])
        message = message[num_limit:]
    short_messages.append(message)

    for short_message in short_messages:
        await channel.send(short_message)