from typing import Any, Dict, List

import discord
from loguru import logger

import functions
import summarizer
from message_history import GPTMessage, History
from search_handler import is_search_needed, search_and_summarize


async def get_reply_message(
    message: discord.Message,
    history: History,
    ai_client: Any,
    text_model: str,
    config: Any,
    optional_messages: List[Dict[str, str]] = [],
) -> str:
    """ユーザーのメッセージに対して、LLMによる返信を取得する
    
    Args:
        message: Discordのメッセージオブジェクト
        history: 会話履歴
        ai_client: AI APIクライアント
        text_model: 使用するモデル名
        config: ボット設定
        optional_messages: 追加のコンテキスト情報
        
    Returns:
        str: 整形されたボット応答
    """
    user_message = message.content
    user_name = message.author.name

    # 会話履歴を取得
    messages = history.get_messages()

    # システムプロンプトを追加
    messages.append({"role": "system", "content": config.role_prompt})
    
    # ユーザーメッセージを追加
    messages.append({"role": "user", "content": f"{user_name}:\n{user_message}"})
    
    # 知識が不足していると判断された場合に検索を実行
    if ai_client.is_knowledge_insufficient(text_model, messages) and not optional_messages:
        logger.info("LLMの知識が不足しています。外部情報を検索します。")
        summary = search_and_summarize(user_message, ai_client, text_model)
        optional_messages.append({
            "role": "system",
            "content": f"「{user_message}」の検索結果要約:\n{summary}",
        })

    # 追加情報があれば会話コンテキストに追加
    if optional_messages:
        messages.extend(optional_messages)
        
    # アシスタント応答のテンプレートを追加
    messages.append({"role": "assistant", "content": f"{config.role_name}:\n"})

    # AIによる応答生成
    bot_reply_message = await generate_ai_response(ai_client, text_model, messages)
    
    # 履歴に追加
    update_history(history, user_name, user_message, bot_reply_message, optional_messages, config)
    
    # 応答の整形
    validated_message = format_response(bot_reply_message, config)
    
    return validated_message


async def generate_ai_response(ai_client: Any, model: str, messages: List[Dict[str, str]]) -> str:
    """AIを使用して応答を生成
    
    Args:
        ai_client: AI APIクライアント
        model: 使用するモデル名
        messages: メッセージ履歴
        
    Returns:
        str: 生成された応答テキスト
    """
    try:
        response = ai_client.chat.completions.create(
            model=model,
            messages=messages,
        )
        bot_reply = response.choices[0].message.content
        # ロール名のプレフィックスを削除
        if ":" in bot_reply:
            bot_reply = bot_reply.split(":", 1)[1].strip()
        return bot_reply
    except Exception as err:
        logger.exception(f"AI response generation failed: {err}")
        return "Error: LLM API failed"


def update_history(
    history: History, 
    user_name: str, 
    user_message: str, 
    bot_reply: str, 
    optional_messages: List[Dict[str, str]], 
    config: Any
) -> None:
    """会話履歴を更新
    
    Args:
        history: 履歴オブジェクト
        user_name: ユーザー名
        user_message: ユーザーメッセージ
        bot_reply: ボットの応答
        optional_messages: 追加のコンテキスト情報
        config: ボット設定
    """
    # ユーザーメッセージを履歴に追加
    history.add(GPTMessage("user", f"{user_name}:\n{user_message}"))
    
    # 追加情報を履歴に追加
    for optional_message in optional_messages:
        history.add(GPTMessage(optional_message["role"], optional_message["content"]))
        
    # ボット応答を履歴に追加
    history.add(GPTMessage("assistant", f"{config.role_name}:\n{bot_reply}"))


def format_response(response: str, config: Any) -> str:
    """ボットの応答を整形
    
    Args:
        response: 生の応答テキスト
        config: ボット設定
        
    Returns:
        str: 整形された応答テキスト
    """
    # 括弧や空行を除去
    validated_message = functions.remove_brackets_and_spaces(response)
    
    # 応答の長さを制限
    validated_message = functions.limit_sentences(validated_message, 2)
    
    return validated_message


async def process_message(
    message: discord.Message,
    history: History,
    ai_client: Any,
    text_model: str,
    config: Any,
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
                    "content": "画像の要約:\n"
                    + attachment.filename
                    + "\n"
                    + summarized_text,
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
                "content": f"「{message.content}」の検索結果要約:\n{summary}",
            }
        )

    # 本文も添付もなく、かつ検索にもヒットしないなら返信なし
    if not message.content and not optional_messages:
        return ""

    return await get_reply_message(
        message, history, ai_client, text_model, config, optional_messages
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
