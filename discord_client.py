import dataclasses
import logging
import os
import re

import discord
import openai
import requests
from duckduckgo_search import DDGS

import functions
import summarizer
from ai_client import load_ai_client
from config import load_config

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

config = None
ai_client = None

logger = logging.getLogger(__name__)

text_model = "gemini-2.0-flash"

@dataclasses.dataclass
class GPTMessage:
    role: str
    content: str


class History:
    def __init__(self, num_output=10):
        self.messages = []
        self.num_output = num_output

    def add(self, message: GPTMessage):
        self.messages.append(message)

    def get_messages(self):
        return [
            dataclasses.asdict(message) for message in self.messages[-self.num_output:]
        ]


history = History()


def search_and_summarize(user_question: str) -> str:
    """
    DuckDuckGo API を使って検索し、出てきた情報をまとめて Gemini で要約する。
    """

    # 検索queryを作成
    messages = [
        {"role": "user", "content": user_question},
        {
            "role": "system",
            "content": (
                f"上記のユーザの質問「{user_question}」に対して、"
                "検索するべき単語を抽出してください。回答は単語のみで構いません。"
            ),
        },
    ]
    try:
        response = ai_client.chat.completions.create(
            model=text_model,
            messages=messages,
        )
        query = response.choices[0].message.content
    except Exception as e:
        logger.exception(e)
        query = user_question

    logger.info(f"Searching for: {query}")
    # DuckDuckGo で検索
    with DDGS() as ddgs:
        results = list(ddgs.text(
            keywords=query,
            region='jp-jp',
            max_results=10
        ))

    if not results:
        return "検索結果が見つかりませんでした。"
    
    # 検索結果をまとめる
    combined_text = f"{results}"

    # モデルに送りすぎるとエラーになりやすいので適当にカット
    combined_text = combined_text[:4096]

    logger.info(f"Search snippets (trimmed): {combined_text[:100]}...")

    if not combined_text.strip():
        return "検索結果から有用な情報を取得できませんでした。"

    # Gemini で要約
    messages = [
        {"role": "user", "content": combined_text},
        {
            "role": "system",
            "content": (
                f"上記の検索結果に基づいて、ユーザの質問「{query}」に答えるための"
                "簡潔な日本語要約を作成してください。"
            ),
        },
    ]
    try:
        response = ai_client.chat.completions.create(
            model=text_model,
            messages=messages,
        )
        summary = response.choices[0].message.content
    except Exception as e:
        logger.exception(e)
        summary = "検索の要約時にエラーが発生しました。"
    return summary


def is_search_needed(user_message: str) -> bool:
    """Determine if a search is needed based on the user's message."""
    # Define keywords or patterns that indicate a search is needed
    search_keywords = ["教えて", "とは", "何", "どうやって", "方法"]
    if any(keyword in user_message for keyword in search_keywords):
        return True

    messages = [
        {"role": "user", "content": user_message},
        {
            "role": "system",
            "content": (
                '上記のユーザーの発言に適切に答えるためにインターネット検索が必要であれば True、不要であれば False を返してください。必ず "True" または "False" のいずれかのみを返してください。'
            ),
        },
    ]

    try:
        response = ai_client.chat.completions.create(
            model=text_model,
            messages=messages,
        )
        if "False" not in response.choices[0].message.content:
            return True
    except Exception as e:
        logger.exception(e)
    return False


async def get_reply_message(message, optional_messages=[]):
    """ユーザーのメッセージに対して、Gemini (via openai ライブラリ) による返信を取得する。
    画像認識はできないので、画像要約だけ別途 openai_client を使う。
    """
    user_message = message.content
    user_name = message.author.name

    messages = history.get_messages()

    # ロールプロンプトを system として追加
    messages.append({"role": "system", "content": config.role_prompt})
    # ユーザーからのメッセージ
    messages.append({"role": "user", "content": user_name + ":\n" + user_message})
    # LLMの知識不足を判定
    if ai_client.is_knowledge_insufficient(text_model, messages):
        logger.info("LLMの知識が不足しています。外部情報を検索します。")
        summary = search_and_summarize(user_message)
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
        # --- Gemini 側でチャット生成 ---
        response = ai_client.chat.completions.create(
            model=text_model,
            messages=messages,
        )
        bot_reply_message = response.choices[0].message.content
        bot_reply_message = bot_reply_message.replace(config.role_name + ":", "").strip()
    except Exception as err:
        logger.exception(err)
        bot_reply_message = "Error: Gemini API failed"

    # 履歴に追加
    history.add(GPTMessage("user", user_name + ":\n" + user_message))
    for optional_message in optional_messages:
        history.add(GPTMessage(optional_message["role"], optional_message["content"]))
    history.add(GPTMessage("assistant", config.role_name + ":\n" + bot_reply_message))

    return bot_reply_message


@client.event
async def on_ready():
    logger.info("We have logged in as {0.user}".format(client))


def ignore_message(message):
    return message.author == client.user or message.author.bot


def check_if_channel_is_target(message):
    return message.channel.id in config.target_channnel_ids


@client.event
async def on_message(message):
    if ignore_message(message):
        # ボット自身の発言は無視
        logger.info("ignore message")
        return

    if not check_if_channel_is_target(message):
        # 対象チャンネル以外は無視
        logger.info("not target channel")
        return

    logger.info(f"channel_id:{message.channel.id}")
    logger.info(f"name:{message.author.name}")
    logger.info(f"message:{message.content[:50]}")

    optional_messages = []

    for attachment in message.attachments:
        if not functions.is_image_attachment(attachment):
            # 画像以外は無視
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

    # --- ここから「～を教えて」を検出して検索する処理 ---
    if is_search_needed(message.content):
        # 検索して要約
        summary = search_and_summarize(message.content)
        # optional_messages に検索結果の要約を追加
        optional_messages.append(
            {
                "role": "system",
                "content": f"「{message.content}」の検索結果要約:\n{summary}"
            }
        )

    # 本文も添付もなく、かつ検索にもヒットしないなら返事しない
    if not message.content and not optional_messages:
        return

    async with message.channel.typing():
        bot_reply_message = await get_reply_message(message, optional_messages)

    await send_messages(message.channel, bot_reply_message)


async def send_messages(channel, message):
    """Discordの制限にあわせて2000文字ごとに分割して送信"""
    num_limit = 2000
    short_messages = []
    while len(message) > num_limit:
        short_messages.append(message[:num_limit])
        message = message[num_limit:]
    short_messages.append(message)

    for short_message in short_messages:
        await channel.send(short_message)


def run():
    global config
    global ai_client
    config = load_config()
    ai_client = load_ai_client()
    client.run(config.discord_api_key)
