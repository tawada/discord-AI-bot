import dataclasses
import logging
import os

import discord
import openai

import summarizer


@dataclasses.dataclass
class Config:
    discord_api_key: str
    openai_api_key: str
    target_channnel_ids: list[int]


config = Config(
    discord_api_key=os.environ["DISCORD_API_KEY"],
    openai_api_key=os.environ["OPENAI_API_KEY"],
    target_channnel_ids=list(map(int, os.environ["CHANNEL_IDS"].split(","))),
)

openai_client = openai.OpenAI(api_key=config.openai_api_key)
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

roleplay = os.environ["ROLE_PROMPT"]
role_bot_name = os.environ["ROLE_NAME"]

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


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


async def get_reply_message(message, optional_messages=[]):
    """ユーザーのメッセージに対して、GPTによる返信を取得する"""
    user_message = message.content
    user_name = message.author.name

    messages = history.get_messages()

    messages.append({"role": "system", "content": roleplay})
    messages.append({"role": "user", "content": user_name + ":\n" + user_message})
    if optional_messages:
        messages.extend(optional_messages)
    messages.append({"role": "assistant", "content": role_bot_name + ":\n"})
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
        )
        bot_reply_message = response.choices[0].message.content
        bot_reply_message = bot_reply_message.replace(role_bot_name + ":", "").strip()
    except Exception as err:
        logger.exception(err)
        bot_reply_message = "Error: OpenAI API failed"
    history.add(GPTMessage("user", user_name + ":\n" + user_message))
    history.add(GPTMessage("assistant", role_bot_name + ":\n" + bot_reply_message))
    for optional_message in optional_messages:
        history.add(GPTMessage(optional_message["role"], optional_message["content"]))
    return bot_reply_message


@client.event
async def on_ready():
    logger.info("We have logged in as {0.user}".format(client))


@client.event
async def on_message(message):
    if message.author == client.user:
        # ボット以外
        return

    logger.info(f"channel_id:{message.channel.id}")
    logger.info(f"name:{message.author.name}")
    logger.info(f"message:{message.content[:50]}")

    if message.channel.id not in config.target_channnel_ids:
        logger.info("not target channel")
        return

    optional_messages = []

    if message.attachments:
        # 添付ファイルがある場合は、そのファイルを要約する
        for attachment in message.attachments:
            # 画像の場合は、その画像を要約する
            if not attachment.filename.endswith((".png", ".jpg", ".jpeg")) and not attachment.url.endswith((".png", ".jpg", ".jpeg")):
                continue
            try:
                summarized_text = summarizer.summarize_image(attachment.url, openai_client)
                logger.info(summarized_text[:50])
                optional_messages.append(
                    {"role": "system", "content": "画像の要約:\n" + attachment.filename + "\n" + summarized_text}
                )
            except RuntimeError:
                pass

    if "http" in message.content:
        # URLが含まれている場合は、そのページを要約する
        idx_s = message.content.find("http")
        idx_e = message.content.find("\n", idx_s)
        if idx_e == -1:
            idx_e = len(message.content)
        url = message.content[idx_s: idx_e + 1]
        try:
            summarized_text = summarizer.summarize_webpage(url, openai_client)
            logger.info(summarized_text[:50])
            optional_messages.append(
                {
                    "role": "system",
                    "content": "ページの要約:\n" + url + "\n" + summarized_text,
                }
            )
        except RuntimeError:
            pass

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


if __name__ == "__main__":
    client.run(config.discord_api_key)
