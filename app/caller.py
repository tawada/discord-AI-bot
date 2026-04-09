import os
import re

import discord
from discord import ui
import logging

from app import agent
from app.agent import AgentResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Discordのアップロード上限 (Nitroなしで25MB)
_MAX_FILE_SIZE = 25 * 1024 * 1024

# Yes/No質問の判定パターン
_YESNO_PATTERN = re.compile(
    r"(よろしいですか|しますか|しましょうか|いいですか|ですか？|ますか？"
    r"|shall I|should I|do you want|would you like|yes or no|proceed\?)",
    re.IGNORECASE,
)


async def _send_response(send_func, res: AgentResponse, caller: "Caller", channel_id: int):
    """AgentResponseをDiscordに送信する共通処理"""
    valid_files = []
    skipped = []
    for f in res.files:
        try:
            size = os.path.getsize(f)
        except OSError:
            continue
        if size <= _MAX_FILE_SIZE:
            valid_files.append(discord.File(f))
        else:
            skipped.append(f"{os.path.basename(f)} ({size / 1024 / 1024:.1f}MB)")

    text = res.text
    if skipped:
        text += "\n\n⚠ アップロード上限(25MB)を超えたファイル: " + ", ".join(skipped)

    kwargs = {}
    if valid_files:
        kwargs["files"] = valid_files
    if _YESNO_PATTERN.search(res.text):
        kwargs["view"] = YesNoView(caller, channel_id)

    try:
        await send_func(text, **kwargs)
    except discord.HTTPException as e:
        logger.error("Failed to send message: %s", e)
        try:
            await send_func(f"⚠ メッセージの送信に失敗しました: {e.status} {e.text}")
        except discord.HTTPException:
            pass


class YesNoView(ui.View):
    """Yes/Noボタンを表示するView"""

    def __init__(self, caller: "Caller", channel_id: int):
        super().__init__(timeout=300)
        self.caller = caller
        self.channel_id = channel_id

    @ui.button(label="Yes", style=discord.ButtonStyle.success)
    async def yes_button(self, interaction: discord.Interaction, button: ui.Button):
        await self._handle(interaction, "Yes")

    @ui.button(label="No", style=discord.ButtonStyle.danger)
    async def no_button(self, interaction: discord.Interaction, button: ui.Button):
        await self._handle(interaction, "No")

    async def _handle(self, interaction: discord.Interaction, answer: str):
        self.stop()
        for child in self.children:
            child.disabled = True
        await interaction.response.edit_message(view=self)

        res = await self.caller.call_agent(answer, self.channel_id)
        await _send_response(interaction.followup.send, res, self.caller, self.channel_id)


class Caller:
    """Discordボットの呼び出しクラス"""

    def __init__(self):
        self.client = self.setup_client()
        self.target_channels: set[int] = set()


    def setup_client(self) -> discord.Client:
        """Discordクライアントの作成"""
        intents = discord.Intents.default()
        intents.message_content = True
        return discord.Client(intents=intents)


    def run(self, api_key: str, target_channels: set[int] | None = None):
        """ボットの実行"""
        self.target_channels = target_channels or set()

        # --- イベントハンドラーの登録 ---
        self.client.event(self.on_ready)
        self.client.event(self.on_message)

        self.client.run(api_key)


    async def on_ready(self):
        """ボット起動時の処理"""
        logger.info("We have logged in as {0.user}".format(self.client))


    async def on_message(self, message: discord.Message):
        """メッセージ受信時の処理"""
        if self.ignore_message(message):
            logger.info("ignore message: ch=%s %s: %s", message.channel.id, message.author, message.content)
            return

        res = await self.call_agent(message.content, message.channel.id)
        await _send_response(message.channel.send, res, self, message.channel.id)


    def ignore_message(self, message: discord.Message) -> bool:
        """メッセージを無視するかどうかの判定"""
        if message.author == self.client.user:
            return True
        # 対象チャンネルなら常に反応
        if message.channel.id in self.target_channels:
            return False
        if not message.content.startswith("!claude"):
            return True
        return False


    async def call_agent(self, content: str, channel_id: int) -> AgentResponse:
        """エージェントの呼び出し"""
        # --- ここでエージェントを呼び出す ---
        return await agent.call(content, channel_id)
