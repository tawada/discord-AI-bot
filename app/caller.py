import asyncio
import os
import re

import discord
from discord import ui
import logging

from app import agent
from app.agent import AgentResponse
from dotenv import dotenv_values

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# チャンネルごとのロック（同一チャンネルでのclaude多重起動を防止）
_channel_locks: dict[int, asyncio.Lock] = {}

# Discordのアップロード上限 (Nitroなしで25MB)
_MAX_FILE_SIZE = 25 * 1024 * 1024
# Discordのメッセージ文字数上限
_MAX_MESSAGE_LENGTH = 2000

# Yes/No質問の判定パターン
_YESNO_PATTERN = re.compile(
    r"(よろしいですか|しますか|しましょうか|いいですか|ですか？|ますか？"
    r"|shall I|should I|do you want|would you like|yes or no|proceed\?)",
    re.IGNORECASE,
)


def _split_message(text: str) -> list[str]:
    """テキストを2000文字ごとに分割する。コードブロック(```)が途中で切れる場合は閉じ/再開を補う。"""
    chunks = []
    while text:
        if len(text) <= _MAX_MESSAGE_LENGTH:
            chunks.append(text)
            break
        # 改行で区切れる位置を探す
        split_at = text.rfind("\n", 0, _MAX_MESSAGE_LENGTH)
        if split_at <= 0:
            split_at = _MAX_MESSAGE_LENGTH
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")

    # コードブロックの開閉を補正
    in_code_block = False
    code_fence = ""
    for i, chunk in enumerate(chunks):
        patched = chunk
        if in_code_block:
            patched = code_fence + "\n" + patched
        # このチャンク内の```を数えて、開閉状態を追跡
        fence_count = 0
        last_fence = ""
        for line in chunk.split("\n"):
            stripped = line.lstrip()
            if stripped.startswith("```"):
                fence_count += 1
                last_fence = stripped
        if fence_count % 2 == 1:
            # 奇数個 = 状態反転
            if not in_code_block:
                in_code_block = True
                code_fence = last_fence.split()[0] if last_fence else "```"
                patched += "\n```"
            else:
                in_code_block = False
                code_fence = ""
        elif in_code_block:
            # 偶数個でブロック内のまま = 閉じて開き直す
            patched += "\n```"
        chunks[i] = patched

    return chunks


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

    # テキストを2000文字ごとに分割
    chunks = _split_message(text)

    # 最後のチャンクにファイルとViewを添付
    for i, chunk in enumerate(chunks):
        kwargs = {}
        is_last = i == len(chunks) - 1
        if is_last and valid_files:
            kwargs["files"] = valid_files
        if is_last and _YESNO_PATTERN.search(res.text):
            kwargs["view"] = YesNoView(caller, channel_id)
        try:
            await send_func(chunk, **kwargs)
        except discord.HTTPException as e:
            logger.error("Failed to send message: %s", e)
            try:
                await send_func(f"⚠ メッセージの送信に失敗しました: {e.status} {e.text}")
            except discord.HTTPException:
                pass
            break


class YesNoView(ui.View):
    """Yes/Noボタンを表示するView"""

    def __init__(self, caller: "Caller", channel_id: int):
        super().__init__(timeout=300)
        self.caller = caller
        self.channel_id = channel_id
        self._handled = False

    @ui.button(label="Yes", style=discord.ButtonStyle.success)
    async def yes_button(self, interaction: discord.Interaction, button: ui.Button):
        await self._handle(interaction, "Yes")

    @ui.button(label="No", style=discord.ButtonStyle.danger)
    async def no_button(self, interaction: discord.Interaction, button: ui.Button):
        await self._handle(interaction, "No")

    async def _handle(self, interaction: discord.Interaction, answer: str):
        if self._handled:
            await interaction.response.defer()
            return
        self._handled = True
        self.stop()
        for child in self.children:
            child.disabled = True
        # 即座にインタラクションに応答してタイムアウトを防ぐ（ロックの外で行う）
        try:
            await interaction.response.edit_message(view=self)
        except discord.HTTPException:
            try:
                await interaction.response.defer()
            except discord.HTTPException:
                pass

        # Claude呼び出しは時間がかかるのでチャンネルに直接送信
        channel = interaction.channel
        send_func = channel.send if channel else interaction.followup.send

        lock = _channel_locks.setdefault(self.channel_id, asyncio.Lock())
        async with lock:
            res = await self.caller.call_agent(answer, self.channel_id)
            await _send_response(send_func, res, self.caller, self.channel_id)


class Caller:
    """Discordボットの呼び出しクラス"""

    ENV_FILE = "/workspace/discord-AI-bot/.env"

    def __init__(self):
        self.client = self.setup_client()
        self.target_channels: set[int] = set()

    def reload_target_channels(self) -> set[int]:
        """.envからTARGET_CHANNEL_IDSを再読み込みして反映する"""
        env = dotenv_values(self.ENV_FILE)
        raw = env.get("TARGET_CHANNEL_IDS", "")
        self.target_channels = {int(ch.strip()) for ch in raw.split(",") if ch.strip()}
        return self.target_channels


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
        self.client.event(self.on_disconnect)
        self.client.event(self.on_resumed)
        self.client.event(self.on_error)

        logger.info("Starting bot...")
        self.client.run(api_key, log_handler=None)
        logger.warning("client.run() has returned — bot has stopped")


    async def on_ready(self):
        """ボット起動時の処理"""
        logger.info("We have logged in as {0.user}".format(self.client))

    async def on_disconnect(self):
        """Gateway切断時"""
        logger.warning("Disconnected from Discord Gateway")

    async def on_resumed(self):
        """Gateway再接続時"""
        logger.info("Resumed Discord Gateway connection")

    async def on_error(self, event: str, *args, **kwargs):
        """イベントハンドラの未処理例外"""
        logger.exception("Unhandled exception in event: %s", event)


    async def on_message(self, message: discord.Message):
        """メッセージ受信時の処理"""
        if self.ignore_message(message):
            logger.info("ignore message: ch=%s %s: %s", message.channel.id, message.author, message.content)
            return

        # !help: コマンド一覧を表示
        if message.content.strip() == "!help":
            help_text = (
                "**コマンド一覧**\n"
                "`!claude <メッセージ>` — Claudeに質問する（対象チャンネル外で使用）\n"
                "`!agent claude` / `!agent opencode` — エージェントの切り替え\n"
                "`!reset` / `!clear` / `!forget` — 会話の記憶をリセット\n"
                "`!reload` — .envからターゲットチャンネル設定を再読み込み\n"
                "`!help` — このヘルプを表示\n"
                "\n対象チャンネルではプレフィックスなしで会話できます。"
            )
            await message.channel.send(help_text)
            return

        # !agent: エージェントの切り替え
        if message.content.strip().startswith("!agent"):
            parts = message.content.strip().split()
            if len(parts) < 2:
                current = agent.get_agent_type(message.channel.id)
                await message.channel.send(
                    f"現在のエージェント: **{current}**\n"
                    f"使い方: `!agent claude` / `!agent opencode`"
                )
                return
            name = parts[1].lower()
            aliases = {
                "claude": "claudecode", "claudecode": "claudecode", "claude-code": "claudecode",
                "opencode": "opencode", "open-code": "opencode", "oc": "opencode",
            }
            agent_type = aliases.get(name)
            if not agent_type:
                await message.channel.send(f"不明なエージェント: {name}\n選択肢: claude / opencode")
                return
            res = agent.set_agent_type(message.channel.id, agent_type)
            await message.channel.send(res.text)
            return

        # !reload: .envからターゲットチャンネルを再読み込み
        if message.content.strip() == "!reload":
            channels = self.reload_target_channels()
            ch_list = ", ".join(str(c) for c in sorted(channels)) or "(なし)"
            await message.channel.send(f"設定を再読み込みしました。\nターゲットチャンネル: {ch_list}")
            return

        lock = _channel_locks.setdefault(message.channel.id, asyncio.Lock())
        async with lock:
            try:
                res = await self.call_agent(message.content, message.channel.id)
                await _send_response(message.channel.send, res, self, message.channel.id)
            except Exception:
                logger.exception("Error handling message: ch=%s %s", message.channel.id, message.content[:100])


    def ignore_message(self, message: discord.Message) -> bool:
        """メッセージを無視するかどうかの判定"""
        if message.author == self.client.user:
            return True
        # 対象チャンネルなら常に反応
        if message.channel.id in self.target_channels:
            return False
        if not message.content.startswith(("!claude", "!reset", "!clear", "!forget", "!reload", "!help", "!agent")):
            return True
        return False


    async def call_agent(self, content: str, channel_id: int) -> AgentResponse:
        """エージェントの呼び出し"""
        # --- ここでエージェントを呼び出す ---
        return await agent.call(content, channel_id)
