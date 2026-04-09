import asyncio
import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# チャンネルごとのセッションID管理（初回は新規作成、2回目以降はresumeで文脈維持）
_channel_sessions: dict[int, str] = {}

_SYSTEM_PROMPT = (
    "あなたはDiscordボット経由でユーザーと会話しています。"
    "画像や動画ファイルを生成・取得した場合、応答テキスト中にそのファイルの絶対パスを必ず含めてください。"
    "パスが含まれていれば自動的にDiscordにアップロードされます。"
    "対応形式: png, jpg, jpeg, gif, webp, bmp, mp4, mov, avi, webm, mkv, wav, mp3, ogg, flac"
)

# ファイルパス検出パターン
_FILE_PATH_PATTERN = re.compile(r"(/[\w./-]+\.(?:png|jpg|jpeg|gif|webp|bmp|mp4|mov|avi|webm|mkv|wav|mp3|ogg|flac))", re.IGNORECASE)

# 対応する拡張子
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".webm", ".mkv"}
_AUDIO_EXTS = {".wav", ".mp3", ".ogg", ".flac"}
_MEDIA_EXTS = _IMAGE_EXTS | _VIDEO_EXTS | _AUDIO_EXTS


@dataclass
class AgentResponse:
    text: str
    files: list[str] = field(default_factory=list)


async def call(message: str, channel_id: int) -> AgentResponse:
    """Dispatch the call to the appropriate agent based on the type of the agent."""

    # ClaudeCodeのみ
    return await call_claudecode(message, channel_id)


async def _run_claude(cmd: list[str]) -> tuple[str, str, int]:
    process = await asyncio.create_subprocess_exec(
        *cmd,
        cwd="/workspace",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    return stdout.decode(), stderr.decode(), process.returncode


async def call_claudecode(message: str, channel_id: int) -> str:
    """Call an agent with the given arguments."""
    session_id = _channel_sessions.get(channel_id)

    base_cmd = [
        "claude", "--dangerously-skip-permissions", "-p", message,
        "--append-system-prompt", _SYSTEM_PROMPT,
        "--output-format", "json",
    ]

    if session_id:
        # 2回目以降: --resume で文脈を維持
        cmd = base_cmd + ["--resume", session_id]
    else:
        # 初回: ランダムUUIDで新規セッション
        session_id = str(uuid.uuid4())
        cmd = base_cmd + ["--session-id", session_id]

    stdout, stderr, returncode = await _run_claude(cmd)
    logger.info("claude exit=%s stdout=%s stderr=%s", returncode, stdout[:200], stderr[:200])

    if returncode != 0:
        # セッションエラーの場合、新規セッションでリトライ
        logger.warning("session error, retrying with new session")
        session_id = str(uuid.uuid4())
        retry_cmd = [
            "claude", "--dangerously-skip-permissions", "-p", message,
            "--append-system-prompt", _SYSTEM_PROMPT,
            "--output-format", "json",
            "--session-id", session_id,
        ]
        stdout, stderr, returncode = await _run_claude(retry_cmd)

    _channel_sessions[channel_id] = session_id

    # JSON出力からresultを取得
    try:
        data = json.loads(stdout)
        text = data.get("result", stdout)
    except (json.JSONDecodeError, TypeError):
        text = stdout if stdout else stderr

    # テキスト中のファイルパスを検出し、存在するメディアファイルを収集
    files = []
    for match in _FILE_PATH_PATTERN.findall(text):
        ext = os.path.splitext(match)[1].lower()
        if ext in _MEDIA_EXTS and os.path.isfile(match):
            files.append(match)

    return AgentResponse(text=text, files=files)
