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
# チャンネルごとのエージェント種別（デフォルト: claudecode）
_channel_agent_type: dict[int, str] = {}

AGENT_TYPES = ("claudecode", "opencode")

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


def _extract_media_files(text: str) -> list[str]:
    """テキスト中のファイルパスを検出し、存在するメディアファイルを収集"""
    files = []
    for match in _FILE_PATH_PATTERN.findall(text):
        ext = os.path.splitext(match)[1].lower()
        if ext in _MEDIA_EXTS and os.path.isfile(match):
            files.append(match)
    return files


def get_agent_type(channel_id: int) -> str:
    """チャンネルのエージェント種別を取得"""
    return _channel_agent_type.get(channel_id, "claudecode")


def set_agent_type(channel_id: int, agent_type: str) -> AgentResponse:
    """チャンネルのエージェント種別を切り替え、セッションをリセットする"""
    if agent_type not in AGENT_TYPES:
        return AgentResponse(text=f"不明なエージェント: {agent_type}\n選択肢: {', '.join(AGENT_TYPES)}")
    _channel_agent_type[channel_id] = agent_type
    _channel_sessions.pop(channel_id, None)
    return AgentResponse(text=f"エージェントを **{agent_type}** に切り替えました。セッションはリセットされます。")


def reset_session(channel_id: int) -> AgentResponse:
    """チャンネルのセッションをリセットする"""
    removed = _channel_sessions.pop(channel_id, None)
    if removed:
        return AgentResponse(text="セッションをリセットしました。会話の記憶がクリアされます。")
    return AgentResponse(text="このチャンネルにはアクティブなセッションがありません。")


async def call(message: str, channel_id: int) -> AgentResponse:
    """Dispatch the call to the appropriate agent based on the type of the agent."""

    # リセットコマンド
    if message.strip() in ("!reset", "!clear", "!forget"):
        return reset_session(channel_id)

    agent_type = get_agent_type(channel_id)
    if agent_type == "opencode":
        return await call_opencode(message, channel_id)
    return await call_claudecode(message, channel_id)


def _get_channel_workdir(channel_id: int) -> str:
    """チャンネルごとの作業ディレクトリを取得・作成する"""
    workdir = f"/workspace/channels/{channel_id}"
    os.makedirs(workdir, exist_ok=True)
    return workdir


async def _run_subprocess(cmd: list[str], channel_id: int) -> tuple[str, str, int]:
    """サブプロセスを実行し、stdout, stderr, returncodeを返す"""
    process = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=_get_channel_workdir(channel_id),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    return stdout.decode(), stderr.decode(), process.returncode


async def call_claudecode(message: str, channel_id: int) -> AgentResponse:
    """Claude Codeエージェントを呼び出す"""
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

    stdout, stderr, returncode = await _run_subprocess(cmd, channel_id)
    logger.info("claude exit=%s stdout=%s stderr=%s", returncode, stdout[:200], stderr[:200])

    if returncode != 0:
        # セッションエラーの場合、新規セッションでリトライ
        logger.warning("claude session error, retrying with new session")
        session_id = str(uuid.uuid4())
        retry_cmd = [
            "claude", "--dangerously-skip-permissions", "-p", message,
            "--append-system-prompt", _SYSTEM_PROMPT,
            "--output-format", "json",
            "--session-id", session_id,
        ]
        stdout, stderr, returncode = await _run_subprocess(retry_cmd, channel_id)

    _channel_sessions[channel_id] = session_id

    # JSON出力からresultを取得
    try:
        data = json.loads(stdout)
        text = data.get("result", stdout)
    except (json.JSONDecodeError, TypeError):
        text = stdout if stdout else stderr

    return AgentResponse(text=text, files=_extract_media_files(text))


async def call_opencode(message: str, channel_id: int) -> AgentResponse:
    """OpenCodeエージェントを呼び出す"""
    session_id = _channel_sessions.get(channel_id)

    # OpenCodeにはsystem prompt用フラグがないのでメッセージに前置
    full_message = f"[SYSTEM]\n{_SYSTEM_PROMPT}\n[/SYSTEM]\n\n{message}"

    base_cmd = [
        "opencode", "run", full_message,
        "--format", "json",
        "--dangerously-skip-permissions",
    ]

    if session_id:
        cmd = base_cmd + ["--session", session_id]
    else:
        session_id = str(uuid.uuid4())
        cmd = base_cmd + ["--session", session_id]

    stdout, stderr, returncode = await _run_subprocess(cmd, channel_id)
    logger.info("opencode exit=%s stdout=%s stderr=%s", returncode, stdout[:200], stderr[:200])

    if returncode != 0:
        # セッションエラーの場合、新規セッションでリトライ
        logger.warning("opencode session error, retrying with new session")
        session_id = str(uuid.uuid4())
        retry_cmd = [
            "opencode", "run", full_message,
            "--format", "json",
            "--dangerously-skip-permissions",
            "--session", session_id,
        ]
        stdout, stderr, returncode = await _run_subprocess(retry_cmd, channel_id)

    _channel_sessions[channel_id] = session_id

    # JSON出力からresultを取得（OpenCodeのキー名に対応）
    try:
        data = json.loads(stdout)
        text = data.get("result") or data.get("output") or data.get("response") or data.get("message") or str(data)
    except (json.JSONDecodeError, TypeError):
        text = stdout if stdout else stderr

    return AgentResponse(text=text, files=_extract_media_files(text))
