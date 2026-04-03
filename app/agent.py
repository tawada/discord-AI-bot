import asyncio
import json
import logging
import uuid

logger = logging.getLogger(__name__)

# チャンネルごとのセッションID管理（初回は新規作成、2回目以降はresumeで文脈維持）
_channel_sessions: dict[int, str] = {}


async def call(message: str, channel_id: int) -> str:
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

    if session_id:
        # 2回目以降: --resume で文脈を維持
        cmd = [
            "claude", "--dangerously-skip-permissions", "-p", message,
            "--resume", session_id, "--output-format", "json",
        ]
    else:
        # 初回: ランダムUUIDで新規セッション
        session_id = str(uuid.uuid4())
        cmd = [
            "claude", "--dangerously-skip-permissions", "-p", message,
            "--session-id", session_id, "--output-format", "json",
        ]

    stdout, stderr, returncode = await _run_claude(cmd)
    logger.info("claude exit=%s stdout=%s stderr=%s", returncode, stdout[:200], stderr[:200])

    if returncode != 0:
        # セッションエラーの場合、新規セッションでリトライ
        logger.warning("session error, retrying with new session")
        session_id = str(uuid.uuid4())
        cmd = [
            "claude", "--dangerously-skip-permissions", "-p", message,
            "--session-id", session_id, "--output-format", "json",
        ]
        stdout, stderr, returncode = await _run_claude(cmd)

    _channel_sessions[channel_id] = session_id

    # JSON出力からresultを取得
    try:
        data = json.loads(stdout)
        return data.get("result", stdout)
    except (json.JSONDecodeError, TypeError):
        return stdout if stdout else stderr
