import atexit
import os
import signal
import sys
import time

import app

from pydantic import ConfigDict, Field
from pydantic_settings import BaseSettings

PID_FILE = "/tmp/discord-ai-bot.pid"


class Settings(BaseSettings):
    discord_api_key: str
    target_channel_ids: str = ""
    model_config = ConfigDict(
        env_file=".env",
        extra="ignore",
    )

    def get_target_channel_ids(self) -> set[int]:
        if not self.target_channel_ids:
            return set()
        return {int(ch.strip()) for ch in self.target_channel_ids.split(",") if ch.strip()}


def kill_other_instances():
    """自分以外のpython3 main.pyプロセスをすべてkillする"""
    my_pid = os.getpid()
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid == my_pid:
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmdline = f.read().decode(errors="ignore")
            if "python3" in cmdline and "main.py" in cmdline:
                print(f"Killing old bot process (PID {pid})...")
                os.kill(pid, signal.SIGKILL)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass


def check_pid_file():
    """PIDファイルで多重起動を防止"""
    kill_other_instances()
    time.sleep(1)

    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))
    atexit.register(lambda: os.remove(PID_FILE) if os.path.exists(PID_FILE) else None)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    _logger = logging.getLogger(__name__)

    check_pid_file()
    try:
        settings = Settings()
        _logger.info("Bot starting with PID %s", os.getpid())
        app.caller.run(
            api_key=settings.discord_api_key,
            target_channels=settings.get_target_channel_ids(),
        )
        _logger.warning("Bot has exited normally")
    except Exception:
        _logger.exception("Bot crashed with unhandled exception")
        sys.exit(1)
