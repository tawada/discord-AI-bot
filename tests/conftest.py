import pytest


@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    monkeypatch.setenv("DISCORD_API_KEY", "test_discord_key")
    monkeypatch.setenv("OPENAI_API_KEY", "test_openai_key")
    monkeypatch.setenv("GEMINI_API_KEY", "test_gemini_key")
    monkeypatch.setenv("CHANNEL_IDS", "123456789")
    monkeypatch.setenv("ROLE_PROMPT", "Test Role Prompt")
    monkeypatch.setenv("ROLE_NAME", "Test Role Name")
