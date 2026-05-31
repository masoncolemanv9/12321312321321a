"""Tests for the OpenRouter heartbeat module (storage toggle + ping path)."""

from __future__ import annotations

import asyncio
import importlib
from unittest.mock import AsyncMock, patch

import pytest


@pytest.fixture()
def fresh_storage(tmp_path, monkeypatch):
    """Clean Storage + reset heartbeat module so the toggle starts off."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    import sys

    for mod in (
        "bot.config",
        "bot.storage",
        "bot.token_tracker",
        "bot.heartbeat",
    ):
        sys.modules.pop(mod, None)
    config = importlib.import_module("bot.config")
    storage_mod = importlib.import_module("bot.storage")
    storage_mod.storage = storage_mod.Storage(data_dir=config.DATA_DIR)
    heartbeat = importlib.import_module("bot.heartbeat")
    return storage_mod.storage, heartbeat


def test_default_is_off(fresh_storage) -> None:
    storage, _ = fresh_storage
    assert storage.get_heartbeat_enabled() is False


def test_toggle_persists(fresh_storage, tmp_path) -> None:
    storage, _ = fresh_storage
    storage.set_heartbeat_enabled(True)
    assert storage.get_heartbeat_enabled() is True
    # Restart simulation.
    from bot.storage import Storage

    storage2 = Storage(data_dir=tmp_path)
    assert storage2.get_heartbeat_enabled() is True


def test_ping_skipped_when_no_key(fresh_storage) -> None:
    """Heartbeat with no OpenRouter key should silently no-op."""
    _, heartbeat = fresh_storage
    result = asyncio.run(heartbeat._ping_once())
    assert result is False


def test_ping_calls_openrouter_and_logs_tokens(fresh_storage) -> None:
    """When a key is present, ping should fire one chat-completion and log usage."""
    storage, heartbeat = fresh_storage
    storage.set_provider_key("openrouter", "sk-or-test")

    # Mock the AsyncOpenAI response shape.
    fake_usage = type("U", (), {"prompt_tokens": 4, "completion_tokens": 1})()
    fake_resp = type(
        "R",
        (),
        {"usage": fake_usage, "choices": [type("C", (), {"message": "pong"})()]},
    )()
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=fake_resp)

    with patch("bot.heartbeat.AsyncOpenAI", return_value=mock_client):
        ok = asyncio.run(heartbeat._ping_once())
    assert ok is True

    log = storage.get_token_log()
    assert len(log) == 1
    assert log[0]["purpose"] == "heartbeat"
    assert log[0]["prompt_tokens"] == 4
    assert log[0]["completion_tokens"] == 1


def test_ping_swallows_api_error(fresh_storage) -> None:
    """An OpenRouter outage must NOT crash the bot — return False, log nothing.

    We use a generic ``Exception`` rather than ``openai.APIError`` because
    constructing the latter requires a real ``httpx.Request``; the heartbeat's
    catch-all ``except Exception`` clause covers both code paths.
    """
    storage, heartbeat = fresh_storage
    storage.set_provider_key("openrouter", "sk-or-test")

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("boom"))
    with patch("bot.heartbeat.AsyncOpenAI", return_value=mock_client):
        ok = asyncio.run(heartbeat._ping_once())
    assert ok is False
    assert storage.get_token_log() == []
