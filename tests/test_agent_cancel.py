"""Tests for the cancellation path in :func:`bot.agent.run_agent`.

The handler attaches a «🛑 Отмена» button keyed to a per-run
``asyncio.Event``. When the user clicks it, the event is set; the
agent loop polls it between iterations and aborts cleanly.

We don't exercise the real OpenAI call here — we just monkey-patch
``_call_model_with_failover`` to assert the cancel check fires before
any model traffic.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import bot.agent as agent
from bot.storage import Storage


@pytest.fixture
def patched_storage(tmp_path: Path, monkeypatch) -> Storage:
    s = Storage(data_dir=tmp_path)
    # Mark the slot as configured so the up-front _build_client() check
    # in run_agent doesn't raise NoApiKeyError before the cancel path
    # even gets to run.
    s.set_provider("openrouter")
    s.set_provider_key("openrouter", "sk-or-test-aaaa")
    monkeypatch.setattr(agent, "storage", s)
    return s


@pytest.mark.asyncio
async def test_run_agent_returns_cancelled_when_event_preset(
    patched_storage, monkeypatch
) -> None:
    """If the cancel event is already set, the agent returns the
    «Задача отменена» short-circuit on its first iteration BEFORE
    making any model request."""
    called: dict[str, int] = {"model": 0}

    async def fake_model_call(_messages):
        called["model"] += 1
        raise AssertionError("model should not be called when cancelled")

    monkeypatch.setattr(agent, "_call_model_with_failover", fake_model_call)

    event = asyncio.Event()
    event.set()
    result = await agent.run_agent(
        user_id=1,
        user_text="hi",
        cwd=None,
        cancel_event=event,
    )
    assert "Задача отменена" in result
    assert called["model"] == 0


@pytest.mark.asyncio
async def test_run_agent_runs_normally_without_cancel_event(
    patched_storage, monkeypatch
) -> None:
    """Sanity: without a cancel_event the loop reaches the model call.

    We patch the model to return an assistant message with no tool
    calls so the loop terminates after one iteration.
    """

    class _FakeMsg:
        content = "hello world"
        tool_calls = None

    async def fake_model_call(_messages):
        return _FakeMsg(), "test-model", "1"

    monkeypatch.setattr(agent, "_call_model_with_failover", fake_model_call)
    result = await agent.run_agent(user_id=2, user_text="ping", cwd=None)
    assert "hello world" in result


@pytest.mark.asyncio
async def test_run_agent_refuses_when_ram_above_threshold(
    patched_storage, monkeypatch
) -> None:
    """RAM-guard short-circuit fires before the model is called when
    RSS / limit ≥ 95% in compress mode."""
    patched_storage.set_ram_limit_mb(500)
    patched_storage.set_ram_behavior("compress")
    monkeypatch.setattr(agent, "_current_rss_mb", lambda: 480)

    async def fake_model_call(_messages):
        raise AssertionError("model should not be called when over limit")

    monkeypatch.setattr(agent, "_call_model_with_failover", fake_model_call)
    result = await agent.run_agent(user_id=3, user_text="hi", cwd=None)
    assert "на пределе" in result or "Память сервера" in result
