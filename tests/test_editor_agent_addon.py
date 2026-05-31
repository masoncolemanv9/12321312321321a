"""Tests for the Editor Agent main-menu addon (``bot.addons.editor_agent``).

The addon is a thin runtime-config UI over the v1/v2/v2.1/v6 dispatch
the worker factory uses (see ``bot/workers/__init__.py``). Tests here
exercise:

* Effective-state resolution (storage override beats env default).
* Toggle handlers actually mutate storage AND mirror values into the
  live ``bot.config`` module so the worker factory picks them up on
  the next call.
* The reset handler clears storage and re-reads env-defaults.
* The screen text & keyboard reflect the current effective state.
"""

from __future__ import annotations

import asyncio
import importlib
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture()
def fresh_addon(tmp_path, monkeypatch):
    """Return ``(storage, addon_state, handlers)`` rebound to a tmp data dir.

    Env defaults are set explicitly so each test starts from a known
    baseline regardless of the host shell.
    """
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("EDITOR_VERSION", "v1")
    monkeypatch.setenv("EDITOR_PROFILE", "light")
    monkeypatch.setenv("EDITOR_V6_ENABLED", "false")
    import sys

    for mod in (
        "bot.config",
        "bot.storage",
        "bot.addons.state",
        "bot.addons.editor_agent",
        "bot.addons.editor_agent.handlers",
    ):
        sys.modules.pop(mod, None)
    config = importlib.import_module("bot.config")
    storage_mod = importlib.import_module("bot.storage")
    storage_mod.storage = storage_mod.Storage(data_dir=config.DATA_DIR)
    addon_state = importlib.import_module("bot.addons.state")
    handlers = importlib.import_module("bot.addons.editor_agent.handlers")
    return storage_mod.storage, addon_state, handlers


def test_defaults_follow_env_when_storage_unset(fresh_addon) -> None:
    _, _, handlers = fresh_addon
    assert handlers.current_version() == "v1"
    assert handlers.current_profile() == "light"
    assert handlers.is_v6_enabled() is False


def test_storage_override_beats_env(fresh_addon) -> None:
    _, addon_state, handlers = fresh_addon
    addon_state.set_("editor_agent", "version_override", "v6")
    addon_state.set_("editor_agent", "profile", "heavy")
    addon_state.set_("editor_agent", "v6_enabled", True)
    assert handlers.current_version() == "v6"
    assert handlers.current_profile() == "heavy"
    assert handlers.is_v6_enabled() is True


def test_invalid_override_falls_back_to_env(fresh_addon) -> None:
    _, addon_state, handlers = fresh_addon
    addon_state.set_("editor_agent", "version_override", "garbage")
    addon_state.set_("editor_agent", "profile", "ultra-violence")
    assert handlers.current_version() == "v1"
    assert handlers.current_profile() == "light"


def test_screen_body_lists_all_three_knobs(fresh_addon) -> None:
    _, _, handlers = fresh_addon
    body = handlers._screen_body()
    # Все три ручки и их текущие значения упомянуты явно.
    assert "v1" in body
    assert "light" in body
    assert "ВЫКЛ" in body
    # Опции тоже описаны, чтобы пользователь знал что выбирает.
    assert "creative planner" in body.lower()
    assert "light" in body and "medium" in body and "heavy" in body


def test_keyboard_has_all_version_and_profile_buttons(fresh_addon) -> None:
    _, _, handlers = fresh_addon
    kb = handlers._kb_screen()
    callbacks: list[str] = []
    for row in kb.inline_keyboard:
        for btn in row:
            if btn.callback_data:
                callbacks.append(btn.callback_data)
    for v in ("v1", "v2", "v2.1", "v6"):
        assert f"editor_agent:ver:{v}" in callbacks
    for p in ("light", "medium", "heavy"):
        assert f"editor_agent:prof:{p}" in callbacks
    # v6 toggle (one or the other depending on state) + reset + back.
    assert any(c.startswith("editor_agent:v6:") for c in callbacks)
    assert "editor_agent:reset" in callbacks
    assert "editor_agent:back" in callbacks


def test_set_version_writes_storage_and_live_config(fresh_addon) -> None:
    _, _, handlers = fresh_addon
    router = handlers.build_editor_agent_router()
    # Find the version handler. aiogram exposes them via ``router.observers``
    # — but for this unit test we just call the implementation directly
    # by reading the router's registered handlers map. Simpler:
    # call the inner functions via a synthetic callback.
    query = _make_query(data="editor_agent:ver:v6", chat_id=42)
    asyncio.run(_dispatch_callback(router, query))
    from bot import config as bot_config

    assert bot_config.EDITOR_VERSION == "v6"
    assert handlers.current_version() == "v6"


def test_set_profile_writes_storage_and_live_config(fresh_addon) -> None:
    _, _, handlers = fresh_addon
    router = handlers.build_editor_agent_router()
    query = _make_query(data="editor_agent:prof:heavy", chat_id=42)
    asyncio.run(_dispatch_callback(router, query))
    from bot import config as bot_config

    assert bot_config.EDITOR_PROFILE == "heavy"
    assert handlers.current_profile() == "heavy"


def test_v6_on_off_round_trip(fresh_addon) -> None:
    _, _, handlers = fresh_addon
    router = handlers.build_editor_agent_router()
    asyncio.run(
        _dispatch_callback(router, _make_query(data="editor_agent:v6:on", chat_id=42))
    )
    assert handlers.is_v6_enabled() is True
    asyncio.run(
        _dispatch_callback(router, _make_query(data="editor_agent:v6:off", chat_id=42))
    )
    assert handlers.is_v6_enabled() is False


def test_reset_clears_overrides(fresh_addon) -> None:
    _, addon_state, handlers = fresh_addon
    addon_state.set_("editor_agent", "version_override", "v6")
    addon_state.set_("editor_agent", "profile", "heavy")
    addon_state.set_("editor_agent", "v6_enabled", True)
    assert handlers.current_version() == "v6"

    router = handlers.build_editor_agent_router()
    asyncio.run(
        _dispatch_callback(router, _make_query(data="editor_agent:reset", chat_id=42))
    )
    # After reset, current_* falls back to env defaults.
    assert handlers.current_version() == "v1"
    assert handlers.current_profile() == "light"
    assert handlers.is_v6_enabled() is False


def test_unknown_version_callback_does_not_corrupt_state(fresh_addon) -> None:
    _, _, handlers = fresh_addon
    router = handlers.build_editor_agent_router()
    asyncio.run(
        _dispatch_callback(
            router, _make_query(data="editor_agent:ver:garbage", chat_id=42)
        )
    )
    # Still at env default, storage untouched.
    assert handlers.current_version() == "v1"


def test_factory_picks_up_runtime_change(fresh_addon) -> None:
    """The integrated v1 → v6 flow: toggling via UI must change which
    worker the factory hands out without a process restart.
    """
    _, _, handlers = fresh_addon
    from bot.workers import build_editor_worker
    from bot.workers.editor import EditorWorker
    from bot.workers.editor_v2 import EditorV2Worker

    # Baseline: env default = v1 → EditorWorker.
    queue = MagicMock()
    assert isinstance(build_editor_worker(queue), EditorWorker)

    # Click the v6 button.
    router = handlers.build_editor_agent_router()
    asyncio.run(
        _dispatch_callback(router, _make_query(data="editor_agent:ver:v6", chat_id=42))
    )
    assert isinstance(build_editor_worker(queue), EditorV2Worker)


# ---- helpers -------------------------------------------------------------


def _make_query(*, data: str, chat_id: int = 42) -> Any:
    """Build a CallbackQuery-shaped mock the addon router can use."""
    query = MagicMock()
    query.data = data
    query.from_user = MagicMock()
    query.from_user.id = 12345
    query.message = MagicMock()
    query.message.chat = MagicMock()
    query.message.chat.id = chat_id
    query.message.edit_text = AsyncMock()
    query.message.answer = AsyncMock()
    query.answer = AsyncMock()
    return query


async def _dispatch_callback(router: Any, query: Any) -> None:
    """Walk router.observers and run the matching callback handler.

    aiogram's ``Router.callback_query`` decorator registers handlers
    under ``observers['callback_query']``; each handler has filters
    that include the ``F.data == ...`` or ``F.data.startswith(...)``
    we used. For tests we bypass the filter machinery: we just call
    every registered callback and rely on the handler's own internal
    branching (or filter mismatch) to make it a no-op for the wrong
    data — for our addon, each handler checks its own data prefix.
    """
    observers = router.observers
    cb_observer = observers.get("callback_query")
    if cb_observer is None:
        return
    for handler_obj in cb_observer.handlers:
        try:
            for f in handler_obj.filters or []:
                callback = getattr(f, "callback", None)
                if callback is None:
                    continue
                result = callback(query)
                if asyncio.iscoroutine(result):
                    result = await result
                if not result:
                    break
            else:
                # All filters matched → invoke handler.
                await handler_obj.callback(query)
                return
        except Exception:  # noqa: BLE001
            continue
