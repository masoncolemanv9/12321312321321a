"""Tests for the /setup API-key flow in bot.wizard.

User-reported bug: clicking "🔑 API ключ" after picking the "Другое"
(custom) provider asked for the key, deleted the user's message, but
silently failed to save the key because ``storage.set_provider_key``
rejected the "custom" label. The fix pins the storage slot to
``openrouter`` for both providers — the agent reads from that single
slot regardless of the active base_url.
"""

from __future__ import annotations

import asyncio
import importlib
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture()
def fresh_modules(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BOT_PERSONA", "boss")
    monkeypatch.setenv("BOT_OWNER_USER_ID", "12345")
    import sys

    for mod in ("bot.config", "bot.storage", "bot.persona", "bot.wizard"):
        sys.modules.pop(mod, None)
    config = importlib.import_module("bot.config")
    storage_mod = importlib.import_module("bot.storage")
    storage_mod.storage = storage_mod.Storage(data_dir=config.DATA_DIR)
    wizard = importlib.import_module("bot.wizard")
    # Wizard's module-level ``storage`` was imported as a name binding —
    # rebind it to the fresh singleton.
    wizard.storage = storage_mod.storage
    # Mark the test user as owner so the wizard accepts their input.
    storage_mod.storage.set_owner_id(12345)
    return storage_mod.storage, wizard


def _make_message(text: str, user_id: int = 12345):
    msg = MagicMock()
    msg.text = text
    msg.from_user = MagicMock()
    msg.from_user.id = user_id
    msg.answer = AsyncMock()
    msg.delete = AsyncMock()
    return msg


def _make_state(slot: str | None = None):
    """Build a minimal aiogram FSMContext stand-in.

    ``capture_api_key`` now reads the active brain slot from FSM data
    so it knows whether the key belongs to Мозг 1 or Мозг 2.
    ``state.get_data()`` therefore has to be awaitable. Default is
    ``{}`` which makes ``capture_api_key`` treat the input as a Brain 1
    key — the legacy behaviour these tests rely on.
    """
    state = MagicMock()
    state.clear = AsyncMock()
    state.update_data = AsyncMock()
    data: dict = {}
    if slot is not None:
        data["brain_cfg_slot"] = slot
    state.get_data = AsyncMock(return_value=data)
    return state


def test_capture_api_key_saves_under_openrouter_slot_for_custom(
    fresh_modules,
) -> None:
    """Custom provider should still persist the key (under openrouter slot)."""
    storage, wizard = fresh_modules
    storage.set_provider("custom")
    storage.set_base_url("https://api.moonshot.ai/v1")

    msg = _make_message("sk-test-custom-key-12345")
    state = _make_state()

    asyncio.run(wizard.capture_api_key(msg, state))

    # Key must be readable by the agent (which queries the openrouter slot).
    assert storage.get_provider_key("openrouter") == "sk-test-custom-key-12345"
    # Provider/base-url must be untouched.
    assert storage.get_provider() == "custom"
    assert storage.get_base_url() == "https://api.moonshot.ai/v1"
    # User's message must be deleted to scrub the secret.
    msg.delete.assert_awaited()
    # State must be cleared so subsequent messages aren't captured as keys.
    state.clear.assert_awaited()


def test_capture_api_key_saves_under_openrouter_slot_for_openrouter(
    fresh_modules,
) -> None:
    """OpenRouter provider still works exactly as before."""
    storage, wizard = fresh_modules
    storage.set_provider("openrouter")

    msg = _make_message("sk-or-v1-real-key")
    state = _make_state()

    asyncio.run(wizard.capture_api_key(msg, state))

    assert storage.get_provider_key("openrouter") == "sk-or-v1-real-key"
    assert storage.get_provider() == "openrouter"
    msg.delete.assert_awaited()


def test_capture_api_key_rejects_empty_input(fresh_modules) -> None:
    """Empty input should ask the user to retry; not delete or save anything."""
    storage, wizard = fresh_modules
    storage.set_provider("custom")

    msg = _make_message("   ")  # whitespace only
    state = _make_state()

    asyncio.run(wizard.capture_api_key(msg, state))

    assert storage.get_provider_key("openrouter") == ""
    msg.delete.assert_not_called()
    state.clear.assert_not_called()
    msg.answer.assert_awaited()
