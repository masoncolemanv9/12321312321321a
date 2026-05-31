"""Pins the inline-keyboard layout for the main menu + settings.

User-requested invariants:

* ⚙️ Настройки must be the LAST row of the main menu (so it never
  hides behind admin-only install batches).
* The main menu must include shortcuts for 🐙 GitHub проекты and
  🖥 Терминал — they used to live only in the Telegram popup menu.
* The settings sub-menu must surface 📚 Узнать о функционале (LLM
  tutorial) and 🖥 Терминал alongside the existing items.
"""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture()
def wizard(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BOT_PERSONA", "boss")
    monkeypatch.setenv("BOT_OWNER_USER_ID", "12345")
    import sys

    for mod in ("bot.config", "bot.storage", "bot.persona", "bot.wizard"):
        sys.modules.pop(mod, None)
    storage_mod = importlib.import_module("bot.storage")
    config = importlib.import_module("bot.config")
    storage_mod.storage = storage_mod.Storage(data_dir=config.DATA_DIR)
    storage_mod.storage.set_owner_id(12345)
    wiz = importlib.import_module("bot.wizard")
    wiz.storage = storage_mod.storage
    return wiz


def _labels(keyboard):
    return [btn.text for row in keyboard.inline_keyboard for btn in row]


def _callbacks(keyboard):
    return [btn.callback_data for row in keyboard.inline_keyboard for btn in row]


def test_main_menu_has_github_and_terminal(wizard):
    kb = wizard._kb_main_after_claim(user_id=12345)
    cbs = _callbacks(kb)
    assert "main:github" in cbs
    assert "main:terminal" in cbs


def test_settings_is_last_row_of_main_menu(wizard):
    kb_admin = wizard._kb_main_after_claim(user_id=12345)
    assert kb_admin.inline_keyboard[-1][0].callback_data == "main:settings"

    # Same invariant for the guest-mode rendering.
    kb_guest = wizard._kb_main_after_claim(user_id=None)
    assert kb_guest.inline_keyboard[-1][0].callback_data == "main:settings"


def test_settings_menu_has_learn_and_terminal(wizard):
    kb = wizard._kb_settings_menu(user_id=12345)
    cbs = _callbacks(kb)
    assert "main:learn" in cbs
    assert "main:terminal" in cbs
    # And the existing items are still present.
    assert "roles:menu" in cbs
    assert "tts:menu" in cbs
    assert "gh:menu" in cbs
    # «Назад» row stays at the very bottom of settings.
    assert kb.inline_keyboard[-1][0].callback_data == "settings:back"


def test_learn_system_prompt_mentions_features(wizard):
    """The injected system prompt must actually describe what the bot
    can do — otherwise the LLM has nothing to teach the user."""
    body = wizard._learn_system_prompt()
    # A handful of feature names must appear so the LLM has anchors.
    for keyword in [
        "Helpzavr",
        "Красивый текст",
        "Голосовой ответчик",
        "Соображалка",
        "Терминал",
        "GitHub",
    ]:
        assert keyword in body, f"system prompt missing keyword: {keyword}"


def test_learn_intro_explains_exit(wizard):
    intro = wizard._learn_intro_body()
    assert "/cancel" in intro
