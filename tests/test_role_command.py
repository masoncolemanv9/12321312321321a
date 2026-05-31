"""Tests for the /role command UI helpers in bot.wizard.

The actual aiogram callback round-trip is exercised in integration; here
we focus on the keyboard builder + text formatter so changes to the
persona roster keep the picker consistent (20 persona buttons + Info +
Back). Heavy mocking of aiogram is intentionally avoided.
"""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture()
def fresh_modules(tmp_path, monkeypatch):
    """Provide a Storage + wizard view freshly bound to a tmp data dir."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BOT_PERSONA", "boss")
    import sys

    for mod in ("bot.config", "bot.storage", "bot.persona", "bot.wizard"):
        sys.modules.pop(mod, None)
    config = importlib.import_module("bot.config")
    storage_mod = importlib.import_module("bot.storage")
    storage_mod.storage = storage_mod.Storage(data_dir=config.DATA_DIR)
    wizard = importlib.import_module("bot.wizard")
    # Claim ownership so menu-builder tests that ask "is this user an
    # admin?" get the same answer they used to before access modes existed.
    storage_mod.storage.set_owner_id(12345)
    return storage_mod.storage, wizard


def _flat_callbacks(markup) -> list[str]:
    out: list[str] = []
    for row in markup.inline_keyboard:
        for btn in row:
            if btn.callback_data:
                out.append(btn.callback_data)
    return out


def test_role_picker_has_20_persona_buttons(fresh_modules) -> None:
    _, wizard = fresh_modules
    markup = wizard._kb_role_picker()
    callbacks = _flat_callbacks(markup)
    picks = [c for c in callbacks if c.startswith("role:pick:")]
    assert len(picks) == 20


def test_role_picker_includes_info_and_back(fresh_modules) -> None:
    _, wizard = fresh_modules
    callbacks = _flat_callbacks(wizard._kb_role_picker())
    assert "role:info" in callbacks
    assert "role:back" in callbacks


def test_role_picker_shows_reset_only_when_override_set(fresh_modules) -> None:
    storage, wizard = fresh_modules
    # No override yet → no reset button.
    callbacks = _flat_callbacks(wizard._kb_role_picker())
    assert "role:reset" not in callbacks
    # Set override → reset button appears.
    storage.set_persona_override("research_lead")
    callbacks = _flat_callbacks(wizard._kb_role_picker())
    assert "role:reset" in callbacks


def test_brain_keyboard_includes_heartbeat_toggle(fresh_modules) -> None:
    _, wizard = fresh_modules
    callbacks = _flat_callbacks(wizard._kb_brain())
    assert "brain:heartbeat_toggle" in callbacks


def test_main_menu_keyboard_has_download_button(fresh_modules) -> None:
    storage, wizard = fresh_modules
    owner_id = storage.get_owner_id()
    # Admin menu (owner) — all settings-y buttons are visible.
    callbacks = _flat_callbacks(wizard._kb_main_after_claim(owner_id))
    assert "main:download" in callbacks
    # «🎭 Сменить роль» duplicated «⚙️ Настройки → 🎭 Роли», so it was
    # removed from the main menu. The picker lives in Settings now.
    assert "main:role" not in callbacks
    assert "main:brain" in callbacks
    assert "main:editor" in callbacks
    assert "main:tokens" in callbacks


def test_main_menu_keyboard_hides_admin_buttons_for_guest(fresh_modules) -> None:
    """In public access mode, a guest (non-owner) sees the feature
    buttons but NOT the admin-only ones (download, brain, tokens).
    """
    storage, wizard = fresh_modules
    storage.set_access_mode("public")
    guest_id = 99999  # Anyone who is not the owner.
    callbacks = _flat_callbacks(wizard._kb_main_after_claim(guest_id))
    # Feature buttons still visible:
    assert "main:helpzavr" in callbacks
    assert "main:pretty" in callbacks
    assert "main:mailbox" in callbacks
    assert "main:media_toggle" in callbacks
    assert "main:settings" in callbacks
    # Admin-only buttons hidden:
    assert "main:download" not in callbacks
    assert "main:role" not in callbacks
    assert "main:brain" not in callbacks
    assert "main:editor" not in callbacks
    assert "main:tokens" not in callbacks


def test_brain_keyboard_has_back_button(fresh_modules) -> None:
    """The «🧠 Перенастроить мозг» screen exposes a «← Назад» button
    so the user can leave it — earlier it had no exit."""
    _, wizard = fresh_modules
    callbacks = _flat_callbacks(wizard._kb_brain())
    assert "brain:back" in callbacks


def test_role_picker_text_mentions_active_persona(fresh_modules) -> None:
    storage, wizard = fresh_modules
    text = wizard._role_picker_text()
    assert "Lilush Boss" in text  # default persona display_name
    storage.set_persona_override("research_lead")
    text = wizard._role_picker_text()
    assert "Research Lead" in text


def test_role_confirm_keyboard_includes_use_button(fresh_modules) -> None:
    _, wizard = fresh_modules
    markup = wizard._kb_role_confirm("debate_lead")
    callbacks = _flat_callbacks(markup)
    assert "role:use:debate_lead" in callbacks
    assert "role:back" in callbacks


def test_persona_descriptions_with_html_chars_get_escaped(fresh_modules) -> None:
    """Two personas (pr_reviewer, watchdog) have literal '<' in their text.

    Telegram's HTML parser would reject the edit_text call if we forgot to
    escape, so any user picking one of these roles would see no response.
    This guards against the regression Devin Review caught in PR #12.
    """
    from bot.persona import _PERSONAS

    danger = [p for p in _PERSONAS.values() if "<" in p.description or ">" in p.description]
    # Sanity: we know these two have '<' — if someone rewrites the
    # description to drop the special chars, this list shrinks; the
    # important thing is the escaping codepath is exercised.
    assert {p.key for p in danger}.issuperset({"pr_reviewer", "watchdog"})

    # The escape helper turns '<' into '&lt;'.
    _, wizard = fresh_modules
    assert wizard._html_escape("Если < 9 — fail") == "Если &lt; 9 — fail"
