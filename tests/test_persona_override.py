"""Tests for the TG-driven persona override (via /role)."""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture()
def fresh_storage(tmp_path, monkeypatch):
    """Clean Storage in a tmp dir, with BOT_PERSONA env defaulted to boss."""
    monkeypatch.setenv("BOT_PERSONA", "boss")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    import sys

    for mod in ("bot.config", "bot.storage", "bot.persona"):
        sys.modules.pop(mod, None)
    config = importlib.import_module("bot.config")
    storage_mod = importlib.import_module("bot.storage")
    persona_mod = importlib.import_module("bot.persona")
    # Replace the module-level singleton so persona.get_persona() reads our state.
    storage_mod.storage = storage_mod.Storage(data_dir=config.DATA_DIR)
    return storage_mod.storage, persona_mod


def test_no_override_falls_back_to_env(fresh_storage, monkeypatch) -> None:
    storage, persona = fresh_storage
    monkeypatch.setenv("BOT_PERSONA", "research_lead")
    # Re-evaluating persona reads the env each call.
    assert storage.get_persona_override() is None
    assert persona.get_persona().key == "research_lead"


def test_override_wins_over_env(fresh_storage, monkeypatch) -> None:
    storage, persona = fresh_storage
    monkeypatch.setenv("BOT_PERSONA", "research_lead")
    storage.set_persona_override("debate_lead")
    p = persona.get_persona()
    assert p.key == "debate_lead"
    assert p.department == "debate"


def test_clear_override_returns_to_env(fresh_storage, monkeypatch) -> None:
    storage, persona = fresh_storage
    monkeypatch.setenv("BOT_PERSONA", "coder_lead")
    storage.set_persona_override("designer-typo")  # unknown → boss
    # Unknown key gracefully degrades; override is still recorded so the
    # owner can see it in /role and reset it.
    assert storage.get_persona_override() == "designer-typo"
    assert persona.get_persona().key == "boss"
    storage.clear_persona_override()
    assert storage.get_persona_override() is None
    assert persona.get_persona().key == "coder_lead"


def test_override_persists_across_storage_reload(fresh_storage, tmp_path) -> None:
    storage, persona = fresh_storage
    storage.set_persona_override("github_scout")
    # Simulate a process restart: build a fresh Storage from disk.
    from bot.storage import Storage

    storage2 = Storage(data_dir=tmp_path)
    assert storage2.get_persona_override() == "github_scout"
