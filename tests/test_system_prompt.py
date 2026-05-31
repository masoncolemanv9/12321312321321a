"""Tests for the persona-aware system prompt in bot.agent.

The user wants two behaviours:
- Before /role is selected (default boss + no override): free discovery chat.
- After /role is selected: bot stays in role, redirects off-topic asks.
"""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture()
def fresh_modules(tmp_path, monkeypatch):
    """Fresh storage + agent in a tmp dir, default BOT_PERSONA=boss."""
    monkeypatch.setenv("BOT_PERSONA", "boss")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    import sys

    for mod in ("bot.config", "bot.storage", "bot.persona", "bot.agent"):
        sys.modules.pop(mod, None)
    config = importlib.import_module("bot.config")
    storage_mod = importlib.import_module("bot.storage")
    persona_mod = importlib.import_module("bot.persona")
    agent_mod = importlib.import_module("bot.agent")
    storage_mod.storage = storage_mod.Storage(data_dir=config.DATA_DIR)
    # Re-bind the agent's storage reference to the fresh singleton.
    agent_mod.storage = storage_mod.storage
    return storage_mod.storage, persona_mod, agent_mod


def test_default_boss_uses_discovery_framing(fresh_modules) -> None:
    """No override on the boss bot → discovery framing (free chat)."""
    _storage, _persona, agent = fresh_modules
    prompt = agent._build_system_prompt()
    assert "DISCOVERY" in prompt
    assert "Chat freely" in prompt
    # No role-specific brief in discovery mode.
    assert "STAY IN ROLE" not in prompt


def test_role_override_switches_to_role_framing(fresh_modules) -> None:
    """After /role picks a non-default persona → role-constrained framing."""
    storage, _persona, agent = fresh_modules
    storage.set_persona_override("d1_skeptic")
    prompt = agent._build_system_prompt()
    assert "STAY IN ROLE" in prompt
    assert "D1 Skeptic" in prompt
    assert "DISCOVERY" not in prompt


def test_non_boss_env_uses_role_framing_without_override(
    fresh_modules, monkeypatch
) -> None:
    """A farm bot deployed as e.g. coder_lead never had discovery mode."""
    _storage, _persona, agent = fresh_modules
    monkeypatch.setenv("BOT_PERSONA", "coder_lead")
    prompt = agent._build_system_prompt()
    assert "STAY IN ROLE" in prompt
    assert "Coder Lead" in prompt
    assert "DISCOVERY" not in prompt


def test_clearing_override_returns_to_discovery_on_boss(fresh_modules) -> None:
    storage, _persona, agent = fresh_modules
    storage.set_persona_override("watchdog")
    assert "STAY IN ROLE" in agent._build_system_prompt()
    storage.clear_persona_override()
    assert "DISCOVERY" in agent._build_system_prompt()


def test_unknown_override_falls_back_safely(fresh_modules) -> None:
    """Unknown override → don't crash; effective persona collapses to boss."""
    storage, persona_mod, agent = fresh_modules
    storage.set_persona_override("totally-made-up-role")
    # get_persona() falls back to boss for unknown keys.
    assert persona_mod.get_persona().key == "boss"
    # Prompt must still be a non-empty string and contain general rules.
    prompt = agent._build_system_prompt()
    assert prompt.strip() != ""
    assert "Telegram bot" in prompt
