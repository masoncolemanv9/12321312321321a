"""Tests for the brain-slot chain.

Slot routing has fixed roles:
* Slot 1 (or «Другое») is the chat brain.
* Slot 2 is reserved for voice/photo Groq overrides and is NEVER used
  as a chat brain — including no silent failover when slot 1 fails.

``_slot_chain`` therefore always returns ``["1"]`` regardless of which
``active_brain_slot`` value is persisted (the storage field is kept
for legacy state.json compatibility but is no longer consulted on the
chat path).
"""

from __future__ import annotations

from pathlib import Path

import bot.agent as agent_mod
from bot.storage import Storage


def test_slot_chain_always_returns_slot_one_when_both_configured(
    tmp_path: Path, monkeypatch
) -> None:
    fresh = Storage(data_dir=tmp_path)
    fresh.set_provider("openrouter")
    fresh.set_provider_key("openrouter", "sk-or-test-aaaaaaaaaa")
    fresh.set_brain_slot2_field("api_key", "gsk-test-bbbbbbbbbb")
    fresh.set_brain_slot2_field("base_url", "https://api.groq.com/openai/v1")
    fresh.set_active_brain_slot("1")
    monkeypatch.setattr(agent_mod, "storage", fresh)

    assert agent_mod._slot_chain() == ["1"]

    # Legacy state.json may have persisted active_brain_slot=2 from before
    # the chat-routing rule was locked to slot 1. Chat must still route
    # through slot 1 — slot 2 stays voice/photo-only.
    fresh.set_active_brain_slot("2")
    assert agent_mod._slot_chain() == ["1"]


def test_slot_chain_returns_slot_one_when_other_empty(
    tmp_path: Path, monkeypatch
) -> None:
    fresh = Storage(data_dir=tmp_path)
    fresh.set_provider("openrouter")
    fresh.set_provider_key("openrouter", "sk-or-only-aaaaaaaaaa")
    fresh.set_active_brain_slot("1")
    monkeypatch.setattr(agent_mod, "storage", fresh)

    assert agent_mod._slot_chain() == ["1"]
