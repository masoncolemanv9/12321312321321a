"""Unit tests for bot.persona — the multi-bot farm roster."""

from __future__ import annotations

import pytest

from bot import persona


def test_default_is_boss(monkeypatch: pytest.MonkeyPatch) -> None:
    """No env var → defaults to boss (single-bot deploy backward-compat)."""
    monkeypatch.delenv("BOT_PERSONA", raising=False)
    p = persona.get_persona()
    assert p.key == "boss"
    assert p.rank == "boss"
    assert p.department == "leadership"


def test_explicit_key_resolves() -> None:
    p = persona.get_persona("research_lead")
    assert p.key == "research_lead"
    assert p.department == "research"
    assert p.rank == "lead"


def test_unknown_key_falls_back_to_boss() -> None:
    """Typo in BOT_PERSONA should not crash the bot on Render — fall back."""
    p = persona.get_persona("not_a_real_persona")
    assert p.key == "boss"


def test_roster_has_20_personas() -> None:
    assert len(persona.list_personas()) == 20
    assert len(persona.known_keys()) == 20


def test_roster_has_one_boss_and_four_leads() -> None:
    """The leadership layer is fixed: 1 boss + 4 department leads."""
    bosses = [p for p in persona.list_personas() if p.rank == "boss"]
    leads = [p for p in persona.list_personas() if p.rank == "lead"]
    assert len(bosses) == 1
    assert len(leads) == 4
    # Each department has exactly one lead.
    lead_depts = {p.department for p in leads}
    assert lead_depts == {"research", "debate", "coder", "devops"}


def test_keys_are_lowercase_snake_case() -> None:
    """render.yaml turns these into kebab-case service names — must be valid."""
    for p in persona.list_personas():
        assert p.key == p.key.lower()
        assert " " not in p.key
        assert "-" not in p.key


def test_first_five_match_render_yaml_blueprint() -> None:
    """The default 5-bot Blueprint is boss + 4 leads, in roster order."""
    expected = ["boss", "research_lead", "debate_lead", "coder_lead", "devops_lead"]
    actual = [p.key for p in persona.list_personas()[:5]]
    assert actual == expected
