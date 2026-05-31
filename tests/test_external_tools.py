"""Tests for the external-tools key plumbing (storage + wizard helpers).

The wizard FSM itself is exercised via storage-level invariants here; the
end-to-end aiogram round-trip is tested separately if/when the bot grows
a fuller test harness. These tests focus on the contract that Researcher
bots will rely on: ``storage.get_external_tool_key("apify")`` returns the
value set via Telegram, falling back to the env var.
"""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture()
def fresh_storage(tmp_path, monkeypatch):
    """Spin up a Storage backed by a clean tmp dir and drop env interference.

    Reloading ``bot.storage`` is necessary so the new ``DATA_DIR`` env
    var takes effect, but the reload creates a *new* module object and
    a *new* :class:`Storage` singleton. Other modules already imported
    in this pytest session (``bot.addons.state``, the helpzavr
    handlers, …) still hold a reference to the *original* singleton.
    Leaving the swap in place poisons every subsequent test that
    expects ``storage`` and ``bot.addons.state.storage`` to be the same
    instance. We stash the originals here and restore them on teardown
    so the rest of the session sees a coherent module graph.
    """
    import sys

    for env in (
        "APIFY_API_TOKEN",
        "FIRECRAWL_API_KEY",
        "TAVILY_API_KEY",
        "BRAVE_SEARCH_API_KEY",
        "EXA_API_KEY",
        "GITHUB_RESEARCH_PAT",
    ):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    saved_modules = {
        name: sys.modules.get(name) for name in ("bot.config", "bot.storage")
    }
    for name in saved_modules:
        sys.modules.pop(name, None)
    try:
        config = importlib.import_module("bot.config")
        storage_mod = importlib.import_module("bot.storage")
        yield storage_mod.Storage(data_dir=config.DATA_DIR)
    finally:
        for name, original in saved_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def test_known_tools_roster() -> None:
    """The roster must contain the tools the user explicitly named."""
    from bot.storage import KNOWN_EXTERNAL_TOOLS

    for tool in ("apify", "firecrawl", "tavily", "brave", "exa", "github_pat"):
        assert tool in KNOWN_EXTERNAL_TOOLS, f"missing tool: {tool}"
        meta = KNOWN_EXTERNAL_TOOLS[tool]
        assert meta["env_var"]
        assert meta["label"]
        assert meta["url"].startswith("http")
        assert meta["hint"]


def test_set_and_get_external_tool_key(fresh_storage) -> None:
    """Round-trip: set → get returns same value, persisted to disk."""
    fresh_storage.set_external_tool_key("apify", "apify_api_xyz123")
    assert fresh_storage.get_external_tool_key("apify") == "apify_api_xyz123"

    # Persisted: a fresh Storage instance on the same dir sees it.
    from bot.storage import Storage

    again = Storage(data_dir=fresh_storage.data_dir)
    assert again.get_external_tool_key("apify") == "apify_api_xyz123"


def test_get_external_tool_key_unknown_returns_empty(fresh_storage) -> None:
    """Querying a non-roster tool returns '' rather than raising."""
    assert fresh_storage.get_external_tool_key("never_heard_of_it") == ""


def test_set_external_tool_key_rejects_unknown(fresh_storage) -> None:
    """Setting an out-of-roster tool name must raise ValueError."""
    with pytest.raises(ValueError, match="unknown external tool"):
        fresh_storage.set_external_tool_key("never_heard_of_it", "abc")


def test_env_var_fallback(fresh_storage, monkeypatch) -> None:
    """If nothing is set via Telegram, env var wins."""
    monkeypatch.setenv("APIFY_API_TOKEN", "from-env-1234")
    assert fresh_storage.get_external_tool_key("apify") == "from-env-1234"

    # Telegram-set takes precedence over env.
    fresh_storage.set_external_tool_key("apify", "from-telegram")
    assert fresh_storage.get_external_tool_key("apify") == "from-telegram"


def test_delete_external_tool_key(fresh_storage) -> None:
    """Deleting an unset key returns False; set+delete returns True."""
    assert fresh_storage.delete_external_tool_key("apify") is False
    fresh_storage.set_external_tool_key("apify", "x")
    assert fresh_storage.delete_external_tool_key("apify") is True
    assert fresh_storage.get_external_tool_key("apify") == ""


def test_list_external_tool_keys_status(fresh_storage, monkeypatch) -> None:
    """list_external_tool_keys returns source/masked metadata per tool."""
    monkeypatch.setenv("FIRECRAWL_API_KEY", "envvalue4567890")
    fresh_storage.set_external_tool_key("apify", "tgvalue1234567890")
    listing = fresh_storage.list_external_tool_keys()
    # Telegram-set
    assert listing["apify"]["source"] == "telegram"
    assert listing["apify"]["masked"].startswith("tgvalu")
    # Env-set
    assert listing["firecrawl"]["source"] == "env"
    assert listing["firecrawl"]["masked"].startswith("envval")
    # Unset
    assert listing["tavily"]["source"] == "none"
    assert listing["tavily"]["masked"] == ""
    # Metadata propagated through
    assert listing["apify"]["label"] == "Apify"
    assert listing["apify"]["env_var"] == "APIFY_API_TOKEN"
