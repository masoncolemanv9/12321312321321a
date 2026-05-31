"""Tests for bot.token_tracker — LLM token-spend accounting."""

from __future__ import annotations

import importlib
import time

import pytest


@pytest.fixture()
def fresh_storage(tmp_path, monkeypatch):
    """Clean Storage in a tmp dir; reset module state so the singleton uses it."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    import sys

    for mod in ("bot.config", "bot.storage", "bot.token_tracker"):
        sys.modules.pop(mod, None)
    config = importlib.import_module("bot.config")
    storage_mod = importlib.import_module("bot.storage")
    storage_mod.storage = storage_mod.Storage(data_dir=config.DATA_DIR)
    tracker = importlib.import_module("bot.token_tracker")
    return storage_mod.storage, tracker


def test_record_persists_entry(fresh_storage) -> None:
    storage, tracker = fresh_storage
    tracker.record("openrouter", "anthropic/claude-opus-4.7", 100, 50, purpose="chat")
    log = storage.get_token_log()
    assert len(log) == 1
    entry = log[0]
    assert entry["provider"] == "openrouter"
    assert entry["model"] == "anthropic/claude-opus-4.7"
    assert entry["prompt_tokens"] == 100
    assert entry["completion_tokens"] == 50
    assert entry["purpose"] == "chat"
    assert "ts" in entry


def test_pricing_known_and_unknown(fresh_storage) -> None:
    _, tracker = fresh_storage
    assert tracker.pricing_for("anthropic/claude-opus-4.7") == (15.0, 75.0)
    assert tracker.pricing_for("nvidia/nemotron-3-super-120b-a12b:free") == (0.0, 0.0)
    # Any unknown :free suffix → zero.
    assert tracker.pricing_for("unknown/whatever:free") == (0.0, 0.0)
    # Truly unknown → zero (conservative).
    assert tracker.pricing_for("some/exotic-model") == (0.0, 0.0)


def test_estimate_cost(fresh_storage) -> None:
    _, tracker = fresh_storage
    # 1M prompt + 1M completion @ (15, 75) → $90
    assert tracker.estimate_cost_usd("anthropic/claude-opus-4.7", 1_000_000, 1_000_000) == pytest.approx(90.0)
    # Free model → 0
    assert tracker.estimate_cost_usd("nvidia/nemotron-3-super-120b-a12b:free", 1_000_000, 1_000_000) == 0.0


def test_format_stats_includes_sections(fresh_storage) -> None:
    storage, tracker = fresh_storage
    tracker.record("openrouter", "anthropic/claude-opus-4.7", 200, 100)
    tracker.record("openrouter", "moonshotai/kimi-k2", 50, 20, purpose="heartbeat")
    text = tracker.format_token_stats()
    assert "Сегодня" in text
    assert "За неделю" in text
    assert "Всего" in text
    assert "anthropic/claude-opus-4.7" in text
    assert "heartbeat" in text  # by_purpose section


def test_old_entries_excluded_from_today(fresh_storage, monkeypatch) -> None:
    storage, tracker = fresh_storage
    # Inject a 2-day-old entry directly into the log.
    log = storage._settings().setdefault("token_log", [])
    log.append(
        {
            "ts": int(time.time()) - 2 * 86_400,
            "provider": "openrouter",
            "model": "anthropic/claude-opus-4.7",
            "prompt_tokens": 1_000_000,
            "completion_tokens": 0,
            "purpose": "chat",
        }
    )
    storage._save()
    # And a recent entry.
    tracker.record("openrouter", "openai/gpt-4o", 10, 5)
    text = tracker.format_token_stats()
    # Today section should only count the recent call.
    today_line = next(line for line in text.split("\n") if "Сегодня" in line)
    assert "1 вызовов" in today_line
    # Weekly section catches both.
    week_line = next(line for line in text.split("\n") if "За неделю" in line)
    assert "2 вызовов" in week_line


def test_log_capped_at_5000(fresh_storage) -> None:
    """The log is FIFO-capped at 5000 entries to keep state.json bounded.

    Inject 5099 entries directly (bypassing _save() on each call would be
    ~5s of disk I/O) then trigger the cap with one real ``record`` call.
    """
    storage, tracker = fresh_storage
    log = storage._settings().setdefault("token_log", [])
    for _i in range(5099):
        log.append(
            {
                "ts": 0,
                "provider": "openrouter",
                "model": "openai/gpt-4o",
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "purpose": "chat",
            }
        )
    storage._save()
    # Two real records — first hits the cap (5100 → 5000), second stays at 5000.
    tracker.record("openrouter", "openai/gpt-4o", 1, 1)
    tracker.record("openrouter", "openai/gpt-4o", 1, 1)
    assert len(storage.get_token_log()) == 5000
