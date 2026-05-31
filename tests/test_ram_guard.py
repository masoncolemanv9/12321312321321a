"""Tests for the «💻 RAM» settings + agent-loop RAM guard.

The agent loop (in :mod:`bot.agent`) reads three settings on every
turn:
* ``storage.get_ram_limit_mb()`` — the soft cap in MB; 0 = disabled.
* ``storage.get_ram_behavior()`` — ``"compress"`` (shrink tool results
  at 80%, refuse at 95%) or ``"refuse"`` (just refuse at the limit).
* ``storage.get_ram_show()`` — whether to suffix status lines with the
  current RSS.

These tests cover the storage primitives plus the pure helpers
``_check_ram_state`` and ``_compress_tool_results``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import bot.agent as agent
from bot.storage import Storage


def test_ram_settings_defaults(tmp_path: Path) -> None:
    s = Storage(data_dir=tmp_path)
    assert s.get_ram_limit_mb() == 0
    assert s.get_ram_behavior() == "compress"
    assert s.get_ram_show() is False


def test_ram_limit_clamps_negative(tmp_path: Path) -> None:
    s = Storage(data_dir=tmp_path)
    s.set_ram_limit_mb(-50)
    assert s.get_ram_limit_mb() == 0
    s.set_ram_limit_mb(400)
    assert s.get_ram_limit_mb() == 400


def test_ram_behavior_rejects_unknown(tmp_path: Path) -> None:
    s = Storage(data_dir=tmp_path)
    with pytest.raises(ValueError):
        s.set_ram_behavior("nuke")
    s.set_ram_behavior("refuse")
    assert s.get_ram_behavior() == "refuse"
    s.set_ram_behavior("compress")
    assert s.get_ram_behavior() == "compress"


def test_ram_show_is_boolish(tmp_path: Path) -> None:
    s = Storage(data_dir=tmp_path)
    s.set_ram_show(1)
    assert s.get_ram_show() is True
    s.set_ram_show(0)
    assert s.get_ram_show() is False


# ─── _check_ram_state pure logic ──────────────────────────────────────


def test_check_ram_state_disabled_when_limit_zero(monkeypatch) -> None:
    monkeypatch.setattr(agent, "_current_rss_mb", lambda: 800)
    assert agent._check_ram_state(0, "compress") == (0, "ok")


def test_check_ram_state_compress_threshold(monkeypatch) -> None:
    """80%-94% in compress mode → compress."""
    monkeypatch.setattr(agent, "_current_rss_mb", lambda: 400)
    rss, action = agent._check_ram_state(500, "compress")
    assert action == "compress"
    assert rss == 400


def test_check_ram_state_compress_refuses_at_95(monkeypatch) -> None:
    monkeypatch.setattr(agent, "_current_rss_mb", lambda: 480)
    rss, action = agent._check_ram_state(500, "compress")
    assert action == "refuse"
    assert rss == 480


def test_check_ram_state_compress_ok_below_80(monkeypatch) -> None:
    monkeypatch.setattr(agent, "_current_rss_mb", lambda: 300)
    rss, action = agent._check_ram_state(500, "compress")
    assert action == "ok"
    assert rss == 300


def test_check_ram_state_refuse_mode_does_not_compress(monkeypatch) -> None:
    """In refuse mode, only the actual limit (100%) triggers refuse."""
    monkeypatch.setattr(agent, "_current_rss_mb", lambda: 400)
    rss, action = agent._check_ram_state(500, "refuse")
    assert action == "ok"

    monkeypatch.setattr(agent, "_current_rss_mb", lambda: 510)
    rss, action = agent._check_ram_state(500, "refuse")
    assert action == "refuse"
    assert rss == 510


def test_check_ram_state_treats_zero_rss_as_unknown(monkeypatch) -> None:
    """When RSS can't be measured, never trigger guard actions."""
    monkeypatch.setattr(agent, "_current_rss_mb", lambda: 0)
    assert agent._check_ram_state(500, "compress") == (0, "ok")
    assert agent._check_ram_state(500, "refuse") == (0, "ok")


# ─── _compress_tool_results ───────────────────────────────────────────


def test_compress_tool_results_trims_big_payloads() -> None:
    big = "x" * 5000
    messages = [
        {"role": "user", "content": "go"},
        {"role": "tool", "name": "list_dir", "content": big},
        {"role": "tool", "name": "read_file", "content": "tiny"},
        {"role": "assistant", "content": "ok"},
    ]
    removed = agent._compress_tool_results(messages)
    assert removed > 0
    # Big tool result shrunk.
    assert len(messages[1]["content"]) < len(big)
    assert messages[1]["content"].endswith("[trimmed by RAM guard]")
    # Tiny one preserved.
    assert messages[2]["content"] == "tiny"
    # Non-tool roles untouched.
    assert messages[0]["content"] == "go"
    assert messages[3]["content"] == "ok"


def test_compress_tool_results_idempotent_on_small() -> None:
    messages = [
        {"role": "tool", "name": "list_dir", "content": "short"},
    ]
    removed = agent._compress_tool_results(messages)
    assert removed == 0
    assert messages[0]["content"] == "short"


# ─── current_rss_mb fallbacks ─────────────────────────────────────────


def test_current_rss_mb_returns_positive_integer() -> None:
    """On any platform that supports psutil OR /proc/self/status this
    should return a positive number. The test process certainly uses
    >0 MB of RSS."""
    from bot.addons.ram_guard.handlers import current_rss_mb

    assert current_rss_mb() > 0
