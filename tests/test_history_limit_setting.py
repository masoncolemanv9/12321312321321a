"""Tests for the user-configurable history limit.

The «📝 Кол-во сообщений» button in /setup → 💾 Память бота persists a
value into ``storage._settings()['history_limit']`` and
``Storage.append_history`` honours it on the very next message
(without restarting the bot).
"""

from __future__ import annotations

from pathlib import Path

from bot.config import HISTORY_LIMIT
from bot.storage import Storage


def test_history_limit_defaults_to_config(tmp_path: Path) -> None:
    s = Storage(data_dir=tmp_path)
    assert s.get_history_limit() == HISTORY_LIMIT


def test_set_history_limit_persists_and_caps_append(tmp_path: Path) -> None:
    s = Storage(data_dir=tmp_path)
    s.set_history_limit(3)
    assert s.get_history_limit() == 3
    user_id = 1
    for i in range(20):
        s.append_history(user_id, {"role": "user", "content": f"msg {i}"})
    history = s.get_history(user_id)
    # cap is 3 pairs = 6 messages.
    assert len(history) == 6
    # Newest entries kept.
    assert history[-1]["content"] == "msg 19"


def test_history_limit_validates_range(tmp_path: Path) -> None:
    s = Storage(data_dir=tmp_path)
    s.set_history_limit(0)  # clamps to >=1
    assert s.get_history_limit() == 1
    s.set_history_limit(10_000)  # clamps to <=500
    assert s.get_history_limit() == 500


def test_history_limit_survives_reload(tmp_path: Path) -> None:
    s = Storage(data_dir=tmp_path)
    s.set_history_limit(77)
    # Simulate restart.
    s2 = Storage(data_dir=tmp_path)
    assert s2.get_history_limit() == 77
