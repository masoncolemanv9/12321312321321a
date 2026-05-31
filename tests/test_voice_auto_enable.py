"""Regression coverage for the «голос/JSON выбираю — а бот молчит» bug.

Before this fix, picking a built-in voice or importing a Voice-Builder
JSON only stored the value but did NOT flip ``set_tts_enabled(True)``
or activate the imported clone. Users (correctly) read that as "the
voice picker is broken". These tests pin the new behaviour so it
doesn't silently regress again.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture()
def fresh_modules(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BOT_PERSONA", "boss")
    monkeypatch.setenv("BOT_OWNER_USER_ID", "12345")
    import sys

    for mod in ("bot.config", "bot.storage", "bot.persona", "bot.wizard", "bot.tts"):
        sys.modules.pop(mod, None)
    config = importlib.import_module("bot.config")
    storage_mod = importlib.import_module("bot.storage")
    storage_mod.storage = storage_mod.Storage(data_dir=config.DATA_DIR)
    wizard = importlib.import_module("bot.wizard")
    wizard.storage = storage_mod.storage
    storage_mod.storage.set_owner_id(12345)
    return storage_mod.storage, wizard


def _make_query(data: str, user_id: int = 12345):
    q = MagicMock()
    q.data = data
    q.from_user = MagicMock()
    q.from_user.id = user_id
    q.answer = AsyncMock()
    q.message = MagicMock()
    q.message.edit_text = AsyncMock()
    q.message.answer = AsyncMock()
    return q


def _make_state():
    state = MagicMock()
    state.clear = AsyncMock()
    state.set_state = AsyncMock()
    state.update_data = AsyncMock()
    state.get_data = AsyncMock(return_value={})
    return state


def test_voice_set_auto_enables_tts(fresh_modules):
    storage, wizard = fresh_modules
    assert storage.get_tts_enabled() is False

    query = _make_query("tts:voice:set:F3")
    asyncio = importlib.import_module("asyncio")
    asyncio.run(wizard.cb_tts_voice_set(query, _make_state()))

    assert storage.get_tts_voice() == "F3"
    assert storage.get_tts_enabled() is True, (
        "Picking a built-in voice must auto-enable narration so the "
        "user actually hears their pick."
    )
    # The «answer» toast should mention the auto-enable so the user
    # isn't confused that the bot suddenly starts talking.
    query.answer.assert_awaited()
    notice = query.answer.await_args.args[0]
    assert "ВКЛ" in notice


def test_voice_set_clears_custom_clone(fresh_modules):
    storage, wizard = fresh_modules
    storage.set_tts_custom_voice_path("/tmp/leftover.json")

    query = _make_query("tts:voice:set:M2")
    asyncio = importlib.import_module("asyncio")
    asyncio.run(wizard.cb_tts_voice_set(query, _make_state()))

    assert storage.get_tts_voice() == "M2"
    # Picking a built-in must wipe any leftover custom voice so the
    # next synth call doesn't ignore the user's new pick.
    assert storage.get_tts_custom_voice_path() == ""


def test_voice_json_import_activates_and_enables(fresh_modules, tmp_path):
    """End-to-end: feeding a JSON document through capture_voice_json
    must save it, set it as the active custom voice, AND enable TTS.
    """
    storage, wizard = fresh_modules

    payload = {"styles": [[0.1, 0.2, 0.3]]}

    msg = MagicMock()
    msg.from_user = MagicMock()
    msg.from_user.id = 12345
    msg.answer = AsyncMock()
    msg.document = MagicMock()
    msg.document.file_name = "myvoice.json"
    msg.document.file_id = "f-1"

    bot = MagicMock()
    bot.get_file = AsyncMock(return_value=MagicMock(file_path="x.json"))

    async def _fake_download(file_id, destination):
        Path(destination).write_text(json.dumps(payload))

    bot.download = AsyncMock(side_effect=_fake_download)
    msg.bot = bot

    state = _make_state()
    asyncio = importlib.import_module("asyncio")
    asyncio.run(wizard.capture_voice_json(msg, state))

    saved_path = storage.get_tts_custom_voice_path()
    assert saved_path, "import must save the JSON as the active voice"
    assert Path(saved_path).exists()
    assert json.loads(Path(saved_path).read_text()) == payload
    assert storage.get_tts_enabled() is True
    state.clear.assert_awaited()
    msg.answer.assert_awaited()
    confirm = msg.answer.await_args.args[0]
    assert "сохран" in confirm.lower()
    assert "вкл" in confirm.lower()


def test_rec_save_auto_enables_tts(fresh_modules, tmp_path, monkeypatch):
    """Hitting «💾 Сохранить как активный» also flips the narrator ON."""
    storage, wizard = fresh_modules

    # Fake an existing recording JSON on disk where cb_tts_rec_save
    # expects it. The helper computes the path from the user id.
    json_path = wizard._user_voice_json_path(12345)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps({"styles": [[1, 2, 3]]}))

    query = _make_query("tts:rec:save")
    state = _make_state()
    asyncio = importlib.import_module("asyncio")
    asyncio.run(wizard.cb_tts_rec_save(query, state))

    assert storage.get_tts_custom_voice_path() == str(json_path)
    assert storage.get_tts_enabled() is True
