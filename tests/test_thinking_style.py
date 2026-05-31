"""Tests for the thinking-style addon.

Covers:

* default style is ``white``
* setting / reading a style round-trips via lilush's addon-state
* unknown values fall back to the default
* :func:`make_status_runner` returns coroutines of the right shape and
  decorates lines with the active model in ``white_model`` mode
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from bot.addons.thinking_style import handlers as th


@pytest.fixture(autouse=True)
def _isolate_addon_state(tmp_path):
    """Force lilush storage into a temp file so tests don't share state."""
    from bot.storage import storage as _s

    saved_file = _s.state_file
    saved_state = _s._state
    _s.state_file = tmp_path / "state.json"
    _s._state = {}
    try:
        yield
    finally:
        _s.state_file = saved_file
        _s._state = saved_state


def test_default_style_is_white():
    assert th.get_style(123) == th.STYLE_WHITE
    assert th.is_typing_only(123) is False
    assert th.show_model_suffix(123) is False


def test_set_and_read_style():
    th._set_style(99, th.STYLE_WHITE_MODEL)
    assert th.get_style(99) == th.STYLE_WHITE_MODEL
    assert th.show_model_suffix(99) is True

    th._set_style(99, th.STYLE_TYPING)
    assert th.get_style(99) == th.STYLE_TYPING
    assert th.is_typing_only(99) is True


def test_unknown_value_falls_back_to_default():
    th._set_style(7, "nonsense")
    assert th.get_style(7) == th.STYLE_WHITE


def _fake_message(chat_id: int = 42) -> MagicMock:
    msg = MagicMock()
    msg.chat.id = chat_id
    sent = MagicMock()
    sent.edit_text = AsyncMock()
    sent.delete = AsyncMock()
    msg.answer = AsyncMock(return_value=sent)
    msg.bot = MagicMock()
    msg.bot.send_chat_action = AsyncMock()
    return msg


def test_status_runner_white_sends_then_edits():
    async def run():
        msg = _fake_message()
        update, finish = th.make_status_runner(msg, model_hint=lambda: "m/x")
        await update("Думаю…")
        # First send happened
        assert msg.answer.await_count == 1
        first_call = msg.answer.await_args
        assert first_call.args[0] == "Думаю…"
        # Force timing so the second edit goes through
        await asyncio.sleep(th._MIN_EDIT_INTERVAL + 0.05)
        await update("Готово")
        sent = msg.answer.return_value
        sent.edit_text.assert_awaited_with("Готово")
        await finish()
        sent.delete.assert_awaited()

    asyncio.run(run())


def test_status_runner_white_model_appends_model():
    async def run():
        msg = _fake_message(chat_id=55)
        th._set_style(55, th.STYLE_WHITE_MODEL)
        update, _ = th.make_status_runner(
            msg, model_hint=lambda: "openrouter/free"
        )
        await update("Думаю…")
        first_call = msg.answer.await_args
        sent_text = first_call.args[0]
        assert "Думаю…" in sent_text
        assert "openrouter/free" in sent_text

    asyncio.run(run())


def test_status_runner_typing_only_fires_native_indicator_no_bubble():
    """The "Чёрная по центру" style must NOT send any chat bubble.
    Status is communicated only via Telegram's native chat-action
    indicator at the top of the chat (``send_chat_action``).
    """
    async def run():
        msg = _fake_message(chat_id=77)
        th._set_style(77, th.STYLE_TYPING)
        update, finish = th.make_status_runner(msg, model_hint=lambda: "m")
        await update("Думаю…")
        # Native chat-action fired at least once. NO bubble was sent.
        await asyncio.sleep(0.05)
        assert msg.bot.send_chat_action.await_count >= 1
        msg.answer.assert_not_called()
        await finish()

    asyncio.run(run())


def test_status_runner_typing_rotates_action_with_status_text():
    """When the status text mentions «рисую» the action should switch
    from ``typing`` to ``upload_photo``; «анимирую» → ``record_video``.
    """
    async def run():
        msg = _fake_message(chat_id=88)
        th._set_style(88, th.STYLE_TYPING)
        update, finish = th.make_status_runner(msg, model_hint=None)
        await update("Думаю…")
        await update("Рисую рамку…")
        await update("Анимирую ролик…")
        actions = [c.args[1] for c in msg.bot.send_chat_action.await_args_list]
        assert "typing" in actions
        assert "upload_photo" in actions
        assert "record_video" in actions
        await finish()

    asyncio.run(run())


def test_action_for_status_mapper():
    assert th._action_for_status("") == "typing"
    assert th._action_for_status("Думаю…") == "typing"
    assert th._action_for_status("Рисую стрелку") == "upload_photo"
    assert th._action_for_status("аннотирую скриншот") == "upload_photo"
    assert th._action_for_status("Анимирую видео") == "record_video"
    assert th._action_for_status("скачиваю файл") == "upload_document"
