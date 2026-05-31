"""Smoke tests for the photo_chooser addon.

We don't spin up a real Telegram client — instead we ask aiogram for the
registered observers and confirm:

* exactly one ``F.photo`` message handler exists in this router
* exactly one ``F.document.mime_type.startswith("image/")`` matcher
* the close / helpzavr / generation callback queries are all wired
* the chooser respects the master ``media_toggle`` kill-switch
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from bot.addons.photo_chooser.handlers import build_photo_chooser_router


def test_router_has_expected_handlers():
    router = build_photo_chooser_router()
    # message observer holds both F.photo and F.document.mime_type matchers.
    msg_handlers = list(router.message.handlers)
    assert len(msg_handlers) == 2, [h.callback.__name__ for h in msg_handlers]
    names = {h.callback.__name__ for h in msg_handlers}
    assert {"on_photo", "on_document_image"} == names

    cb_handlers = list(router.callback_query.handlers)
    assert len(cb_handlers) == 3
    names = {h.callback.__name__ for h in cb_handlers}
    assert {"on_close", "on_helpzavr", "on_generation"} == names


def _fake_photo_size(file_id: str, w: int, h: int) -> MagicMock:
    ps = MagicMock()
    ps.file_id = file_id
    ps.width = w
    ps.height = h
    return ps


def _fake_photo_message(owner_id: int, *, chat_id: int = 1) -> MagicMock:
    msg = MagicMock()
    msg.from_user.id = owner_id
    msg.chat.id = chat_id
    msg.photo = [
        _fake_photo_size("small", 90, 90),
        _fake_photo_size("medium", 320, 240),
        _fake_photo_size("large", 1280, 960),
    ]
    msg.caption = "что это?"
    msg.answer = AsyncMock()
    return msg


@pytest.fixture(autouse=True)
def _set_owner_and_isolate(tmp_path):
    from bot.storage import storage as _s

    saved_file, saved_state = _s.state_file, _s._state
    _s.state_file = tmp_path / "state.json"
    _s._state = {}
    _s.set_owner_id(42)
    try:
        yield
    finally:
        _s.state_file = saved_file
        _s._state = saved_state


def _get_handler(router, name):
    for h in router.message.handlers:
        if h.callback.__name__ == name:
            return h.callback
    raise AssertionError(name)


def test_on_photo_picks_largest_and_sets_state():
    import asyncio

    router = build_photo_chooser_router()
    on_photo = _get_handler(router, "on_photo")
    msg = _fake_photo_message(owner_id=42)
    state = MagicMock()
    state.clear = AsyncMock()
    state.update_data = AsyncMock()
    state.set_state = AsyncMock()
    bot = MagicMock()

    asyncio.run(on_photo(msg, state, bot))

    msg.answer.assert_awaited()
    call = state.update_data.await_args
    assert call.kwargs["chooser_file_id"] == "large"
    assert call.kwargs["chooser_caption"] == "что это?"
    assert call.kwargs["chooser_is_document"] is False


def test_on_photo_silent_when_non_owner():
    import asyncio

    router = build_photo_chooser_router()
    on_photo = _get_handler(router, "on_photo")
    msg = _fake_photo_message(owner_id=999)  # not the owner
    state = MagicMock()
    state.clear = AsyncMock()
    bot = MagicMock()

    asyncio.run(on_photo(msg, state, bot))

    msg.answer.assert_not_called()
    state.clear.assert_not_called()


def test_on_photo_silent_when_media_kill_switch_off():
    import asyncio

    from bot.addons.media_toggle.handlers import _set_media_enabled

    router = build_photo_chooser_router()
    on_photo = _get_handler(router, "on_photo")
    msg = _fake_photo_message(owner_id=42, chat_id=7)
    _set_media_enabled(7, False)
    state = MagicMock()
    state.clear = AsyncMock()
    bot = MagicMock()

    asyncio.run(on_photo(msg, state, bot))

    msg.answer.assert_not_called()


def test_on_photo_skips_chooser_when_helpzavr_enabled():
    """When the user already turned Helpzavr ON via its own settings
    screen, a fresh photo must bypass the chooser bubble entirely and
    head straight into the Helpzavr flow — either running the pipeline
    (caption present) or asking for the prompt (caption empty). The
    chooser must NOT show the чузер text.
    """
    import asyncio

    from bot.addons.helpzavr.handlers import _set_enabled

    _set_enabled(33, True)  # mode is on

    router = build_photo_chooser_router()
    on_photo = _get_handler(router, "on_photo")
    msg = _fake_photo_message(owner_id=42, chat_id=33)
    msg.caption = ""  # no caption → prompt-after-photo flow
    state = MagicMock()
    state.clear = AsyncMock()
    state.update_data = AsyncMock()
    state.set_state = AsyncMock()
    bot = MagicMock()

    # Helpzavr keys aren't set in this isolated state → the route
    # short-circuits with a "Set keys first" message and never shows
    # the chooser. That's enough to verify the chooser was bypassed.
    asyncio.run(on_photo(msg, state, bot))

    # message.answer was called — but NOT with the chooser text.
    for call in msg.answer.await_args_list:
        sent = call.args[0] if call.args else ""
        assert "Что делаем с этим фото" not in sent, (
            "chooser must not appear when Helpzavr mode is on"
        )
