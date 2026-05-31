"""Main-menu Groq overrides: voice → transcript, photo+caption → vision.

Both handlers fire only outside any FSM state (``StateFilter(None)``)
so the existing wizard / Helpzavr / media flows are untouched. They
also bail out silently when the user has the «Другое» brain
configured — see :func:`bot.groq_helpers.should_use_groq_override`.

Wiring is in ``bot/main.py``: the ``groq_router`` is registered
*before* the photo chooser router so the photo+caption case is
intercepted first; voice handling has no other competitor today.
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import tempfile
from typing import Any

from aiogram import Bot, F, Router
from aiogram.filters import StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.types import Message

from .agent import NoApiKeyError, run_agent
from .groq_helpers import (
    describe_image_with_groq,
    is_other_brain_active,
    should_use_groq_override,
    transcribe_audio_with_groq,
)
from .storage import storage

logger = logging.getLogger(__name__)
groq_router = Router(name="groq_overrides")


def _html_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _is_authorized(message: Message) -> bool:
    """Mirror of ``handlers._is_authorized`` without importing it (cycle)."""
    user = message.from_user
    if user is None:
        return False
    # Owner override always wins.
    owner = storage.get_owner_id()
    if owner is not None and user.id == owner:
        return True
    # Defer to the access module for co-owner / public access modes.
    from .access import can_use

    return can_use(user.id)


async def _download_to_tempfile(
    bot: Bot, file_id: str, suffix: str
) -> str:
    """Pull ``file_id`` from Telegram into a fresh temp file. Returns path."""
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="groq_")
    os.close(fd)
    try:
        await bot.download(file_id, destination=path)
    except Exception:
        with contextlib.suppress(Exception):
            os.unlink(path)
        raise
    return path


def _voice_or_audio_target(message: Message) -> tuple[str, str] | None:
    """Pick (file_id, suffix) from a voice / audio / .ogg document message.

    Returns ``None`` if the message isn't any of those — caller skips.
    """
    if message.voice is not None:
        return message.voice.file_id, ".ogg"
    if message.audio is not None:
        # Audio attachment — keep the file's extension if we can guess it.
        mime = (message.audio.mime_type or "").lower()
        suffix = ".mp3" if "mpeg" in mime else ".m4a" if "mp4" in mime else ".audio"
        return message.audio.file_id, suffix
    if message.document is not None:
        mime = (message.document.mime_type or "").lower()
        if mime.startswith("audio/") or mime == "video/ogg":
            return message.document.file_id, ".ogg"
    return None


# ---- voice → Groq Whisper → run agent on transcript -----------------------


@groq_router.message(StateFilter(None), F.voice)
@groq_router.message(StateFilter(None), F.audio)
async def on_voice_or_audio(
    message: Message, state: FSMContext, bot: Bot
) -> None:
    """Transcribe voice/audio with Groq and feed the result to the agent.

    Only fires in the main menu (no FSM state) so /work and addon FSMs
    are not interrupted. Gracefully bows out when «Другое» is active or
    no Groq slot is configured.
    """
    if not _is_authorized(message):
        return
    if not should_use_groq_override():
        if is_other_brain_active():
            await message.answer(
                "🎙 Голосовые на Groq не подключал — активна Другое (имеет приоритет). "
                "Пришли текстом."
            )
        else:
            await message.answer(
                "🎙 Голосовые работают через Groq. Зайди в "
                "/setup → 🧠 Авто мозг → Мозг 2 и вставь ключ Groq."
            )
        return

    target = _voice_or_audio_target(message)
    if target is None:
        return
    file_id, suffix = target

    await bot.send_chat_action(message.chat.id, "typing")
    try:
        audio_path = await _download_to_tempfile(bot, file_id, suffix)
    except Exception as exc:  # noqa: BLE001
        logger.exception("groq voice: download failed")
        await message.answer(
            f"Не получилось скачать голосовое: {_html_escape(str(exc))}"
        )
        return

    try:
        try:
            transcript = await transcribe_audio_with_groq(audio_path)
        finally:
            with contextlib.suppress(Exception):
                os.unlink(audio_path)
    except Exception as exc:  # noqa: BLE001
        logger.exception("groq voice: transcription failed")
        await message.answer(
            "Groq Whisper не справился: "
            f"<code>{_html_escape(str(exc))}</code>"
        )
        return

    if not transcript:
        await message.answer("🎙 Groq вернул пустой транскрипт.")
        return

    # Echo the transcript so the user can see what Groq heard, mirroring
    # the tg_bot+(13) behaviour the user described.
    await message.answer(f"🎙 <i>Услышал:</i> {_html_escape(transcript)}")

    # Now run the regular agent on the transcript text. We import lazily
    # to avoid a cycle with handlers.py (which imports this module
    # indirectly through main.py).
    if not storage.is_enabled():
        await message.answer(
            "Бот выключен. Включи через <code>/enable</code>."
        )
        return
    if storage.get_brain() == "devin":
        await message.answer(
            "Brain=devin — транскрипт записан, жди ответ от Devin."
        )
        return
    user_id = message.from_user.id if message.from_user else 0
    cwd = storage.get_cwd(user_id) if user_id else None

    from .handlers import _make_status_updater, _send_long

    on_status, finish_status = _make_status_updater(message)
    try:
        answer = await run_agent(user_id, transcript, cwd, on_status=on_status)
    except NoApiKeyError as exc:
        await finish_status()
        await message.answer(str(exc))
        return
    except Exception as exc:  # noqa: BLE001
        logger.exception("agent failed after groq transcript")
        await finish_status()
        await message.answer(f"Ошибка агента: {_html_escape(str(exc))}")
        return
    await finish_status()
    await _send_long(message, answer, code=False)


# ---- photo + caption → Groq Vision ---------------------------------------


def _largest_photo_size(message: Message) -> Any | None:
    if not message.photo:
        return None
    return max(message.photo, key=lambda p: p.width * p.height)


def _groq_override_ready(message: Message) -> bool:
    """Filter used so the photo+caption handler only matches when Groq
    is wired up. If it returns ``False`` the next router (photo
    chooser) gets the photo and asks the user what to do — that's the
    backward-compatible path the user signed off on.
    """
    return should_use_groq_override()


@groq_router.message(
    StateFilter(None), F.photo, F.caption, _groq_override_ready
)
async def on_photo_with_caption(
    message: Message, state: FSMContext, bot: Bot
) -> None:
    """Photo with non-empty caption → Groq Vision answer in the same chat.

    A caption is the user's intent — exactly the "указ" / instruction
    the user described. Without a caption (or without a Groq slot
    configured), we don't match and the existing photo chooser handles
    the photo.
    """
    if not _is_authorized(message):
        return
    caption = (message.caption or "").strip()
    if not caption:
        return  # belt-and-braces — F.caption already enforced non-empty

    photo = _largest_photo_size(message)
    if photo is None:
        return

    await bot.send_chat_action(message.chat.id, "typing")
    buf = io.BytesIO()
    try:
        await bot.download(photo.file_id, destination=buf)
    except Exception as exc:  # noqa: BLE001
        logger.exception("groq vision: download failed")
        await message.answer(
            f"Не получилось скачать фото: {_html_escape(str(exc))}"
        )
        return
    image_bytes = buf.getvalue()

    try:
        answer = await describe_image_with_groq(image_bytes, caption)
    except Exception as exc:  # noqa: BLE001
        logger.exception("groq vision: API call failed")
        await message.answer(
            "Groq Vision не справился: "
            f"<code>{_html_escape(str(exc))}</code>"
        )
        return

    from .handlers import _send_long

    await _send_long(message, answer, code=False)
