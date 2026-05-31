"""UI + helpers for the "Соображалка" (thinking-style) addon.

The addon stores a single per-chat string under
``addons.thinking_style.by_chat.<chat_id>.style``. Valid values:

* ``"white"`` — the rich, multi-line thinking stream (default).
* ``"white_model"`` — same as ``white`` plus the model name suffix.
* ``"typing"`` — only the Telegram native ``send_chat_action("typing")``
  indicator; no status message is rendered.

``make_status_runner`` returns ``(update, finish)`` callables — drop-in
replacement for the inline ``_make_status_updater`` that used to live
in ``handlers.py``. Now both the LLM chat and the per-addon pipelines
share the same renderer so the user's "Соображалка" choice applies
everywhere uniformly.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from aiogram import Bot, F, Router
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from .. import state as addon_state

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---- constants -----------------------------------------------------------


STYLE_WHITE = "white"
STYLE_WHITE_MODEL = "white_model"
STYLE_TYPING = "typing"

_DEFAULT_STYLE = STYLE_WHITE

_STYLE_LABELS: dict[str, str] = {
    STYLE_WHITE: "⚪ Белая",
    STYLE_WHITE_MODEL: "⚪ Белая + модель",
    STYLE_TYPING: "⬛ Чёрная по центру",
}

_STYLE_HELP = (
    "<b>🧠 Соображалка</b> — как бот показывает, что он думает.\n\n"
    "<b>⚪ Белая</b> — статус-сообщение в чате, текст меняется по шагам "
    "(«Думаю…» → «Читаю файл…» → «Готово»). Тот же стиль, что и сейчас.\n\n"
    "<b>⚪ Белая + модель</b> — то же самое, но сбоку видно <i>какая "
    "модель</i> сейчас отвечает (полезно если включён fallback на "
    "несколько моделей).\n\n"
    "<b>⬛ Чёрная по центру</b> — никаких сообщений в чате. "
    "Прогресс идёт родным индикатором Telegram вверху экрана: "
    "<i>«…печатает»</i> («думаю»), <i>«…отправляет фото»</i> («рисую»), "
    "<i>«…записывает видео»</i> («анимирую»). На телефоне "
    "этот индикатор стоит по центру сверху.\n\n"
    "Текущий стиль: <b>{style}</b>"
)

# ---- chat-action mapper --------------------------------------------------

# Map status-text fragments to Telegram chat-action types. Used by the
# "⚫ Чёрная по центру" branch so the native indicator at the top of
# the chat reflects what the bot is doing right now — e.g. «rendering an
# image» → «… отправляет фото», «drafting a reply» → «… печатает».
_ACTION_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        (
            "рису",
            "рисую",
            "рисуют",
            "аннот",
            "рамк",
            "стрелк",
            "скриншот",
            "фото",
            "изображ",
            "картин",
        ),
        "upload_photo",
    ),
    (
        (
            "аним",
            "видео",
            "клип",
            "ролик",
        ),
        "record_video",
    ),
    (
        (
            "скач",
            "загруж",
            "залив",
            "файл",
        ),
        "upload_document",
    ),
)


def _action_for_status(text: str) -> str:
    """Pick a Telegram chat-action that best matches ``text``.

    Telegram only accepts a fixed set of actions (typing, upload_photo,
    record_video, upload_document…). We map by simple substring on the
    user-visible verb so the indicator rotates in sync with the
    pipeline step. Falls back to ``"typing"`` («… печатает»).
    """
    if not text:
        return "typing"
    lowered = text.lower()
    for needles, action in _ACTION_RULES:
        for needle in needles:
            if needle in lowered:
                return action
    return "typing"


# ---- state helpers -------------------------------------------------------


def get_style(chat_id: int) -> str:
    raw = addon_state.chat_get(
        "thinking_style", chat_id, "style", _DEFAULT_STYLE
    )
    style = str(raw or _DEFAULT_STYLE)
    if style not in _STYLE_LABELS:
        return _DEFAULT_STYLE
    return style


def _set_style(chat_id: int, value: str) -> None:
    if value not in _STYLE_LABELS:
        value = _DEFAULT_STYLE
    addon_state.chat_set("thinking_style", chat_id, "style", value)


def is_typing_only(chat_id: int) -> bool:
    """True when the user picked the ``typing``-only style."""
    return get_style(chat_id) == STYLE_TYPING


def show_model_suffix(chat_id: int) -> bool:
    return get_style(chat_id) == STYLE_WHITE_MODEL


# ---- keyboards -----------------------------------------------------------


def _kb_screen(current: str) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    for key, label in _STYLE_LABELS.items():
        marker = "• " if key == current else "   "
        rows.append(
            [
                InlineKeyboardButton(
                    text=f"{marker}{label}",
                    callback_data=f"thinking:style:{key}",
                )
            ]
        )
    rows.append(
        [InlineKeyboardButton(text="↩️ Вернуться", callback_data="thinking:back")]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


async def show_screen(target: Message | CallbackQuery) -> None:
    if isinstance(target, CallbackQuery):
        if target.message is None:
            await target.answer()
            return
        chat_id = target.message.chat.id
    else:
        chat_id = target.chat.id
    current = get_style(chat_id)
    text = _STYLE_HELP.format(style=_STYLE_LABELS[current])
    kb = _kb_screen(current)
    if isinstance(target, CallbackQuery):
        try:
            await target.message.edit_text(  # type: ignore[union-attr]
                text, reply_markup=kb, disable_web_page_preview=True
            )
        except Exception:  # noqa: BLE001
            await target.message.answer(  # type: ignore[union-attr]
                text, reply_markup=kb, disable_web_page_preview=True
            )
    else:
        await target.answer(text, reply_markup=kb, disable_web_page_preview=True)


# ---- status-runner -------------------------------------------------------


StatusUpdate = Callable[[str], Awaitable[None]]
StatusFinish = Callable[[], Awaitable[None]]

# Soft Telegram limit is ~1 edit/sec/chat. Pick something safely under it
# so a long chain of fast updates doesn't flood-control the bot.
_MIN_EDIT_INTERVAL = 0.8


def make_status_runner(
    message: Message,
    bot: Bot | None = None,
    *,
    model_hint: Callable[[], str] | None = None,
) -> tuple[StatusUpdate, StatusFinish]:
    """Build ``(update, finish)`` honoring the chat's "Соображалка" style.

    * For ``white`` / ``white_model`` styles the first ``update(text)``
      call sends a fresh status bubble and subsequent calls edit it in
      place (rate-limited).
    * For ``typing`` the function does not send any chat bubble — it
      periodically refreshes ``send_chat_action("typing")`` so Telegram
      keeps showing the native indicator. ``finish()`` cancels the
      refresh task.

    ``model_hint`` is called lazily each time we need to render a status
    line in the ``white_model`` style. We keep it as a callable so the
    caller can update which model is active mid-stream (fallback chain).
    """
    chat_id = message.chat.id
    style = get_style(chat_id)
    effective_bot = bot or message.bot

    def _decorate(text: str) -> str:
        text = (text or "").strip() or "🔄"
        if style == STYLE_WHITE_MODEL and model_hint is not None:
            try:
                model = model_hint() or ""
            except Exception:  # noqa: BLE001
                model = ""
            if model:
                # Tail with a soft separator + monospaced model so the
                # line stays readable in both light and dark themes.
                return f"{text}  ·  <code>{model}</code>"
        return text

    # ---- typing / "black, centered" branch ---------------------------
    # Renders progress as Telegram's native chat-action indicator at
    # the TOP of the chat (under the bot's name) — what the user calls
    # "чёрная по центру". No bubble appears in the message list. The
    # action type rotates with the current status verb so the user
    # sees «...печатает» → «...отправляет фото» → «...записывает видео»
    # roughly mapping to «думаю → рисую → анимирую».
    #
    # In Telegram the chat-action expires after ~5s so we refresh in a
    # background task. ``update(text)`` only updates which action is
    # active; ``finish()`` stops the loop.
    if style == STYLE_TYPING:
        stop_event = asyncio.Event()
        current_action: dict = {"action": "typing"}

        async def _typing_loop() -> None:
            while not stop_event.is_set():
                with contextlib.suppress(Exception):
                    if effective_bot is not None:
                        await effective_bot.send_chat_action(
                            chat_id, current_action["action"]
                        )
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=4.0)
                except TimeoutError:
                    continue

        task = asyncio.create_task(_typing_loop())

        async def update(text: str) -> None:
            current_action["action"] = _action_for_status(text)
            # Refresh immediately so the indicator switches without
            # waiting for the next 4-second loop tick.
            with contextlib.suppress(Exception):
                if effective_bot is not None:
                    await effective_bot.send_chat_action(
                        chat_id, current_action["action"]
                    )

        async def finish() -> None:
            stop_event.set()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(task, timeout=1.0)

        return update, finish

    # ---- white / white_model branch ----------------------------------
    state: dict = {"msg": None, "last_text": "", "last_edit": 0.0}

    async def update(text: str) -> None:
        rendered = _decorate(text)
        now = time.monotonic()
        if state["msg"] is None:
            try:
                state["msg"] = await message.answer(rendered)
                state["last_text"] = rendered
                state["last_edit"] = now
            except Exception:  # noqa: BLE001
                logger.debug("status: send failed", exc_info=True)
            return
        if rendered == state["last_text"]:
            return
        if now - state["last_edit"] < _MIN_EDIT_INTERVAL:
            return
        try:
            await state["msg"].edit_text(rendered)
            state["last_text"] = rendered
            state["last_edit"] = now
        except Exception:  # noqa: BLE001
            logger.debug("status: edit failed", exc_info=True)

    async def finish() -> None:
        msg = state["msg"]
        if msg is None:
            return
        with contextlib.suppress(Exception):
            await msg.delete()

    return update, finish


# ---- router --------------------------------------------------------------


def build_thinking_style_router() -> Router:
    router = Router(name="thinking_style_addon")

    @router.callback_query(F.data.startswith("thinking:style:"))
    async def on_pick(query: CallbackQuery) -> None:
        if query.message is None or query.data is None:
            await query.answer()
            return
        style = query.data.split(":", 2)[-1]
        _set_style(query.message.chat.id, style)
        await query.answer(
            f"Стиль: {_STYLE_LABELS.get(style, style)}"
        )
        await show_screen(query)

    @router.callback_query(F.data == "thinking:back")
    async def on_back(query: CallbackQuery) -> None:
        """Back-arrow on the Соображалка screen returns to the Settings
        sub-screen (which is where the user came from)."""
        try:
            from aiogram.types import (
                InlineKeyboardButton as _IKB,
            )
            from aiogram.types import (
                InlineKeyboardMarkup as _IKM,
            )
            kb = _IKM(
                inline_keyboard=[
                    [_IKB(text="🧠 Соображалка", callback_data="main:thinking")],
                    [_IKB(text="💾 Память бота", callback_data="main:memory")],
                    [_IKB(text="💻 RAM", callback_data="main:ram")],
                    [_IKB(text="↩️ Назад", callback_data="settings:back")],
                ]
            )
            if query.message is not None:
                await query.message.edit_text(
                    "<b>⚙️ Настройки</b>\n\n"
                    "<b>🧠 Соображалка</b> — как бот показывает что он "
                    "думает (белый бабл / белый + модель / тёмная плитка по центру).\n\n"
                    "<b>💾 Память бота</b> — сколько последних сообщений "
                    "помнит и кнопка очистить.",
                    reply_markup=kb,
                )
        except Exception:  # noqa: BLE001
            logger.debug("thinking: back failed", exc_info=True)
        await query.answer()

    return router
