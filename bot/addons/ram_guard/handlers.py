"""UI handlers for the "💻 RAM" settings sub-screen.

Three controls live here:

* «📏 Лимит RAM: N MB» — owner inputs an integer in megabytes.
  Default 0 means «no guard». Render Free has ~512MB; recommend 400.
* «⚙️ Поведение при лимите» — toggle between two strategies:
    - compress: at 80% of the limit, the agent loop truncates large
      tool results inside the in-flight ``messages`` list; at 95% it
      aborts the run with a friendly «память на пределе» message.
    - refuse: at the limit, abort the run with the same message; no
      pre-emptive compression.
* «📊 Расчёт RAM заранее» — vkl/vykl. When ON, every status-bubble
  update gets a «(RSS XXX MB)» suffix so the owner sees how much
  the bot is using right now.

Cancellation of an in-flight task is wired separately in
:mod:`bot.agent` and surfaced via the «🛑 Отмена» button attached
to the status bubble — it is NOT controlled from this menu.
"""

from __future__ import annotations

import logging
import os

from aiogram import F, Router
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from ...storage import storage

logger = logging.getLogger(__name__)


class RamStates(StatesGroup):
    awaiting_limit = State()


def current_rss_mb() -> int:
    """Return the current process RSS in megabytes.

    Uses :mod:`psutil` when available, falls back to ``/proc/self/status``
    on Linux (Render runs Linux), and returns 0 if neither is usable —
    the caller treats 0 as "RSS unknown, skip guard / suffix".
    """
    try:
        import psutil  # type: ignore[import-untyped]

        return int(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024))
    except Exception:  # noqa: BLE001
        pass
    # ``/proc/self/status`` exposes ``VmRSS:   12345 kB``.
    try:
        with open("/proc/self/status", encoding="ascii") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    return kb // 1024
    except Exception:  # noqa: BLE001
        return 0
    return 0


def _kb_screen() -> InlineKeyboardMarkup:
    limit = storage.get_ram_limit_mb()
    behavior = storage.get_ram_behavior()
    show = storage.get_ram_show()
    limit_text = f"📏 Лимит RAM: {limit} MB" if limit > 0 else "📏 Лимит RAM: ✕ выкл"
    behavior_text = (
        "⚙️ При лимите: сжимать → отказать"
        if behavior == "compress"
        else "⚙️ При лимите: просто отказать"
    )
    show_text = "📊 Расчёт RAM: ВКЛ" if show else "📊 Расчёт RAM: ВЫКЛ"
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=limit_text, callback_data="ram:set_limit")],
            [InlineKeyboardButton(text=behavior_text, callback_data="ram:toggle_behavior")],
            [InlineKeyboardButton(text=show_text, callback_data="ram:toggle_show")],
            [InlineKeyboardButton(text="↩️ Назад", callback_data="ram:back")],
        ]
    )


def _screen_text() -> str:
    rss = current_rss_mb()
    limit = storage.get_ram_limit_mb()
    behavior = storage.get_ram_behavior()
    show = storage.get_ram_show()
    pct = ""
    if limit > 0:
        pct = f" ({int(rss * 100 / limit)}% от лимита)"
    return (
        "<b>💻 RAM</b>\n\n"
        f"Текущее RSS бота: <b>{rss} MB</b>{pct}.\n"
        f"Лимит: <b>{limit if limit else '—'} MB</b>.\n"
        f"Поведение при лимите: <b>{'сжимать → отказать' if behavior == 'compress' else 'просто отказать'}</b>.\n"
        f"Расчёт RAM в соображалке: <b>{'ВКЛ' if show else 'ВЫКЛ'}</b>.\n\n"
        "<i>Render Free даёт ~512 MB. Рекомендуем лимит 400 MB — "
        "это оставит запас для сборки страниц и потоков. Если "
        "поставить 0 — guard выключен.</i>\n\n"
        "<b>Сжимать → отказать:</b> когда RSS близок к 80% от лимита, "
        "бот урезает большие tool-результаты из истории текущей задачи "
        "(чтобы LLM не таскал гигантские чанки). При 95% — отказывает: "
        "«Память на пределе, разбей задачу или нажми /reset».\n\n"
        "<b>Просто отказать:</b> ничего не урезает, только отказывает "
        "когда лимит превышен.\n\n"
        "<b>Расчёт RAM:</b> в плашке-соображалке («Думаю…») рядом будет "
        "видно «RSS 180 MB» — удобно понимать, когда сервер близок к "
        "потолку."
    )


async def show_screen(message_or_query) -> None:
    if isinstance(message_or_query, CallbackQuery):
        target_msg: Message | None = message_or_query.message  # type: ignore[assignment]
    else:
        target_msg = message_or_query

    text = _screen_text()
    kb = _kb_screen()
    if isinstance(message_or_query, CallbackQuery):
        try:
            await target_msg.edit_text(text, reply_markup=kb)  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            await target_msg.answer(text, reply_markup=kb)  # type: ignore[union-attr]
    else:
        await target_msg.answer(text, reply_markup=kb)


def build_ram_router() -> Router:
    router = Router(name="ram_guard_addon")

    @router.callback_query(F.data == "ram:set_limit")
    async def on_set_limit(query: CallbackQuery, state: FSMContext) -> None:
        if query.message is None:
            await query.answer()
            return
        await state.set_state(RamStates.awaiting_limit)
        current = storage.get_ram_limit_mb()
        await query.message.edit_text(
            f"<b>📏 Лимит RAM</b>\n\n"
            f"Сейчас: <b>{current if current else 'выкл'}</b> MB.\n\n"
            "Введи число в мегабайтах (0–4096). 0 — выключить guard.\n\n"
            "Render Free: ~512 MB всего. Безопасный лимит: <b>400</b>.\n"
            "Render Hobby/Starter: 512 MB → 400.\n"
            "Своя VPS на 1 GB: 800.\n"
            "Своя VPS на 2 GB+: 1600+.",
            reply_markup=InlineKeyboardMarkup(
                inline_keyboard=[
                    [InlineKeyboardButton(text="↩️ Отмена", callback_data="ram:cancel_input")]
                ]
            ),
        )
        await query.answer()

    @router.callback_query(F.data == "ram:cancel_input")
    async def on_cancel_input(query: CallbackQuery, state: FSMContext) -> None:
        await state.clear()
        await show_screen(query)
        await query.answer()

    @router.message(RamStates.awaiting_limit)
    async def capture_limit(message: Message, state: FSMContext) -> None:
        raw = (message.text or "").strip()
        try:
            n = int(raw)
            if n < 0 or n > 4096:
                raise ValueError
        except ValueError:
            await message.answer("Нужно число от 0 до 4096. Попробуй ещё раз или нажми «↩️ Отмена».")
            return
        storage.set_ram_limit_mb(n)
        await state.clear()
        if n == 0:
            await message.answer("Guard выключен. RSS никак не ограничивается.")
        else:
            await message.answer(f"Лимит RAM установлен: <b>{n}</b> MB.")
        await show_screen(message)

    @router.callback_query(F.data == "ram:toggle_behavior")
    async def on_toggle_behavior(query: CallbackQuery) -> None:
        current = storage.get_ram_behavior()
        new = "refuse" if current == "compress" else "compress"
        storage.set_ram_behavior(new)
        await query.answer(
            "Поведение: " + ("сжимать → отказать" if new == "compress" else "просто отказать"),
            show_alert=False,
        )
        await show_screen(query)

    @router.callback_query(F.data == "ram:toggle_show")
    async def on_toggle_show(query: CallbackQuery) -> None:
        new = not storage.get_ram_show()
        storage.set_ram_show(new)
        await query.answer("Расчёт RAM: " + ("ВКЛ" if new else "ВЫКЛ"))
        await show_screen(query)

    @router.callback_query(F.data == "ram:back")
    async def on_back(query: CallbackQuery, state: FSMContext) -> None:
        await state.clear()
        try:
            kb = InlineKeyboardMarkup(
                inline_keyboard=[
                    [InlineKeyboardButton(text="🧠 Соображалка", callback_data="main:thinking")],
                    [InlineKeyboardButton(text="💾 Память бота", callback_data="main:memory")],
                    [InlineKeyboardButton(text="💻 RAM", callback_data="main:ram")],
                    [InlineKeyboardButton(text="↩️ Назад", callback_data="settings:back")],
                ]
            )
            if query.message is not None:
                await query.message.edit_text(
                    "<b>⚙️ Настройки</b>\n\n"
                    "<b>🧠 Соображалка</b> — как бот показывает что он "
                    "думает.\n\n"
                    "<b>💾 Память бота</b> — сколько последних сообщений "
                    "помнит и кнопка очистить.\n\n"
                    "<b>💻 RAM</b> — лимит памяти сервера, расчёт и "
                    "поведение при превышении.",
                    reply_markup=kb,
                )
        except Exception:  # noqa: BLE001
            logger.debug("ram: back failed", exc_info=True)
        await query.answer()

    return router
