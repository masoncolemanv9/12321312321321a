"""Memory screen — shows how many turns the bot remembers and lets the
owner clear them or change the limit.

Data source is :mod:`bot.storage`'s built-in history (see
``storage.get_history`` / ``storage.append_history``), which caps at
``storage.get_history_limit() * 2`` total messages — user + assistant
alternating.

This addon is purely a UI shim: it does NOT replace or mirror the
storage, so memory remains consistent with what the agent actually
loads in :func:`bot.agent._build_messages`.
"""

from __future__ import annotations

import logging

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


class MemoryStates(StatesGroup):
    awaiting_history_limit = State()


def _kb_screen(has_history: bool) -> InlineKeyboardMarkup:
    limit = storage.get_history_limit()
    rows: list[list[InlineKeyboardButton]] = [
        [
            InlineKeyboardButton(
                text=f"📝 Кол-во сообщений: {limit}",
                callback_data="memory:set_limit",
            )
        ],
    ]
    if has_history:
        rows.append(
            [
                InlineKeyboardButton(
                    text="🗑 Очистить память", callback_data="memory:clear"
                )
            ]
        )
    rows.append(
        [InlineKeyboardButton(text="↩️ Назад", callback_data="memory:back")]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _count_pairs(history: list[dict]) -> tuple[int, int]:
    u = sum(1 for m in history if m.get("role") == "user")
    a = sum(1 for m in history if m.get("role") == "assistant")
    return u, a


def _screen_text(user_id: int) -> str:
    history = storage.get_history(user_id)
    u, a = _count_pairs(history)
    cap = storage.get_history_limit()
    return (
        "<b>🧠 Память бота</b>\n\n"
        "Это то, что бот <b>помнит</b> о вашем разговоре — последние "
        "сообщения подмешиваются в контекст каждого нового запроса.\n\n"
        f"Сейчас сохранено: <b>{u}</b> ваших · <b>{a}</b> ответов бота.\n"
        f"Лимит: <b>{cap}</b> пар (≈ {cap * 2} сообщений).\n\n"
        "<i>Нажми «📝 Кол-во сообщений» чтобы изменить лимит.</i>"
    )


async def show_screen(message_or_query) -> None:
    if isinstance(message_or_query, CallbackQuery):
        user_id = message_or_query.from_user.id if message_or_query.from_user else 0
        target_msg: Message | None = message_or_query.message  # type: ignore[assignment]
    else:
        user_id = message_or_query.from_user.id if message_or_query.from_user else 0
        target_msg = message_or_query

    history = storage.get_history(user_id)
    text = _screen_text(user_id)
    kb = _kb_screen(has_history=bool(history))
    if isinstance(message_or_query, CallbackQuery):
        try:
            await target_msg.edit_text(text, reply_markup=kb)  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            await target_msg.answer(text, reply_markup=kb)  # type: ignore[union-attr]
    else:
        await target_msg.answer(text, reply_markup=kb)


def build_memory_router() -> Router:
    router = Router(name="memory_addon")

    @router.callback_query(F.data == "memory:clear")
    async def on_clear(query: CallbackQuery) -> None:
        if query.from_user is None or query.message is None:
            await query.answer()
            return
        storage.clear_history(query.from_user.id)
        await query.answer("Память очищена", show_alert=False)
        await show_screen(query)

    @router.callback_query(F.data == "memory:set_limit")
    async def on_set_limit(query: CallbackQuery, state: FSMContext) -> None:
        if query.from_user is None or query.message is None:
            await query.answer()
            return
        current = storage.get_history_limit()
        await state.set_state(MemoryStates.awaiting_history_limit)
        await query.message.edit_text(
            f"<b>📝 Кол-во сообщений</b>\n\n"
            f"Сейчас: <b>{current}</b> пар (≈ {current * 2} сообщений).\n\n"
            "Введи новое число (1–500).\n"
            "Чем больше — тем дольше бот помнит контекст, но тем дороже "
            "каждый запрос (больше токенов).\n\n"
            "Рекомендации:\n"
            "• 10–20 — лёгкие разговоры, экономия токенов\n"
            "• 40–60 — нормальная работа с проектом\n"
            "• 100+ — длинные сессии (дорого, но бот не забывает)",
            reply_markup=InlineKeyboardMarkup(
                inline_keyboard=[
                    [InlineKeyboardButton(text="↩️ Отмена", callback_data="memory:cancel_input")]
                ]
            ),
        )
        await query.answer()

    @router.callback_query(F.data == "memory:cancel_input")
    async def on_cancel_input(query: CallbackQuery, state: FSMContext) -> None:
        await state.clear()
        await show_screen(query)
        await query.answer()

    @router.message(MemoryStates.awaiting_history_limit)
    async def capture_history_limit(message: Message, state: FSMContext) -> None:
        raw = (message.text or "").strip()
        try:
            n = int(raw)
            if n < 1 or n > 500:
                raise ValueError
        except ValueError:
            await message.answer("Нужно число от 1 до 500. Попробуй ещё раз или нажми «↩️ Отмена».")
            return
        storage.set_history_limit(n)
        await state.clear()
        await message.answer(
            f"Лимит памяти установлен: <b>{n}</b> пар (≈ {n * 2} сообщений)."
        )
        await show_screen(message)

    @router.callback_query(F.data == "memory:back")
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
                    "думает (белый бабл / белый + модель / тёмная плитка по центру).\n\n"
                    "<b>💾 Память бота</b> — сколько последних сообщений "
                    "помнит и кнопка очистить.\n\n"
                    "<b>💻 RAM</b> — лимит памяти сервера, расчёт и поведение "
                    "при превышении.",
                    reply_markup=kb,
                )
        except Exception:  # noqa: BLE001
            logger.debug("memory: back failed", exc_info=True)
        await query.answer()

    return router
