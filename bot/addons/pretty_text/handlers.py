"""Pretty-text — main-menu button that toggles Markdown→TG-HTML mode.

When the toggle is ON, **any** plain-text message the user sends (that
isn't a slash-command) is run through the chosen style and echoed back.

Style modes:

* ``mechanical`` — pure Markdown→TG-HTML via
  :func:`pretty_text.core.format_for_telegram`. No LLM call, ever, and
  the text is never modified.
* ``standard``   — LLM in *decoration-only* mode: adds bold/italic/
  emojis/line-breaks, but the underlying words are verified to be
  unchanged. If the model paraphrases anyway, we fall back to
  ``mechanical`` and warn the user.
* ``random``     — applies one of five hand-tuned rule-based presets
  (no LLM). The user can press "🎲 Случайная стилистика" again to
  cycle to the next preset. A "💾 Сохранить" button under each result
  pins the current preset for the ``saved`` style.
* ``saved``      — applies the user's pinned preset on every message.
  Falls back to ``mechanical`` if no preset has been saved yet.

Per-chat state under :mod:`bot.addons.state`:

* ``pretty_text.by_chat.<chat_id>.enabled`` — master toggle.
* ``pretty_text.by_chat.<chat_id>.style``   — one of the keys above.
* ``pretty_text.by_chat.<chat_id>.random_preset`` — current preset id
  cycling through ``random`` mode.
* ``pretty_text.by_chat.<chat_id>.saved_preset`` — preset id pinned by
  the "Сохранить" button. ``None`` until the user saves one.
"""

from __future__ import annotations

import contextlib
import logging

from aiogram import F, Router
from aiogram.filters import StateFilter
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from .. import state as addon_state
from .core import format_for_telegram
from .presets import (
    PRESET_LABELS,
    next_preset,
    render_preset,
)
from .standard import rewrite_standard_decor_only

logger = logging.getLogger(__name__)


# ---- state helpers -------------------------------------------------------


STYLE_MECHANICAL = "mechanical"
STYLE_STANDARD = "standard"
STYLE_RANDOM = "random"
STYLE_SAVED = "saved"

_DEFAULT_STYLE = STYLE_MECHANICAL

_STYLE_LABELS: dict[str, str] = {
    STYLE_MECHANICAL: "✂️ Без LLM (Markdown → TG-HTML)",
    STYLE_STANDARD: "📰 Стандарт (LLM только оформляет)",
    STYLE_RANDOM: "🎲 Случайная стилистика",
    STYLE_SAVED: "📁 Сохранённый стиль",
}


def is_enabled(chat_id: int) -> bool:
    return bool(addon_state.chat_get("pretty_text", chat_id, "enabled", False))


def _set_enabled(chat_id: int, value: bool) -> None:
    addon_state.chat_set("pretty_text", chat_id, "enabled", value)


def get_style(chat_id: int) -> str:
    val = str(
        addon_state.chat_get("pretty_text", chat_id, "style", _DEFAULT_STYLE)
        or _DEFAULT_STYLE
    )
    return val if val in _STYLE_LABELS else _DEFAULT_STYLE


def set_style(chat_id: int, style: str) -> None:
    if style not in _STYLE_LABELS:
        style = _DEFAULT_STYLE
    addon_state.chat_set("pretty_text", chat_id, "style", style)


def get_random_preset(chat_id: int) -> str | None:
    val = addon_state.chat_get("pretty_text", chat_id, "random_preset", None)
    return str(val) if val else None


def set_random_preset(chat_id: int, preset_id: str | None) -> None:
    addon_state.chat_set("pretty_text", chat_id, "random_preset", preset_id)


def get_saved_preset(chat_id: int) -> str | None:
    val = addon_state.chat_get("pretty_text", chat_id, "saved_preset", None)
    return str(val) if val else None


def set_saved_preset(chat_id: int, preset_id: str | None) -> None:
    addon_state.chat_set("pretty_text", chat_id, "saved_preset", preset_id)


# ---- keyboards -----------------------------------------------------------


def _kb_screen(enabled: bool, style: str) -> InlineKeyboardMarkup:
    toggle = (
        InlineKeyboardButton(text="⏸ Выключить", callback_data="pretty:off")
        if enabled
        else InlineKeyboardButton(text="🔘 Включить", callback_data="pretty:on")
    )
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [toggle],
            [
                InlineKeyboardButton(
                    text="🎨 Поменять стиль", callback_data="pretty:styles"
                )
            ],
            [InlineKeyboardButton(text="↩️ Назад", callback_data="pretty:back")],
        ]
    )


def _kb_styles(current: str, saved_label: str) -> InlineKeyboardMarkup:
    """Style picker keyboard. The current style is prefixed with ``• ``."""
    def _mark(key: str) -> str:
        return "• " if key == current else "   "

    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=f"{_mark(STYLE_MECHANICAL)}{_STYLE_LABELS[STYLE_MECHANICAL]}",
                    callback_data=f"pretty:style:{STYLE_MECHANICAL}",
                )
            ],
            [
                InlineKeyboardButton(
                    text=f"{_mark(STYLE_STANDARD)}{_STYLE_LABELS[STYLE_STANDARD]}",
                    callback_data=f"pretty:style:{STYLE_STANDARD}",
                )
            ],
            [
                InlineKeyboardButton(
                    text=f"{_mark(STYLE_RANDOM)}{_STYLE_LABELS[STYLE_RANDOM]}",
                    callback_data=f"pretty:style:{STYLE_RANDOM}",
                )
            ],
            [
                InlineKeyboardButton(
                    text=f"{_mark(STYLE_SAVED)}{_STYLE_LABELS[STYLE_SAVED]} ({saved_label})",
                    callback_data=f"pretty:style:{STYLE_SAVED}",
                )
            ],
            [InlineKeyboardButton(text="↩️ Назад", callback_data="pretty:home")],
        ]
    )


def _kb_after_send(style: str, preset_id: str | None) -> InlineKeyboardMarkup:
    """Inline keyboard shown under each formatted message. Adds 🔄/💾
    when the current style is ``random`` so the user can cycle or pin.
    """
    rows: list[list[InlineKeyboardButton]] = []
    if style == STYLE_RANDOM:
        rows.append(
            [
                InlineKeyboardButton(
                    text="🔄 Другой стиль", callback_data="pretty:random:next"
                ),
                InlineKeyboardButton(
                    text="💾 Сохранить", callback_data="pretty:random:save"
                ),
            ]
        )
    rows.append(
        [
            InlineKeyboardButton(
                text="🎨 Поменять стиль", callback_data="pretty:styles"
            ),
            InlineKeyboardButton(text="⏸ Выключить", callback_data="pretty:off"),
        ]
    )
    if style == STYLE_RANDOM and preset_id:
        preset_label = PRESET_LABELS.get(preset_id, preset_id)
        rows.insert(
            0,
            [
                InlineKeyboardButton(
                    text=f"⊳ Сейчас: {preset_label}",
                    callback_data="pretty:noop",
                )
            ],
        )
    return InlineKeyboardMarkup(inline_keyboard=rows)


# ---- screen text ---------------------------------------------------------


SCREEN_TEXT = (
    "<b>✨ Красивый текст</b>\n\n"
    "Когда режим <b>включён</b> — любое твоё сообщение (не команда) "
    "бот возьмёт, переоформит и пришлёт обратно красиво.\n\n"
    "Состояние: <b>{state}</b>\n"
    "Стиль: <b>{style}</b>"
)


STYLES_TEXT = (
    "<b>🎨 Стили оформления</b>\n\n"
    "<b>✂️ Без LLM</b> — мгновенно конвертирует Markdown в TG-HTML. "
    "Текст не меняется ни на символ.\n\n"
    "<b>📰 Стандарт</b> — LLM добавляет <b>жирный</b>, <i>курсив</i>, "
    "эмодзи и переносы строк. Слова сверяются с оригиналом: если "
    "модель попыталась их поменять — оформление откатывается.\n\n"
    "<b>🎲 Случайная стилистика</b> — рулетка из 5 готовых пресетов "
    "(новости / минимал / буллеты / просторно / канал-пост). Жми "
    "«🔄 Другой стиль» под результатом чтобы перебрать, «💾 Сохранить» "
    "чтобы зафиксировать понравившийся.\n\n"
    "<b>📁 Сохранённый стиль</b> — применяет тот пресет что ты "
    "запомнил из «случайной».\n\n"
    "Текущий стиль: <b>{style}</b>"
)


async def show_screen(message_or_query) -> None:
    """Render the pretty-text main screen via Message or CallbackQuery."""
    if isinstance(message_or_query, CallbackQuery):
        chat_id = message_or_query.message.chat.id  # type: ignore[union-attr]
    else:
        chat_id = message_or_query.chat.id
    enabled = is_enabled(chat_id)
    style = get_style(chat_id)
    text = SCREEN_TEXT.format(
        state="ВКЛ ✅" if enabled else "ВЫКЛ ⛔",
        style=_STYLE_LABELS.get(style, style),
    )
    kb = _kb_screen(enabled, style)
    if isinstance(message_or_query, CallbackQuery):
        try:
            await message_or_query.message.edit_text(text, reply_markup=kb)  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            await message_or_query.message.answer(text, reply_markup=kb)  # type: ignore[union-attr]
    else:
        await message_or_query.answer(text, reply_markup=kb)


async def show_styles(query: CallbackQuery) -> None:
    if query.message is None:
        await query.answer()
        return
    chat_id = query.message.chat.id
    current = get_style(chat_id)
    saved_id = get_saved_preset(chat_id)
    saved_label = PRESET_LABELS.get(saved_id, "не задан") if saved_id else "не задан"
    text = STYLES_TEXT.format(style=_STYLE_LABELS.get(current, current))
    kb = _kb_styles(current, saved_label)
    try:
        await query.message.edit_text(text, reply_markup=kb)
    except Exception:  # noqa: BLE001
        await query.message.answer(text, reply_markup=kb)


# ---- format dispatch ----------------------------------------------------


async def _apply_style(
    text: str, style: str, chat_id: int
) -> tuple[str, str | None]:
    """Run the chosen style and return ``(formatted_html, preset_id)``.

    ``preset_id`` is the preset that was actually applied — only set
    when ``style`` is ``random`` or ``saved`` (used for the post-send
    keyboard). For other styles it's ``None``.
    """
    if style == STYLE_STANDARD:
        try:
            decorated = await rewrite_standard_decor_only(text)
            return format_for_telegram(decorated), None
        except Exception as exc:  # noqa: BLE001
            logger.warning("pretty-text standard fallback: %s", exc)
            return format_for_telegram(text), None
    if style == STYLE_RANDOM:
        preset = get_random_preset(chat_id) or next_preset(None)
        decorated = render_preset(preset, text)
        return format_for_telegram(decorated), preset
    if style == STYLE_SAVED:
        saved = get_saved_preset(chat_id)
        if saved:
            decorated = render_preset(saved, text)
            return format_for_telegram(decorated), saved
        # No saved preset → fall back to mechanical.
        return format_for_telegram(text), None
    # mechanical / unknown
    return format_for_telegram(text), None


# ---- router --------------------------------------------------------------


def build_pretty_text_router() -> Router:
    router = Router(name="pretty_text_addon")

    @router.callback_query(F.data == "pretty:on")
    async def on_enable(query: CallbackQuery) -> None:
        if query.message is None:
            await query.answer()
            return
        _set_enabled(query.message.chat.id, True)
        await query.answer("Режим включён")
        await show_screen(query)

    @router.callback_query(F.data == "pretty:off")
    async def on_disable(query: CallbackQuery) -> None:
        if query.message is None:
            await query.answer()
            return
        _set_enabled(query.message.chat.id, False)
        await query.answer("Режим выключен")
        await show_screen(query)

    @router.callback_query(F.data == "pretty:styles")
    async def on_styles(query: CallbackQuery) -> None:
        await show_styles(query)

    @router.callback_query(F.data == "pretty:home")
    async def on_home(query: CallbackQuery) -> None:
        await show_screen(query)
        await query.answer()

    @router.callback_query(F.data.startswith("pretty:style:"))
    async def on_pick_style(query: CallbackQuery) -> None:
        if query.message is None:
            await query.answer()
            return
        style = (query.data or "").split(":", 2)[-1]
        chat_id = query.message.chat.id
        set_style(chat_id, style)
        # When picking "random", seed a preset if none yet so the first
        # message uses something concrete.
        if style == STYLE_RANDOM and get_random_preset(chat_id) is None:
            set_random_preset(chat_id, next_preset(None))
        await query.answer(f"Стиль: {_STYLE_LABELS.get(style, style)}")
        await show_styles(query)

    @router.callback_query(F.data == "pretty:random:next")
    async def on_random_next(query: CallbackQuery) -> None:
        if query.message is None:
            await query.answer()
            return
        chat_id = query.message.chat.id
        nxt = next_preset(get_random_preset(chat_id))
        set_random_preset(chat_id, nxt)
        await query.answer(
            f"Стиль: {PRESET_LABELS.get(nxt, nxt)} — пришли текст ещё раз"
        )

    @router.callback_query(F.data == "pretty:random:save")
    async def on_random_save(query: CallbackQuery) -> None:
        if query.message is None:
            await query.answer()
            return
        chat_id = query.message.chat.id
        current = get_random_preset(chat_id) or next_preset(None)
        set_saved_preset(chat_id, current)
        await query.answer(
            f"Сохранено: {PRESET_LABELS.get(current, current)}",
            show_alert=False,
        )

    @router.callback_query(F.data == "pretty:noop")
    async def on_noop(query: CallbackQuery) -> None:
        await query.answer()

    @router.callback_query(F.data == "pretty:back")
    async def on_back(query: CallbackQuery) -> None:
        try:
            from ...wizard import _kb_main_after_claim  # type: ignore
            uid = query.from_user.id if query.from_user else None
            kb = _kb_main_after_claim(uid)
            if query.message is not None:
                await query.message.edit_text(
                    "Главное меню. Выбери что сделать:", reply_markup=kb
                )
        except Exception:  # noqa: BLE001
            logger.debug("pretty: back failed", exc_info=True)
        await query.answer()

    async def _pretty_enabled(message: Message) -> bool:
        return is_enabled(message.chat.id)

    @router.message(
        StateFilter(None),
        F.text & ~F.text.startswith("/"),
        _pretty_enabled,
    )
    async def on_text(message: Message) -> None:
        text = message.text or ""
        chat_id = message.chat.id
        style = get_style(chat_id)

        # Route progress through the shared "Соображалка" runner so the
        # user's chosen style (white bubble / white + model / black-
        # centered indicator) is respected here too. The LLM-backed
        # Standard branch is the only one that takes long enough to be
        # worth a progress indicator — for mechanical presets we skip
        # status updates entirely.
        from ...config import DEFAULT_MODEL
        from ...storage import storage as _st
        from ..thinking_style import make_status_runner

        status_update = None
        status_finish = None
        if style == STYLE_STANDARD:
            # Surface the chat model for ⚪ Белая + модель — same one that
            # rewrites the text in ``standard.rewrite_standard_decor_only``.
            def _model_hint() -> str:
                return _st.get_model() or DEFAULT_MODEL

            status_update, status_finish = make_status_runner(
                message, model_hint=_model_hint
            )
            await status_update("✨ Оформляю текст…")

        try:
            formatted, preset_id = await _apply_style(text, style, chat_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("pretty-text: format failed")
            if status_finish is not None:
                with contextlib.suppress(Exception):
                    await status_finish()
            with contextlib.suppress(Exception):
                await message.answer(f"Не получилось оформить: {exc}")
            return
        if status_finish is not None:
            with contextlib.suppress(Exception):
                await status_finish()

        kb = _kb_after_send(style, preset_id)
        await message.answer(
            formatted,
            reply_markup=kb,
            disable_web_page_preview=True,
        )

    return router
