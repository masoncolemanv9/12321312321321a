"""Telegram UI for the media brain (img2video / img2img / rmbg).

Two distinct flows live here, both glued together by a single router so
they share callback prefixes (``media:`` for the wizard, ``photo:`` for
the per-photo handler).

1. ``media:*`` callbacks — ``/setup → 🎬 Мозг для видео и фото``
   wizard. Manages up to 3 provider slots (URL + API key + per-task
   model) and which slot is *active* for each task. Slot names are
   auto-derived from the URL (``deapi``, ``fal``, …) and fall back to
   the slot id when the URL is unknown.

2. ``photo:*`` callbacks + ``F.photo`` entry — runs whenever the owner
   sends a photo. Asks Фото/Видео, then drills into a prompt picker
   (img2img) or a quality picker (img2video) or a one-tap background
   removal (img-rmbg). Photo bytes are not cached — the file_id stashed
   in FSM data is re-downloaded on every leaf action (Telegram keeps
   file_id alive long enough for our use).
"""

from __future__ import annotations

import contextlib
import logging

import httpx
from aiogram import F, Router
from aiogram.filters import StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from .media import (
    KNOWN_MEDIA_PROVIDERS,
    VIDEO_QUALITY_MAP,
    MediaError,
    image_to_image,
    image_to_video,
    remove_background,
)
from .media_llm import NoLLMError, generate_prompt_variants, translate_to_english
from .storage import storage

logger = logging.getLogger(__name__)
media_router = Router(name="media")


# ---- helpers -------------------------------------------------------------


def _is_owner(user_id: int | None) -> bool:
    """Backward-compat alias for "can change media settings".

    Media-generation settings (API URL, API key, model slots) used to be
    strictly owner-only. Now they're gated on :func:`bot.access.can_admin`
    so that co-owners and users in ``full_public`` mode can change them
    too.
    """
    from .access import can_admin

    return can_admin(user_id)


def _mask(s: str, *, head: int = 4, tail: int = 4) -> str:
    if not s:
        return "—"
    if len(s) <= head + tail + 1:
        return "***"
    return f"{s[:head]}…{s[-tail:]}"


def _html(s: str) -> str:
    # Avoid pulling handlers._html_escape — keep this file standalone.
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class _ProgressShim:
    """Drop-in replacement for an in-chat progress message that routes
    edits through the shared :func:`make_status_runner`.

    The media flows historically did::

        progress_msg = await msg.answer("📥 Качаю фото…")
        await progress_msg.edit_text("…")
        await progress_msg.delete()

    Wrapping that pattern in this shim means the user's chosen
    "Соображалка" style is honoured here too — in ⬛ "чёрная по центру"
    no bubble is rendered at all and progress shows up as Telegram's
    native action indicator at the top of the chat. White / white+model
    keep behaving like before.
    """

    def __init__(self, msg: Message, initial: str) -> None:
        from .addons.thinking_style import make_status_runner
        from .config import DEFAULT_MODEL

        # ⚪ Белая + модель appends "· <model>" after every status line.
        # The user-visible "thinking" steps in media flows are LLM calls
        # (prompt-helper / RU→EN translate), so we surface the same
        # chat model as the main agent uses.
        def _model_hint() -> str:
            return storage.get_model() or DEFAULT_MODEL

        self._update, self._finish = make_status_runner(
            msg, model_hint=_model_hint
        )
        self._initial = initial

    async def open(self) -> _ProgressShim:
        await self._update(self._initial)
        return self

    async def edit_text(self, text: str, **_kwargs) -> None:
        # ``_kwargs`` swallows aiogram-specific kwargs that the runner
        # doesn't care about (parse_mode etc.) — the runner always uses
        # the default HTML parse mode anyway.
        await self._update(text)

    async def delete(self) -> None:
        await self._finish()


async def _open_progress(msg: Message, initial: str) -> _ProgressShim:
    """Build and prime a :class:`_ProgressShim` in one call."""
    shim = _ProgressShim(msg, initial)
    return await shim.open()


# ---- FSM states ----------------------------------------------------------


class MediaWizardStates(StatesGroup):
    awaiting_url = State()
    awaiting_api_key = State()
    awaiting_model_value = State()  # used for video_model / photo_model / rmbg_model


class PhotoFlowStates(StatesGroup):
    """FSM states for the photo flow.

    The ``awaiting_*`` states only fire under their own ``StateFilter``
    so plain text outside the flow still reaches the LLM chat handler.
    Variant selection is callback-driven (no text capture); only the
    keywords/prompt text-input branches need a state.
    """

    # Photo branch (img2img).
    awaiting_photo_prompt = State()
    awaiting_photo_keywords = State()
    # Video branch (img2video) — separate states so video prompt entry
    # does not collide with photo prompt entry if user re-uploads.
    awaiting_video_prompt = State()
    awaiting_video_keywords = State()


# ---- keyboards: media wizard --------------------------------------------


def _slot_summary_line(slot_id: str) -> str:
    slot = storage.get_media_slot(slot_id)
    if slot is None:
        return f"<b>{slot_id}.</b> <i>пусто</i>"
    name = slot.name or slot_id
    parts: list[str] = []
    parts.append(f"URL: <code>{_html(slot.url) if slot.url else '—'}</code>")
    parts.append(f"key: <code>{_mask(slot.api_key)}</code>")
    if slot.video_model:
        parts.append(f"🎬 <code>{_html(slot.video_model)}</code>")
    if slot.photo_model:
        parts.append(f"📷 <code>{_html(slot.photo_model)}</code>")
    if slot.rmbg_model:
        parts.append(f"🪄 <code>{_html(slot.rmbg_model)}</code>")
    return f"<b>{slot_id} ({_html(name)}):</b> " + " · ".join(parts)


def _active_summary_line() -> str:
    parts: list[str] = []
    for task in ("video", "photo", "rmbg"):
        slot = storage.get_active_media_slot(task)
        label = {"video": "🎬", "photo": "📷", "rmbg": "🪄"}[task]
        if slot is None:
            parts.append(f"{label}<i>—</i>")
        else:
            parts.append(f"{label}<b>{_html(slot.name or slot.slot_id)}</b>")
    return "Активные слоты: " + " | ".join(parts)


def _kb_media_menu() -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    for sid in storage.MEDIA_SLOT_IDS:
        slot = storage.get_media_slot(sid)
        if slot is None:
            label = f"⚪️ Слот {sid} — пусто"
        else:
            marker = "✅" if slot.configured else "⚙️"
            name = slot.name or sid
            label = f"{marker} Слот {sid} — {name}"
        rows.append([InlineKeyboardButton(text=label, callback_data=f"media:slot:{sid}")])
    rows.append([InlineKeyboardButton(text="↩️  Назад к мозгам", callback_data="media:back")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _kb_media_slot(slot_id: str) -> InlineKeyboardMarkup:
    slot = storage.get_media_slot(slot_id)
    url = (slot.url if slot else "") or "—"
    api = _mask(slot.api_key) if slot else "—"
    vmodel = (slot.video_model if slot else "") or "—"
    pmodel = (slot.photo_model if slot else "") or "—"
    rmodel = (slot.rmbg_model if slot else "") or "—"

    active_marks: dict[str, str] = {}
    for task in ("video", "photo", "rmbg"):
        cur = storage.get_active_media_slot(task)
        active_marks[task] = "▶️" if cur is not None and cur.slot_id == slot_id else "⏵"

    rows: list[list[InlineKeyboardButton]] = [
        [InlineKeyboardButton(text=f"🌐 URL: {url[:40]}", callback_data=f"media:edit:{slot_id}:url")],
        [InlineKeyboardButton(text=f"🔑 API key: {api}", callback_data=f"media:edit:{slot_id}:api_key")],
        [InlineKeyboardButton(text=f"🎬 Видео-модель: {vmodel[:30]}", callback_data=f"media:model:{slot_id}:video_model")],
        [InlineKeyboardButton(text=f"📷 Фото-модель: {pmodel[:30]}", callback_data=f"media:model:{slot_id}:photo_model")],
        [InlineKeyboardButton(text=f"🪄 Rmbg-модель: {rmodel[:30]}", callback_data=f"media:model:{slot_id}:rmbg_model")],
        [
            InlineKeyboardButton(text=f"{active_marks['video']} использовать для видео", callback_data=f"media:active:{slot_id}:video"),
        ],
        [
            InlineKeyboardButton(text=f"{active_marks['photo']} использовать для фото", callback_data=f"media:active:{slot_id}:photo"),
        ],
        [
            InlineKeyboardButton(text=f"{active_marks['rmbg']} использовать для убрать-фон", callback_data=f"media:active:{slot_id}:rmbg"),
        ],
        [InlineKeyboardButton(text="🗑 Очистить слот", callback_data=f"media:delete:{slot_id}")],
        [InlineKeyboardButton(text="↩️  Назад к слотам", callback_data="media:menu")],
    ]
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _kb_model_picker(slot_id: str, field: str) -> InlineKeyboardMarkup:
    """Show preset model slugs from KNOWN_MEDIA_PROVIDERS + 'свой slug'."""
    slot = storage.get_media_slot(slot_id)
    provider = (slot.name if slot else "") or "deapi"
    meta = KNOWN_MEDIA_PROVIDERS.get(provider) or KNOWN_MEDIA_PROVIDERS["deapi"]
    # ``field`` is e.g. ``video_model`` → look at ``video_models`` key in meta.
    list_key = field.replace("_model", "_models")
    options: list[tuple[str, str]] = meta.get(list_key, [])  # type: ignore[arg-type]
    rows: list[list[InlineKeyboardButton]] = []
    for slug, hint in options:
        rows.append(
            [
                InlineKeyboardButton(
                    text=f"{slug} — {hint[:40]}",
                    callback_data=f"media:setmodel:{slot_id}:{field}:{slug}",
                )
            ]
        )
    rows.append([InlineKeyboardButton(text="✏️ Свой slug сообщением", callback_data=f"media:askmodel:{slot_id}:{field}")])
    rows.append([InlineKeyboardButton(text="↩️  Назад", callback_data=f"media:slot:{slot_id}")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _media_menu_text() -> str:
    parts = ["<b>🎬 Мозг для видео и фото</b>", ""]
    parts.append(_active_summary_line())
    parts.append("")
    for sid in storage.MEDIA_SLOT_IDS:
        parts.append(_slot_summary_line(sid))
    parts.append("")
    parts.append("<i>Заполни любой слот (URL + API key + минимум одна модель). "
                 "Затем сделай слот активным для нужной задачи кнопкой ▶️.</i>")
    return "\n".join(parts)


# ---- callbacks: media wizard --------------------------------------------


@media_router.callback_query(F.data == "media:menu")
async def cb_media_menu(query: CallbackQuery, state: FSMContext) -> None:
    if not _is_owner(query.from_user.id):
        await query.answer("Только владелец.", show_alert=True)
        return
    await state.clear()
    if query.message is not None:
        await query.message.edit_text(
            _media_menu_text(),
            reply_markup=_kb_media_menu(),
            disable_web_page_preview=True,
        )
    await query.answer()


@media_router.callback_query(F.data == "media:back")
async def cb_media_back(query: CallbackQuery, state: FSMContext) -> None:
    # Lazy import to avoid circular dependency with wizard.py.
    from .wizard import _kb_brain  # type: ignore[attr-defined]

    if not _is_owner(query.from_user.id):
        await query.answer("Только владелец.", show_alert=True)
        return
    await state.clear()
    if query.message is not None:
        await query.message.edit_text("Выбери мозг:", reply_markup=_kb_brain())
    await query.answer()


@media_router.callback_query(F.data.startswith("media:slot:"))
async def cb_media_slot(query: CallbackQuery, state: FSMContext) -> None:
    if not _is_owner(query.from_user.id):
        await query.answer("Только владелец.", show_alert=True)
        return
    slot_id = (query.data or "").split(":")[-1]
    if slot_id not in storage.MEDIA_SLOT_IDS:
        await query.answer("Нет такого слота.", show_alert=True)
        return
    await state.clear()
    slot = storage.get_media_slot(slot_id)
    name = (slot.name if slot else "") or slot_id
    if query.message is not None:
        await query.message.edit_text(
            f"<b>Слот {slot_id} ({_html(name)})</b>\n\n"
            f"{_slot_summary_line(slot_id)}\n\n"
            "<i>Заполни URL, API key и хотя бы одну модель. "
            "После заполнения URL слот сам поймёт, что это deAPI/fal/Replicate.</i>",
            reply_markup=_kb_media_slot(slot_id),
            disable_web_page_preview=True,
        )
    await query.answer()


@media_router.callback_query(F.data.startswith("media:edit:"))
async def cb_media_edit(query: CallbackQuery, state: FSMContext) -> None:
    if not _is_owner(query.from_user.id):
        await query.answer("Только владелец.", show_alert=True)
        return
    parts = (query.data or "").split(":")
    if len(parts) != 4:
        await query.answer("Bad callback.", show_alert=True)
        return
    _, _, slot_id, field = parts
    if slot_id not in storage.MEDIA_SLOT_IDS:
        await query.answer("Нет такого слота.", show_alert=True)
        return
    await state.update_data(slot_id=slot_id, field=field)
    if field == "url":
        await state.set_state(MediaWizardStates.awaiting_url)
        prompt = (
            f"Пришли URL для слота <b>{slot_id}</b> одним сообщением.\n\n"
            "Примеры:\n"
            "• <code>https://api.deapi.ai</code>\n"
            "• <code>https://fal.run</code>\n"
            "• <code>https://api.replicate.com</code>\n\n"
            "<i>Слот автоматически получит имя по URL.</i>"
        )
    elif field == "api_key":
        await state.set_state(MediaWizardStates.awaiting_api_key)
        prompt = (
            f"Пришли API key для слота <b>{slot_id}</b> одним сообщением. "
            "Я удалю твоё сообщение сразу после сохранения."
        )
    else:
        await query.answer("Bad field.", show_alert=True)
        return
    if query.message is not None:
        await query.message.answer(prompt, disable_web_page_preview=True)
    await query.answer()


@media_router.callback_query(F.data.startswith("media:model:"))
async def cb_media_model(query: CallbackQuery, state: FSMContext) -> None:
    """Show the model picker for video_model / photo_model / rmbg_model."""
    if not _is_owner(query.from_user.id):
        await query.answer("Только владелец.", show_alert=True)
        return
    parts = (query.data or "").split(":")
    if len(parts) != 4:
        await query.answer("Bad callback.", show_alert=True)
        return
    _, _, slot_id, field = parts
    if slot_id not in storage.MEDIA_SLOT_IDS or field not in (
        "video_model", "photo_model", "rmbg_model",
    ):
        await query.answer("Bad slot/field.", show_alert=True)
        return
    await state.clear()
    slot = storage.get_media_slot(slot_id)
    provider = (slot.name if slot else "") or "deapi"
    meta = KNOWN_MEDIA_PROVIDERS.get(provider) or KNOWN_MEDIA_PROVIDERS["deapi"]
    list_key = field.replace("_model", "_models")
    options = meta.get(list_key, [])
    field_label = {
        "video_model": "🎬 видео-модели",
        "photo_model": "📷 фото-модели (img2img)",
        "rmbg_model": "🪄 модели убрать-фон",
    }[field]
    text_lines = [
        f"<b>Слот {slot_id} ({_html(provider)}): выбор {field_label}</b>",
        "",
        meta.get("hint", ""),
        "",
        f"Подходящие модели для <b>{provider}</b>:" if options else
        f"Для провайдера <b>{provider}</b> у меня нет каталога — кинь slug сообщением.",
    ]
    for slug, hint in options:
        text_lines.append(f"• <code>{_html(slug)}</code> — {_html(hint)}")
    if query.message is not None:
        await query.message.edit_text(
            "\n".join(text_lines),
            reply_markup=_kb_model_picker(slot_id, field),
            disable_web_page_preview=True,
        )
    await query.answer()


@media_router.callback_query(F.data.startswith("media:setmodel:"))
async def cb_media_setmodel(query: CallbackQuery, state: FSMContext) -> None:
    if not _is_owner(query.from_user.id):
        await query.answer("Только владелец.", show_alert=True)
        return
    # callback_data shape: media:setmodel:<slot>:<field>:<slug>
    parts = (query.data or "").split(":", 4)
    if len(parts) != 5:
        await query.answer("Bad callback.", show_alert=True)
        return
    _, _, slot_id, field, slug = parts
    if slot_id not in storage.MEDIA_SLOT_IDS or field not in (
        "video_model", "photo_model", "rmbg_model",
    ):
        await query.answer("Bad slot/field.", show_alert=True)
        return
    storage.set_media_slot_field(slot_id, field, slug)
    if query.message is not None:
        await query.message.edit_text(
            f"<b>Слот {slot_id}</b> — {field} = <code>{_html(slug)}</code>\n\n"
            + _slot_summary_line(slot_id),
            reply_markup=_kb_media_slot(slot_id),
            disable_web_page_preview=True,
        )
    await query.answer("Сохранено")


@media_router.callback_query(F.data.startswith("media:askmodel:"))
async def cb_media_askmodel(query: CallbackQuery, state: FSMContext) -> None:
    if not _is_owner(query.from_user.id):
        await query.answer("Только владелец.", show_alert=True)
        return
    parts = (query.data or "").split(":")
    if len(parts) != 4:
        await query.answer("Bad callback.", show_alert=True)
        return
    _, _, slot_id, field = parts
    if slot_id not in storage.MEDIA_SLOT_IDS or field not in (
        "video_model", "photo_model", "rmbg_model",
    ):
        await query.answer("Bad slot/field.", show_alert=True)
        return
    await state.set_state(MediaWizardStates.awaiting_model_value)
    await state.update_data(slot_id=slot_id, field=field)
    field_label = {
        "video_model": "🎬 видео-модели",
        "photo_model": "📷 фото-модели",
        "rmbg_model": "🪄 модели убрать-фон",
    }[field]
    if query.message is not None:
        await query.message.answer(
            f"Пришли slug {field_label} для слота <b>{slot_id}</b> одним сообщением.",
            disable_web_page_preview=True,
        )
    await query.answer()


@media_router.callback_query(F.data.startswith("media:active:"))
async def cb_media_active(query: CallbackQuery, state: FSMContext) -> None:
    if not _is_owner(query.from_user.id):
        await query.answer("Только владелец.", show_alert=True)
        return
    parts = (query.data or "").split(":")
    if len(parts) != 4:
        await query.answer("Bad callback.", show_alert=True)
        return
    _, _, slot_id, task = parts
    if slot_id not in storage.MEDIA_SLOT_IDS or task not in storage.MEDIA_TASKS:
        await query.answer("Bad callback.", show_alert=True)
        return
    slot = storage.get_media_slot(slot_id)
    if slot is None or not slot.configured:
        await query.answer("Слот пуст — заполни URL и API key.", show_alert=True)
        return
    storage.set_active_media_slot(task, slot_id)
    if query.message is not None:
        await query.message.edit_text(
            f"<b>Слот {slot_id}</b> назначен активным для <b>{task}</b>.\n\n"
            + _slot_summary_line(slot_id),
            reply_markup=_kb_media_slot(slot_id),
            disable_web_page_preview=True,
        )
    await query.answer("Активирован")


@media_router.callback_query(F.data.startswith("media:delete:"))
async def cb_media_delete(query: CallbackQuery, state: FSMContext) -> None:
    if not _is_owner(query.from_user.id):
        await query.answer("Только владелец.", show_alert=True)
        return
    slot_id = (query.data or "").split(":")[-1]
    if slot_id not in storage.MEDIA_SLOT_IDS:
        await query.answer("Bad slot.", show_alert=True)
        return
    storage.delete_media_slot(slot_id)
    if query.message is not None:
        await query.message.edit_text(
            f"Слот {slot_id} очищен.\n\n" + _media_menu_text(),
            reply_markup=_kb_media_menu(),
            disable_web_page_preview=True,
        )
    await query.answer("Удалён")


# ---- text capture for media wizard --------------------------------------

_NOT_A_COMMAND = F.text & ~F.text.startswith("/")


@media_router.message(StateFilter(MediaWizardStates.awaiting_url), _NOT_A_COMMAND)
async def capture_media_url(message: Message, state: FSMContext) -> None:
    if not _is_owner(message.from_user.id if message.from_user else None):
        return
    url = (message.text or "").strip()
    if not url.startswith(("http://", "https://")):
        await message.answer(
            "URL должен начинаться с <code>http://</code> или <code>https://</code>. Попробуй ещё раз."
        )
        return
    data = await state.get_data()
    slot_id = str(data.get("slot_id") or "")
    if slot_id not in storage.MEDIA_SLOT_IDS:
        await message.answer("Потерял контекст слота — открой /setup → 🎬 Мозг для видео и фото заново.")
        await state.clear()
        return
    storage.set_media_slot_field(slot_id, "url", url.rstrip("/"))
    await state.clear()
    await message.answer(
        f"URL сохранён для слота {slot_id}.\n\n" + _slot_summary_line(slot_id),
        reply_markup=_kb_media_slot(slot_id),
        disable_web_page_preview=True,
    )


@media_router.message(StateFilter(MediaWizardStates.awaiting_api_key), _NOT_A_COMMAND)
async def capture_media_api_key(message: Message, state: FSMContext) -> None:
    if not _is_owner(message.from_user.id if message.from_user else None):
        return
    key = (message.text or "").strip()
    if not key:
        await message.answer("Пустое сообщение, попробуй ещё раз.")
        return
    data = await state.get_data()
    slot_id = str(data.get("slot_id") or "")
    if slot_id not in storage.MEDIA_SLOT_IDS:
        await message.answer("Потерял контекст слота — открой /setup → 🎬 Мозг для видео и фото заново.")
        await state.clear()
        return
    storage.set_media_slot_field(slot_id, "api_key", key)
    with contextlib.suppress(Exception):
        await message.delete()
    await state.clear()
    await message.answer(
        f"API key сохранён для слота {slot_id}, твоё сообщение удалено.\n\n"
        + _slot_summary_line(slot_id),
        reply_markup=_kb_media_slot(slot_id),
        disable_web_page_preview=True,
    )


@media_router.message(StateFilter(MediaWizardStates.awaiting_model_value), _NOT_A_COMMAND)
async def capture_media_model(message: Message, state: FSMContext) -> None:
    if not _is_owner(message.from_user.id if message.from_user else None):
        return
    slug = (message.text or "").strip()
    if not slug:
        await message.answer("Пустое сообщение, попробуй ещё раз.")
        return
    data = await state.get_data()
    slot_id = str(data.get("slot_id") or "")
    field = str(data.get("field") or "")
    if slot_id not in storage.MEDIA_SLOT_IDS or field not in (
        "video_model", "photo_model", "rmbg_model",
    ):
        await message.answer("Потерял контекст — открой /setup → 🎬 Мозг для видео и фото заново.")
        await state.clear()
        return
    storage.set_media_slot_field(slot_id, field, slug)
    await state.clear()
    await message.answer(
        f"{field} = <code>{_html(slug)}</code> сохранено для слота {slot_id}.\n\n"
        + _slot_summary_line(slot_id),
        reply_markup=_kb_media_slot(slot_id),
        disable_web_page_preview=True,
    )


# ---- photo flow entry ---------------------------------------------------


def _kb_photo_kind() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="📷 Фото", callback_data="photo:kind:photo"),
                InlineKeyboardButton(text="🎬 Видео", callback_data="photo:kind:video"),
            ],
            [InlineKeyboardButton(text="❌ Назад", callback_data="photo:cancel")],
        ]
    )


def _kb_photo_action() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="✨ Указ (LLM-помощник)", callback_data="photo:photo:idea")],
            [InlineKeyboardButton(text="🪄 Убрать фон", callback_data="photo:photo:rmbg")],
            [InlineKeyboardButton(text="↩️ Вернуться", callback_data="photo:back:kind")],
        ]
    )


def _kb_variants(branch: str, count: int) -> InlineKeyboardMarkup:
    """3-variant picker. ``branch`` is ``"photo"`` or ``"video"``.

    Variant texts are NOT in callback_data (Telegram 64-byte limit) —
    instead the picked index goes back into the handler which reads
    the cached variants from FSM data.
    """
    numbers = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣"]
    rows: list[list[InlineKeyboardButton]] = [
        [
            InlineKeyboardButton(
                text=f"{numbers[i]} Вариант {i + 1}",
                callback_data=f"photo:{branch}:var:{i}",
            )
        ]
        for i in range(min(count, len(numbers)))
    ]
    rows.append(
        [
            InlineKeyboardButton(text="🔁 Ещё варианты", callback_data=f"photo:{branch}:idea"),
            InlineKeyboardButton(text="↩️ Вернуться", callback_data=f"photo:back:{branch}_root"),
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _kb_video_prompt() -> InlineKeyboardMarkup:
    """Pre-quality screen for the video branch: prompt entry options."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="✨ Указ (LLM-помощник)", callback_data="photo:video:idea")],
            [InlineKeyboardButton(text="→ Пропустить (caption/дефолт)", callback_data="photo:video:skip_prompt")],
            [InlineKeyboardButton(text="↩️ Вернуться", callback_data="photo:back:kind")],
        ]
    )


def _kb_video_quality() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="480", callback_data="photo:video:q:480"),
                InlineKeyboardButton(text="720", callback_data="photo:video:q:720"),
                InlineKeyboardButton(text="1080", callback_data="photo:video:q:1080"),
            ],
            [
                InlineKeyboardButton(text="2К", callback_data="photo:video:q:2k"),
                InlineKeyboardButton(text="Авто", callback_data="photo:video:q:auto"),
            ],
            [InlineKeyboardButton(text="🎞 GIF", callback_data="photo:video:q:gif")],
            [InlineKeyboardButton(text="↩️ Вернуться", callback_data="photo:back:video_root")],
        ]
    )


# NOTE: the auto-``F.photo`` entry point used to live here. It now sits
# in :mod:`bot.addons.photo_chooser` which asks the user where to route
# the photo (Helpzavr vs generation flow) before anything reacts. When
# the user picks "🎬 Генерация фото и видео" the chooser stashes the
# ``photo_file_id`` / ``photo_chat_id`` / ``caption_prompt`` FSM data
# this module's callbacks already expect, then re-shows ``_kb_photo_kind``.
# So this file's photo flow is unchanged — it just no longer competes
# with Helpzavr for an incoming photo.


@media_router.callback_query(F.data == "photo:cancel")
async def cb_photo_cancel(query: CallbackQuery, state: FSMContext) -> None:
    if not _is_owner(query.from_user.id):
        await query.answer("Только владелец.", show_alert=True)
        return
    await state.clear()
    if query.message is not None:
        with contextlib.suppress(Exception):
            await query.message.delete()
    await query.answer("Отменено")


@media_router.callback_query(F.data == "photo:back:kind")
async def cb_photo_back_kind(query: CallbackQuery, state: FSMContext) -> None:
    if not _is_owner(query.from_user.id):
        await query.answer("Только владелец.", show_alert=True)
        return
    if query.message is not None:
        await query.message.edit_text(
            "Что делаем с фото?",
            reply_markup=_kb_photo_kind(),
        )
    await query.answer()


@media_router.callback_query(F.data == "photo:back:photo_root")
async def cb_photo_back_photo_root(query: CallbackQuery, state: FSMContext) -> None:
    """Return to the photo-branch root (re-arm prompt capture)."""
    if not _is_owner(query.from_user.id):
        await query.answer("Только владелец.", show_alert=True)
        return
    await state.set_state(PhotoFlowStates.awaiting_photo_prompt)
    if query.message is not None:
        await query.message.edit_text(
            "📷 <b>Фото</b>: введи промпт сообщением, нажми «✨ Указ» для LLM-подсказки или «🪄 Убрать фон» ниже.",
            reply_markup=_kb_photo_action(),
        )
    await query.answer()


@media_router.callback_query(F.data == "photo:back:video_root")
async def cb_photo_back_video_root(query: CallbackQuery, state: FSMContext) -> None:
    """Return to the video-branch root (re-arm prompt entry)."""
    if not _is_owner(query.from_user.id):
        await query.answer("Только владелец.", show_alert=True)
        return
    await state.set_state(PhotoFlowStates.awaiting_video_prompt)
    if query.message is not None:
        await query.message.edit_text(
            "🎬 <b>Видео</b>: напиши промпт для движения, жми «✨ Указ» или «Пропустить».",
            reply_markup=_kb_video_prompt(),
        )
    await query.answer()


@media_router.callback_query(F.data == "photo:kind:photo")
async def cb_photo_kind_photo(query: CallbackQuery, state: FSMContext) -> None:
    if not _is_owner(query.from_user.id):
        await query.answer("Только владелец.", show_alert=True)
        return
    # Activate the prompt-capture state — text outside this state still
    # reaches the LLM chat handler in the main router.
    await state.set_state(PhotoFlowStates.awaiting_photo_prompt)
    if query.message is not None:
        await query.message.edit_text(
            "📷 <b>Фото</b>: введи промпт сообщением — что сделать с картинкой, "
            "жми «✨ Указ» чтобы LLM выдал 3 варианта, "
            "или «🪄 Убрать фон» — верну PNG без фона.\n\n"
            "<i>Результат придёт и сжатым превью, и исходным файлом (без сжатия).</i>",
            reply_markup=_kb_photo_action(),
        )
    await query.answer()


@media_router.callback_query(F.data == "photo:kind:video")
async def cb_photo_kind_video(query: CallbackQuery, state: FSMContext) -> None:
    if not _is_owner(query.from_user.id):
        await query.answer("Только владелец.", show_alert=True)
        return
    # Pre-quality screen: gather/refine the motion prompt first.
    await state.set_state(PhotoFlowStates.awaiting_video_prompt)
    if query.message is not None:
        await query.message.edit_text(
            "🎬 <b>Видео</b>: напиши промпт для движения сообщением (описание ракурса/эмоции/движения), "
            "жми «✨ Указ» чтобы LLM выдал 3 варианта, или «Пропустить» — "
            "использую caption как промпт (или дефолт если пустой).\n\n"
            "<i>После промпта — выбор качества (включая GIF).</i>",
            reply_markup=_kb_video_prompt(),
        )
    await query.answer()


@media_router.message(
    StateFilter(PhotoFlowStates.awaiting_photo_prompt), _NOT_A_COMMAND,
)
async def capture_photo_prompt(message: Message, state: FSMContext) -> None:
    """Capture the img2img prompt and run img2img immediately (dual delivery)."""
    if not _is_owner(message.from_user.id if message.from_user else None):
        return
    data = await state.get_data()
    if not data.get("photo_file_id"):
        await message.answer("Контекст фото потерян — пришли фото заново.")
        await state.clear()
        return
    prompt_ru = (message.text or "").strip()
    if not prompt_ru:
        return
    await state.set_state(None)
    # User typed prompt directly — assume it's already in the language
    # they want. We still pass it through translate_to_english so the
    # diffusion model gets natural English (it tolerates Russian poorly).
    await _run_img2img_flow(message, state, prompt_ru=prompt_ru)


@media_router.callback_query(F.data == "photo:photo:rmbg")
async def cb_photo_rmbg(query: CallbackQuery, state: FSMContext) -> None:
    if not _is_owner(query.from_user.id):
        await query.answer("Только владелец.", show_alert=True)
        return
    data = await state.get_data()
    file_id = data.get("photo_file_id")
    if not file_id or query.message is None or query.bot is None:
        await query.answer("Контекст потерян — пришли фото заново.", show_alert=True)
        return
    await query.answer("Запускаю…")
    msg = query.message
    progress_msg = await _open_progress(msg, "📥 Качаю фото…")

    try:
        image_bytes = await _download_photo(query.bot, str(file_id))
    except Exception as exc:  # noqa: BLE001
        logger.exception("photo download failed (rmbg)")
        await progress_msg.edit_text(f"Ошибка скачивания: {_html(str(exc))}")
        return

    async def on_progress(status: str, pct: float) -> None:
        with contextlib.suppress(Exception):
            await progress_msg.edit_text(f"🪄 rmbg: {status} {pct:.0f}%")

    try:
        png_url, jpg_url = await remove_background(image_bytes, on_progress=on_progress)
    except MediaError as exc:
        await progress_msg.edit_text(f"deAPI: {_html(str(exc))}")
        return
    except Exception as exc:  # noqa: BLE001
        logger.exception("rmbg failed")
        await progress_msg.edit_text(f"Ошибка: {_html(str(exc))}")
        return

    try:
        png_bytes = await _fetch_url(png_url)
        jpg_bytes = await _fetch_url(jpg_url) if jpg_url else None
    except Exception as exc:  # noqa: BLE001
        logger.exception("rmbg result download failed")
        await progress_msg.edit_text(f"deAPI вернул URL, но не скачался: {_html(str(exc))}\n{png_url}")
        return

    with contextlib.suppress(Exception):
        await progress_msg.delete()

    # Dual delivery: inline JPG preview (cheap to view) + PNG document
    # with alpha channel preserved (for Photoshop).
    caption = "🪄 Без фона · PNG с альфой во вложении (готово для Photoshop)"
    if jpg_bytes:
        with contextlib.suppress(Exception):
            await msg.answer_photo(
                BufferedInputFile(jpg_bytes, filename="no-bg-preview.jpg"),
                caption=caption,
            )
    await msg.answer_document(
        BufferedInputFile(png_bytes, filename="no-bg.png"),
        caption=None if jpg_bytes else caption,
    )
    await state.clear()


# ---- LLM prompt-helper ("Указ") -----------------------------------------


async def _start_idea_flow(
    query: CallbackQuery, state: FSMContext, *, branch: str
) -> None:
    """Common entry for ✨ Указ in either photo or video branch."""
    if branch == "photo":
        await state.set_state(PhotoFlowStates.awaiting_photo_keywords)
    else:
        await state.set_state(PhotoFlowStates.awaiting_video_keywords)
    if query.message is not None:
        await query.message.edit_text(
            "✨ <b>Указ</b>: пришли ключевые слова или короткое описание идеи "
            f"({'что хочешь увидеть на финальной картинке' if branch == 'photo' else 'какое движение / эмоция / ракурс'}).\n\n"
            "<i>LLM сгенерирует 3 варианта развёрнутого промпта на русском, "
            "ты выберешь — я переведу на английский и отправлю в модель.</i>",
            reply_markup=InlineKeyboardMarkup(
                inline_keyboard=[[
                    InlineKeyboardButton(
                        text="↩️ Вернуться",
                        callback_data=f"photo:back:{branch}_root",
                    ),
                ]],
            ),
        )
    await query.answer()


@media_router.callback_query(F.data == "photo:photo:idea")
async def cb_photo_idea(query: CallbackQuery, state: FSMContext) -> None:
    if not _is_owner(query.from_user.id):
        await query.answer("Только владелец.", show_alert=True)
        return
    await _start_idea_flow(query, state, branch="photo")


@media_router.callback_query(F.data == "photo:video:idea")
async def cb_video_idea(query: CallbackQuery, state: FSMContext) -> None:
    if not _is_owner(query.from_user.id):
        await query.answer("Только владелец.", show_alert=True)
        return
    await _start_idea_flow(query, state, branch="video")


async def _handle_keywords(
    message: Message, state: FSMContext, *, branch: str
) -> None:
    """Generate 3 RU variants from user's keywords and present picker."""
    keywords = (message.text or "").strip()
    if not keywords:
        return
    data = await state.get_data()
    if not data.get("photo_file_id"):
        await message.answer("Контекст фото потерян — пришли фото заново.")
        await state.clear()
        return
    progress_msg = await _open_progress(message, "✨ LLM думает…")
    try:
        variants = await generate_prompt_variants(keywords, kind=branch, count=3)
    except NoLLMError as exc:
        await progress_msg.edit_text(
            f"⚠️ {_html(str(exc))}\n\n"
            "Использую твой текст как промпт без обработки.",
        )
        # No variants — treat keywords as a direct prompt.
        if branch == "photo":
            await state.set_state(None)
            await _run_img2img_flow(message, state, prompt_ru=keywords)
        else:
            await state.update_data(prompt=keywords, prompt_translated=False)
            await state.set_state(None)
            await message.answer(
                f"Промпт: <code>{_html(keywords)}</code>\n\nВыбери качество:",
                reply_markup=_kb_video_quality(),
            )
        return
    except Exception as exc:  # noqa: BLE001
        logger.exception("LLM variant generation failed")
        await progress_msg.edit_text(f"LLM ошибка: {_html(str(exc))}")
        return

    await state.update_data(variants=variants, variants_branch=branch)
    # Variant text is shown in the message body (callback data only has
    # the index — 64-byte limit makes inline labels impractical).
    body_lines = [f"✨ <b>Варианты промпта</b> для «{_html(keywords)[:200]}»:\n"]
    for i, v in enumerate(variants):
        body_lines.append(f"<b>{i + 1}.</b> {_html(v)}")
    with contextlib.suppress(Exception):
        await progress_msg.delete()
    await message.answer(
        "\n\n".join(body_lines),
        reply_markup=_kb_variants(branch, len(variants)),
    )
    # State is cleared because variant selection is callback-driven.
    await state.set_state(None)


@media_router.message(
    StateFilter(PhotoFlowStates.awaiting_photo_keywords), _NOT_A_COMMAND,
)
async def capture_photo_keywords(message: Message, state: FSMContext) -> None:
    if not _is_owner(message.from_user.id if message.from_user else None):
        return
    await _handle_keywords(message, state, branch="photo")


@media_router.message(
    StateFilter(PhotoFlowStates.awaiting_video_keywords), _NOT_A_COMMAND,
)
async def capture_video_keywords(message: Message, state: FSMContext) -> None:
    if not _is_owner(message.from_user.id if message.from_user else None):
        return
    await _handle_keywords(message, state, branch="video")


@media_router.callback_query(F.data.regexp(r"^photo:(photo|video):var:\d+$"))
async def cb_variant_pick(query: CallbackQuery, state: FSMContext) -> None:
    if not _is_owner(query.from_user.id):
        await query.answer("Только владелец.", show_alert=True)
        return
    parts = (query.data or "").split(":")
    branch = parts[1]
    try:
        idx = int(parts[3])
    except (ValueError, IndexError):
        await query.answer("Bad variant.", show_alert=True)
        return
    data = await state.get_data()
    variants = data.get("variants") or []
    if not (0 <= idx < len(variants)):
        await query.answer("Вариант не найден.", show_alert=True)
        return
    chosen_ru = variants[idx]
    await query.answer("Перевожу промпт на EN…")
    if query.message is None:
        return
    msg = query.message
    if branch == "photo":
        await _run_img2img_flow(msg, state, prompt_ru=chosen_ru)
    else:
        # Video: translate now, then show quality picker.
        try:
            prompt_en = await translate_to_english(chosen_ru)
        except Exception as exc:  # noqa: BLE001
            logger.exception("translate failed")
            prompt_en = chosen_ru
            await msg.answer(f"⚠️ Перевод не удался ({_html(str(exc))}) — отправлю русский текст как есть.")
        await state.update_data(prompt=prompt_en, prompt_ru=chosen_ru, prompt_translated=True)
        await state.set_state(None)
        await msg.answer(
            f"✨ Промпт: <code>{_html(chosen_ru)}</code>\n"
            f"EN: <code>{_html(prompt_en)}</code>\n\nВыбери качество:",
            reply_markup=_kb_video_quality(),
        )


# ---- video pre-quality screen handlers ----------------------------------


@media_router.callback_query(F.data == "photo:video:skip_prompt")
async def cb_video_skip_prompt(query: CallbackQuery, state: FSMContext) -> None:
    """Skip prompt entry — use caption or default."""
    if not _is_owner(query.from_user.id):
        await query.answer("Только владелец.", show_alert=True)
        return
    data = await state.get_data()
    caption_prompt = (data.get("caption_prompt") or "").strip()
    prompt = caption_prompt or "Cinematic gentle motion, smooth camera move, soft natural light"
    await state.update_data(prompt=prompt, prompt_translated=False)
    await state.set_state(None)
    if query.message is not None:
        await query.message.edit_text(
            f"🎬 <b>Видео</b>: промпт = <code>{_html(prompt)[:300]}</code>\n\nВыбери качество:",
            reply_markup=_kb_video_quality(),
        )
    await query.answer()


@media_router.message(
    StateFilter(PhotoFlowStates.awaiting_video_prompt), _NOT_A_COMMAND,
)
async def capture_video_prompt(message: Message, state: FSMContext) -> None:
    """User typed a free-form motion prompt — translate, show quality picker."""
    if not _is_owner(message.from_user.id if message.from_user else None):
        return
    data = await state.get_data()
    if not data.get("photo_file_id"):
        await message.answer("Контекст фото потерян — пришли фото заново.")
        await state.clear()
        return
    prompt_ru = (message.text or "").strip()
    if not prompt_ru:
        return
    progress_msg = await _open_progress(message, "✨ Перевожу промпт на EN…")
    prompt_en = await translate_to_english(prompt_ru)
    await state.update_data(prompt=prompt_en, prompt_ru=prompt_ru, prompt_translated=True)
    await state.set_state(None)
    with contextlib.suppress(Exception):
        await progress_msg.delete()
    await message.answer(
        f"🎬 Промпт принят:\n<code>{_html(prompt_ru)}</code>\n"
        f"EN: <code>{_html(prompt_en)}</code>\n\nВыбери качество:",
        reply_markup=_kb_video_quality(),
    )


# ---- video quality (incl. GIF) ------------------------------------------


@media_router.callback_query(F.data.startswith("photo:video:q:"))
async def cb_video_quality(query: CallbackQuery, state: FSMContext) -> None:
    if not _is_owner(query.from_user.id):
        await query.answer("Только владелец.", show_alert=True)
        return
    quality = (query.data or "").split(":")[-1].lower()
    if quality not in VIDEO_QUALITY_MAP:
        await query.answer("Bad quality.", show_alert=True)
        return
    data = await state.get_data()
    file_id = data.get("photo_file_id")
    if not file_id or query.message is None or query.bot is None:
        await query.answer("Контекст потерян — пришли фото заново.", show_alert=True)
        return
    prompt = (data.get("prompt") or "").strip() or "Cinematic gentle motion, smooth camera move, soft natural light"
    await query.answer("Запускаю…")
    msg = query.message
    await _run_video_flow(msg, state, file_id=str(file_id), quality=quality, prompt=prompt, bot=query.bot)


# ---- workhorses: img2img / video flow + dual delivery -------------------


async def _run_img2img_flow(
    msg: Message, state: FSMContext, *, prompt_ru: str
) -> None:
    """Translate prompt → call img2img → dual delivery (inline + document).

    ``msg`` is the message the user replied with (used for ``answer_*``);
    ``state`` carries ``photo_file_id``.
    """
    data = await state.get_data()
    file_id = data.get("photo_file_id")
    if not file_id or msg.bot is None:
        await msg.answer("Контекст потерян — пришли фото заново.")
        return
    progress_msg = await _open_progress(msg, "✨ Перевожу промпт на EN…")
    prompt_en = await translate_to_english(prompt_ru)
    with contextlib.suppress(Exception):
        await progress_msg.edit_text(f"📥 Качаю фото…\n<i>EN: {_html(prompt_en)[:200]}</i>")
    try:
        image_bytes = await _download_photo(msg.bot, str(file_id))
    except Exception as exc:  # noqa: BLE001
        logger.exception("photo download failed (img2img)")
        await progress_msg.edit_text(f"Ошибка скачивания: {_html(str(exc))}")
        return

    async def on_progress(status: str, pct: float) -> None:
        with contextlib.suppress(Exception):
            await progress_msg.edit_text(f"📷 img2img: {status} {pct:.0f}%")

    # We always request PNG to get an uncompressed source. deAPI returns
    # both PNG and JPG alt-format URLs by default, so we get inline-
    # preview material for free.
    try:
        png_url, jpg_url = await image_to_image(
            image_bytes, prompt_en, fmt="png", on_progress=on_progress,
        )
    except MediaError as exc:
        await progress_msg.edit_text(f"deAPI: {_html(str(exc))}")
        return
    except Exception as exc:  # noqa: BLE001
        logger.exception("img2img failed")
        await progress_msg.edit_text(f"Ошибка: {_html(str(exc))}")
        return

    try:
        png_bytes = await _fetch_url(png_url)
        jpg_bytes = await _fetch_url(jpg_url) if jpg_url else None
    except Exception as exc:  # noqa: BLE001
        logger.exception("img2img download failed")
        await progress_msg.edit_text(f"deAPI вернул URL, но не скачался: {_html(str(exc))}")
        return

    with contextlib.suppress(Exception):
        await progress_msg.delete()

    caption = (
        f"📷 img2img · <code>{_html(prompt_ru)[:200]}</code>\n"
        f"<i>EN: {_html(prompt_en)[:200]}</i>"
    )
    if jpg_bytes:
        with contextlib.suppress(Exception):
            await msg.answer_photo(
                BufferedInputFile(jpg_bytes, filename="img2img-preview.jpg"),
                caption=caption,
            )
    await msg.answer_document(
        BufferedInputFile(png_bytes, filename="img2img.png"),
        caption=None if jpg_bytes else caption,
    )
    await state.clear()


async def _run_video_flow(
    msg: Message,
    state: FSMContext,
    *,
    file_id: str,
    quality: str,
    prompt: str,
    bot,
) -> None:
    """Run img2video, deliver inline preview + uncompressed MP4 document.

    For ``quality == "gif"`` we still ask deAPI for a small mp4 (smallest
    quality preset) and additionally try to convert it to a real .gif
    via ffmpeg. The mp4 is always sent as the source file.
    """
    is_gif = quality == "gif"
    deapi_quality = "480" if is_gif else quality  # smallest preset for gif
    progress_msg = await _open_progress(
        msg, f"📥 Качаю фото, качество: {quality}…"
    )
    try:
        image_bytes = await _download_photo(bot, file_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("photo download failed (i2v)")
        await progress_msg.edit_text(f"Ошибка скачивания: {_html(str(exc))}")
        return

    async def on_progress(status: str, pct: float) -> None:
        with contextlib.suppress(Exception):
            await progress_msg.edit_text(
                f"🎬 img2video [{quality}]: {status} {pct:.0f}%"
            )

    try:
        video_url = await image_to_video(
            image_bytes, prompt, quality=deapi_quality, on_progress=on_progress,
        )
    except MediaError as exc:
        await progress_msg.edit_text(f"deAPI: {_html(str(exc))}")
        return
    except Exception as exc:  # noqa: BLE001
        logger.exception("img2video failed")
        await progress_msg.edit_text(f"Ошибка: {_html(str(exc))}")
        return

    try:
        mp4_bytes = await _fetch_url(video_url)
    except Exception as exc:  # noqa: BLE001
        logger.exception("video download failed")
        await progress_msg.edit_text(f"deAPI вернул URL, но не скачался: {_html(str(exc))}\n{video_url}")
        return

    with contextlib.suppress(Exception):
        await progress_msg.delete()

    caption = f"🎬 {quality} · <code>{_html(prompt)[:600]}</code>"
    if is_gif:
        # Try real-gif conversion via ffmpeg; fall back to mp4-as-animation.
        gif_bytes = await _convert_mp4_to_gif(mp4_bytes)
        if gif_bytes:
            with contextlib.suppress(Exception):
                await msg.answer_animation(
                    BufferedInputFile(gif_bytes, filename="loop.gif"),
                    caption=caption,
                )
            await msg.answer_document(
                BufferedInputFile(gif_bytes, filename="loop.gif"),
            )
        else:
            # ffmpeg not available — send mp4 as animation (TG renders it
            # like a GIF) and also as the source file.
            with contextlib.suppress(Exception):
                await msg.answer_animation(
                    BufferedInputFile(mp4_bytes, filename="loop.mp4"),
                    caption=caption + "\n<i>ffmpeg недоступен; отдаю mp4-as-animation</i>",
                )
            await msg.answer_document(BufferedInputFile(mp4_bytes, filename="loop.mp4"))
    else:
        # Inline preview + uncompressed MP4 document.
        with contextlib.suppress(Exception):
            await msg.answer_video(
                BufferedInputFile(mp4_bytes, filename="video.mp4"),
                caption=caption,
            )
        await msg.answer_document(
            BufferedInputFile(mp4_bytes, filename="video.mp4"),
        )
    await state.clear()


async def _convert_mp4_to_gif(mp4_bytes: bytes) -> bytes | None:
    """Convert mp4 bytes to gif via ffmpeg subprocess. Returns ``None``
    if ffmpeg is missing or the conversion fails.

    Uses palettegen+paletteuse for good color, capped at 12 fps / 480px
    width to keep the gif size reasonable.
    """
    import asyncio
    import shutil
    import tempfile
    from pathlib import Path as _P

    if not shutil.which("ffmpeg"):
        return None
    with tempfile.TemporaryDirectory() as tmp:
        mp4 = _P(tmp) / "in.mp4"
        gif = _P(tmp) / "out.gif"
        mp4.write_bytes(mp4_bytes)
        # Two-pass via filter_complex inside a single ffmpeg invocation.
        cmd = [
            "ffmpeg", "-y", "-i", str(mp4),
            "-vf",
            "fps=12,scale=480:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
            "-loop", "0",
            str(gif),
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            logger.warning("ffmpeg mp4→gif failed: %s", stderr.decode(errors="replace")[:300])
            return None
        return gif.read_bytes()


# ---- private helpers ----------------------------------------------------


async def _download_photo(bot, file_id: str) -> bytes:
    """Download the largest TG photo behind ``file_id`` as bytes."""
    tg_file = await bot.get_file(file_id)
    buf = await bot.download_file(tg_file.file_path)
    if hasattr(buf, "read"):
        return buf.read()
    if isinstance(buf, (bytes, bytearray)):
        return bytes(buf)
    # aiogram may return an aiohttp BytesIO-like buffer; coerce.
    return bytes(buf or b"")


async def _fetch_url(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.content
