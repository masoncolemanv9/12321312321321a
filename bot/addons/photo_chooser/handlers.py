"""Photo chooser — the single ``F.photo`` entry point.

When a photo arrives the bot answers with an inline keyboard asking
the user where the photo should be routed:

* 🤖 Helpzavr — screenshot annotator / vision Q&A
* 🎬 Генерация фото и видео — img2img / img2video / rmbg flow
* ❌ Закрыть — drop the photo silently

The chosen handler is invoked directly with the cached ``file_id`` so
the photo's bytes don't need to be uploaded twice. State is stashed in
FSM data — Telegram keeps the ``file_id`` alive plenty long for this.

This addon takes over from ``media_ui.on_photo`` and Helpzavr's
``F.photo`` / ``F.document.mime_type`` matchers — both of those have
been disabled in favour of this single chooser. That solves the "they
fight over my photo" complaint while keeping each pipeline reachable
explicitly.
"""

from __future__ import annotations

import contextlib
import io
import logging

from aiogram import Bot, F, Router
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

logger = logging.getLogger(__name__)


class PhotoChooserStates(StatesGroup):
    """Tiny FSM state — only used to remember we are waiting for a click."""

    awaiting_choice = State()


def _kb_chooser() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="🤖 Helpzavr", callback_data="photo_chooser:helpzavr"
                )
            ],
            [
                InlineKeyboardButton(
                    text="🎬 Генерация фото и видео",
                    callback_data="photo_chooser:generation",
                )
            ],
            [
                InlineKeyboardButton(
                    text="❌ Закрыть", callback_data="photo_chooser:close"
                )
            ],
        ]
    )


CHOOSER_TEXT = (
    "Что делаем с этим фото?\n\n"
    "🤖 <b>Helpzavr</b> — разобрать скриншот, подсказать что вписать, "
    "нарисовать стрелку/рамку, ответить на вопрос по картинке.\n"
    "🎬 <b>Генерация фото и видео</b> — img2img / убрать фон / "
    "анимировать в короткий ролик.\n\n"
    "<i>Если ничего не подходит — закрой, фото проигнорируется.</i>"
)


def build_photo_chooser_router() -> Router:
    router = Router(name="photo_chooser_addon")

    # ---- entry: a photo arrived ------------------------------------------

    def _is_owner(user_id: int | None) -> bool:
        """Backward-compat alias for "is allowed to USE the bot".

        Photo upload is a feature, not a settings change, so this is
        gated on :func:`bot.access.can_use` rather than admin rights —
        guests in ``public``/``full_public`` mode see the chooser too.
        """
        # Lazy import to avoid pulling lilush.storage at module import.
        from ...access import can_use

        return can_use(user_id)

    def _photos_are_silenced(chat_id: int) -> bool:
        """Return True when the master "Генерация фото и видео" toggle is
        OFF — the chooser stays silent so casual photos don't trigger a
        prompt at all."""
        try:
            from ..media_toggle.handlers import is_media_enabled

            return not is_media_enabled(chat_id)
        except Exception:  # noqa: BLE001
            return False

    def _helpzavr_enabled(chat_id: int) -> bool:
        """Return True when the user has switched Helpzavr ON via the
        '🔘 Включить режим' button on its own settings screen. Photos
        in that case route directly to the Helpzavr pipeline; the
        chooser stays out of the way."""
        try:
            from ..helpzavr.handlers import is_enabled as _hz_is_enabled

            return bool(_hz_is_enabled(chat_id))
        except Exception:  # noqa: BLE001
            return False

    async def _route_to_helpzavr(
        message: Message,
        state: FSMContext,
        bot: Bot,
        file_id: str,
        caption: str,
        is_document: bool,
    ) -> None:
        """Run the Helpzavr flow directly without showing the chooser
        bubble — used when the user is already in Helpzavr mode.

        Mirrors :func:`on_helpzavr` but skips the chooser-related
        bookkeeping (FSM data, deleting the chooser message).
        """
        try:
            from PIL import Image  # type: ignore[import-not-found]

            from ..helpzavr.handlers import (
                _run_pipeline,
                get_groq_key,
                get_openrouter_key,
                request_prompt_for_photo,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("photo_chooser: helpzavr import failed")
            await message.answer(
                f"Аддон Helpzavr недоступен ({exc}). "
                "Проверь, что pillow + pytesseract установлены."
            )
            return

        if not (get_groq_key() and get_openrouter_key()):
            await message.answer(
                "Сначала задай Groq и OpenRouter ключи в "
                "<b>🤖 Helpzavr</b> на /start."
            )
            return

        # No caption → ask for prompt first, run pipeline on next text.
        if not caption:
            await request_prompt_for_photo(
                message, state, file_id, is_document=is_document
            )
            return

        await state.clear()
        buf = io.BytesIO()
        try:
            await bot.download(file_id, destination=buf)
        except Exception as exc:  # noqa: BLE001
            logger.exception("photo_chooser: helpzavr direct download failed")
            await message.answer(
                f"Не получилось скачать фото из Telegram: {exc}"
            )
            return
        buf.seek(0)
        try:
            img = Image.open(buf)
            img.load()
        except Exception as exc:  # noqa: BLE001
            await message.answer(
                f"Не получилось открыть изображение: {exc}"
            )
            return
        try:
            await _run_pipeline(message, bot, img, caption)
        except Exception as exc:  # noqa: BLE001
            logger.exception("photo_chooser: helpzavr direct pipeline crashed")
            await message.answer(f"Ошибка пайплайна Helpzavr: {exc}")

    @router.message(F.photo)
    async def on_photo(
        message: Message, state: FSMContext, bot: Bot
    ) -> None:
        if not _is_owner(message.from_user.id if message.from_user else None):
            return
        if not message.photo:
            return
        photo = max(message.photo, key=lambda p: p.width * p.height)
        caption = (message.caption or "").strip()

        # If Helpzavr is already turned ON for this chat — bypass the
        # chooser entirely. The user explicitly said "I'm in Helpzavr,
        # don't ask me again".
        if _helpzavr_enabled(message.chat.id):
            await _route_to_helpzavr(
                message,
                state,
                bot,
                photo.file_id,
                caption,
                is_document=False,
            )
            return

        # If even the generation toggle is off, stay silent.
        if _photos_are_silenced(message.chat.id):
            return

        await state.clear()
        await state.update_data(
            chooser_file_id=photo.file_id,
            chooser_caption=caption,
            chooser_is_document=False,
        )
        await state.set_state(PhotoChooserStates.awaiting_choice)
        await message.answer(CHOOSER_TEXT, reply_markup=_kb_chooser())

    @router.message(F.document.mime_type.startswith("image/"))
    async def on_document_image(
        message: Message, state: FSMContext, bot: Bot
    ) -> None:
        if not _is_owner(message.from_user.id if message.from_user else None):
            return
        if message.document is None:
            return
        caption = (message.caption or "").strip()
        file_id = message.document.file_id

        if _helpzavr_enabled(message.chat.id):
            await _route_to_helpzavr(
                message, state, bot, file_id, caption, is_document=True
            )
            return

        if _photos_are_silenced(message.chat.id):
            return

        await state.clear()
        await state.update_data(
            chooser_file_id=file_id,
            chooser_caption=caption,
            chooser_is_document=True,
        )
        await state.set_state(PhotoChooserStates.awaiting_choice)
        await message.answer(CHOOSER_TEXT, reply_markup=_kb_chooser())

    # ---- callbacks --------------------------------------------------------

    @router.callback_query(F.data == "photo_chooser:close")
    async def on_close(query: CallbackQuery, state: FSMContext) -> None:
        await state.clear()
        if query.message is not None:
            with contextlib.suppress(Exception):
                await query.message.delete()
        await query.answer("Закрыто")

    @router.callback_query(F.data == "photo_chooser:helpzavr")
    async def on_helpzavr(
        query: CallbackQuery, state: FSMContext, bot: Bot
    ) -> None:
        data = await state.get_data()
        file_id = data.get("chooser_file_id")
        caption = (data.get("chooser_caption") or "").strip()
        is_document = bool(data.get("chooser_is_document"))
        if not file_id or query.message is None:
            await state.clear()
            await query.answer(
                "Фото уже не в памяти — пришли ещё раз", show_alert=True
            )
            return

        # Lazy imports — Helpzavr's deps (Pillow + OCR) are optional.
        try:
            from PIL import Image  # type: ignore[import-not-found]

            from ..helpzavr.handlers import (
                _run_pipeline,
                get_groq_key,
                get_openrouter_key,
                request_prompt_for_photo,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("photo_chooser: helpzavr import failed")
            await query.message.answer(
                f"Аддон Helpzavr недоступен ({exc}). "
                "Проверь, что pillow + pytesseract установлены."
            )
            await state.clear()
            await query.answer()
            return

        if not (get_groq_key() and get_openrouter_key()):
            await query.message.answer(
                "Сначала задай Groq и OpenRouter ключи в "
                "<b>🤖 Helpzavr</b> на /start."
            )
            await state.clear()
            await query.answer("Нужны ключи Helpzavr")
            return

        # No caption → tg_bot13-style: ask the user to type a prompt
        # and run the pipeline once that next text arrives.
        if not caption:
            with contextlib.suppress(Exception):
                await query.message.delete()
            await request_prompt_for_photo(
                query.message, state, file_id, is_document=is_document
            )
            await query.answer()
            return

        await state.clear()

        # Remove the chooser bubble entirely — from this point on the
        # only status visible should be whatever the user's Соображалка
        # style draws (white bubble / white + model / black centered
        # tile). A static "Helpzavr работает с фото…" label here would
        # double up with that and clutter the chat.
        with contextlib.suppress(Exception):
            await query.message.delete()

        # Download the file bytes (Telegram keeps file_id alive long
        # enough for this single-step roundtrip).
        buf = io.BytesIO()
        try:
            await bot.download(file_id, destination=buf)
        except Exception as exc:  # noqa: BLE001
            logger.exception("photo_chooser: download failed")
            await query.message.answer(
                f"Не получилось скачать фото из Telegram: {exc}"
            )
            await query.answer()
            return
        buf.seek(0)
        try:
            img = Image.open(buf)
            img.load()
        except Exception as exc:  # noqa: BLE001
            await query.message.answer(
                f"Не получилось открыть изображение: {exc}"
            )
            await query.answer()
            return
        try:
            await _run_pipeline(query.message, bot, img, caption)
        except Exception as exc:  # noqa: BLE001
            logger.exception("photo_chooser: helpzavr pipeline crashed")
            await query.message.answer(
                f"Ошибка пайплайна Helpzavr: {exc}"
            )
        await query.answer()

    @router.callback_query(F.data == "photo_chooser:generation")
    async def on_generation(
        query: CallbackQuery, state: FSMContext
    ) -> None:
        data = await state.get_data()
        file_id = data.get("chooser_file_id")
        caption = (data.get("chooser_caption") or "").strip()
        if not file_id or query.message is None:
            await state.clear()
            await query.answer(
                "Фото уже не в памяти — пришли ещё раз", show_alert=True
            )
            return

        # Reset state — about to hand off to media_ui's photo flow which
        # uses its own FSM keys (``photo_file_id`` / ``photo_chat_id`` /
        # ``caption_prompt``). Match those exactly.
        await state.clear()
        await state.update_data(
            photo_file_id=file_id,
            photo_chat_id=query.message.chat.id,
            caption_prompt=caption,
        )

        try:
            from ...media_ui import _kb_photo_kind
        except Exception as exc:  # noqa: BLE001
            logger.exception("photo_chooser: media_ui import failed")
            await query.message.answer(
                f"Генерация недоступна ({exc}). Проверь логи."
            )
            await query.answer()
            return

        try:
            await query.message.edit_text(
                "Что делаем с фото?\n\n"
                "📷 <b>Фото</b> — обработать как изображение "
                "(img2img / убрать фон)\n"
                "🎬 <b>Видео</b> — анимировать в короткий ролик",
                reply_markup=_kb_photo_kind(),
            )
        except Exception:  # noqa: BLE001
            # Fallback: send a fresh message if edit fails (e.g. the
            # chooser bubble already has incompatible content type).
            await query.message.answer(
                "Что делаем с фото?",
                reply_markup=_kb_photo_kind(),
            )
        await query.answer()

    return router
