"""Helpzavr — screenshot-helper addon (from tg_bot+(13).zip).

Flow:
1. User clicks 🤖 Helpzavr on /start menu → bot opens settings/help screen.
2. User clicks "🔘 Включить режим" → bot enters helpzavr mode.
3. User sends a photo (with or without caption) → bot runs the pipeline:
   vision (Groq) → thinking (OpenRouter) → refine (OCR + CV) → annotate
   (red box + arrow + callout).
4. Status message edits in place at each stage — matching tg_bot13's
   exact UX:  «Смотрю на картинку…» → «Анализирую расположение полей…»
   → «Думаю, что туда написать…» → «Уточняю границы поля…» → «Рисую…»

Mode and API keys live in lilush's storage via :mod:`bot.addons.state`,
not in tg_bot13's own ``storage.py`` (which we don't use to avoid the
two-singleton problem).
"""

from __future__ import annotations

import contextlib
import io
import logging
import re

from aiogram import Bot, F, Router
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
from PIL import Image

from .. import state as addon_state

logger = logging.getLogger(__name__)


class HelpzavrStates(StatesGroup):
    """FSM state: photo received without caption, waiting for the user's
    text prompt. This is tg_bot13's ``awaiting_prompt`` flow restored —
    see :func:`request_prompt_for_photo`.
    """

    awaiting_prompt = State()


_AWAITING_PROMPT_TEXT = (
    "Получил картинку. Что с ней сделать?\n\n"
    "Например:\n"
    "• <i>«что написать в email?»</i> — обведу поле и подскажу.\n"
    "• <i>«где Releases?»</i> — покажу где это на скрине.\n"
    "• <i>«опиши»</i> — расскажу что на картинке."
)


async def request_prompt_for_photo(
    message: Message,
    state: FSMContext,
    file_id: str,
    *,
    is_document: bool = False,
) -> None:
    """Tell the user we got the photo but need a prompt next, and park
    the FSM in :attr:`HelpzavrStates.awaiting_prompt`. The next text
    message will be picked up by ``on_prompt_text`` below and used as
    the question for the helpzavr pipeline.

    This is the entry-point invoked by ``photo_chooser`` when the user
    is in Helpzavr-mode AND sent a photo without a caption — the same
    behavior tg_bot13 exposed natively.
    """
    await state.clear()
    await state.update_data(hz_file_id=file_id, hz_is_document=is_document)
    await state.set_state(HelpzavrStates.awaiting_prompt)
    await message.answer(_AWAITING_PROMPT_TEXT)

# Default models that tg_bot13's .env.example ships with. Wizard can
# override later via free-text input, but the defaults work out of the
# box on free tiers of both providers.
DEFAULT_GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
DEFAULT_OPENROUTER_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"

# Heuristic from tg_bot13: question that asks WHERE / WHAT vs FILL.
_LOCATE_PATTERN = re.compile(
    r"\b(где|найди|найти|покажи|опиши|расскажи|что (?:тут|здесь|на)|реши|разбери|объясни|"
    r"where|find|show|describe|what(?:'s| is)|explain|solve|locate)\b",
    re.IGNORECASE | re.UNICODE,
)
_FILL_PATTERN = re.compile(
    r"\b(вписать|написать|ввести|введи|заполни|заполнить|помоги заполнить|"
    r"fill|enter|type)\b",
    re.IGNORECASE | re.UNICODE,
)


# ---- settings ------------------------------------------------------------


def is_enabled(chat_id: int) -> bool:
    return bool(addon_state.chat_get("helpzavr", chat_id, "enabled", False))


def _set_enabled(chat_id: int, value: bool) -> None:
    addon_state.chat_set("helpzavr", chat_id, "enabled", value)


def _looks_like_groq_key(value: str) -> bool:
    """Return True when ``value`` resembles a real Groq API key.

    Real Groq keys look like ``gsk_<base62>`` (50+ chars). Model names
    (e.g. ``meta-llama/llama-4-scout-17b-16e-instruct``) always contain
    a ``/`` and never start with ``gsk_`` — distinguishing the two is
    enough to catch the common "user pasted the model name into the
    key prompt" mistake.
    """
    v = (value or "").strip()
    if not v or "/" in v:
        return False
    return v.startswith("gsk_") and len(v) >= 20


def _looks_like_openrouter_key(value: str) -> bool:
    """Same idea for OpenRouter — keys look like ``sk-or-v1-...``."""
    v = (value or "").strip()
    return v.startswith("sk-or-") and len(v) >= 20


def _brain_slot2_groq_key() -> str:
    """Return the Groq key from lilush's «Brain 2» slot if that slot is
    pointed at ``api.groq.com``, else "". Lets the user configure the
    same Groq key once in either Brain 2 or Helpzavr and have both
    places see it.
    """
    try:
        from ...storage import storage as _lilush_storage
        slot2 = _lilush_storage.get_brain_slot2() or {}
    except Exception:  # noqa: BLE001
        return ""
    base_url = (slot2.get("base_url") or "").lower()
    api_key = (slot2.get("api_key") or "").strip()
    if "groq.com" in base_url and _looks_like_groq_key(api_key):
        return api_key
    return ""


def get_groq_key() -> str:
    """Resolve the Groq API key with sane fallbacks.

    Order:
    1. ``addons.helpzavr.groq_api_key`` if it actually looks like a key.
       A stored value that is obviously a model name (``meta-llama/...``)
       is ignored so a previously-confused state.json doesn't strand the
       user with permanent 401s.
    2. lilush's Brain 2 slot if it points to ``api.groq.com`` — the user
       only needs to configure one Groq key for both Brain 2 chat and
       Helpzavr vision.
    3. ``GROQ_API_KEY`` environment variable.
    """
    import os
    stored = addon_state.get("helpzavr", "groq_api_key", "")
    if stored and _looks_like_groq_key(stored):
        return stored
    via_brain2 = _brain_slot2_groq_key()
    if via_brain2:
        return via_brain2
    return os.environ.get("GROQ_API_KEY", "")


def get_openrouter_key() -> str:
    import os
    stored = addon_state.get("helpzavr", "openrouter_api_key", "")
    if stored and _looks_like_openrouter_key(stored):
        return stored
    # Re-use lilush's OpenRouter key if the user didn't set a Helpzavr one.
    from ...storage import storage as _lilush_storage
    via_lilush = _lilush_storage.get_provider_key("openrouter")
    if via_lilush:
        return via_lilush
    return os.environ.get("OPENROUTER_API_KEY", "")


def _set_groq_key(key: str) -> None:
    addon_state.set_("helpzavr", "groq_api_key", key.strip())


def _set_openrouter_key(key: str) -> None:
    addon_state.set_("helpzavr", "openrouter_api_key", key.strip())


def get_groq_model() -> str:
    return addon_state.get("helpzavr", "groq_model", "") or DEFAULT_GROQ_MODEL


def get_openrouter_model() -> str:
    return (
        addon_state.get("helpzavr", "openrouter_model", "")
        or DEFAULT_OPENROUTER_MODEL
    )


def _mask(key: str) -> str:
    if not key:
        return "не задан"
    if len(key) <= 10:
        return "***"
    return f"{key[:6]}***{key[-4:]}"


# ---- UI ------------------------------------------------------------------


SCREEN_TEXT = (
    "<b>🤖 Helpzavr</b> — помощник по скриншотам\n\n"
    "Что умеет:\n"
    "• <b>Скриншот формы</b> с вопросом «что вписать в email?» — обведу "
    "нужное поле красной рамкой, нарисую стрелку, подскажу что вписать "
    "и приведу пример.\n"
    "• <b>Любая картинка</b> с вопросом «где Releases?», «опиши», "
    "«реши капчу» — отвечу свободным текстом + пометкой на картинке если "
    "уместно.\n\n"
    "Что нужно <b>один раз</b> настроить:\n"
    "1. <b>Groq API</b> (зрение) — бесплатный, без карты. "
    "<a href=\"https://console.groq.com\">console.groq.com</a> → "
    "API Keys → Create.\n"
    "2. <b>OpenRouter API</b> (думалка) — также бесплатный. "
    "<a href=\"https://openrouter.ai\">openrouter.ai</a> → Keys → Create.\n\n"
    "Состояние:\n"
    "• Режим: <b>{state}</b>\n"
    "• Groq key: <code>{groq}</code> | модель: <code>{gm}</code>\n"
    "• OpenRouter key: <code>{oro}</code> | модель: <code>{om}</code>"
)


def _kb_screen(enabled: bool, has_keys: bool, is_admin: bool = True) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    if has_keys:
        toggle = (
            InlineKeyboardButton(
                text="⏸ Выключить режим", callback_data="hz:off"
            )
            if enabled
            else InlineKeyboardButton(
                text="🔘 Включить режим", callback_data="hz:on"
            )
        )
        rows.append([toggle])
    if is_admin:
        rows.append(
            [InlineKeyboardButton(text="🔑 Groq API key", callback_data="hz:setkey:groq")]
        )
        rows.append(
            [
                InlineKeyboardButton(
                    text="🔑 OpenRouter API key", callback_data="hz:setkey:openrouter"
                )
            ]
        )
    rows.append([InlineKeyboardButton(text="↩️ Назад", callback_data="hz:back")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


async def show_screen(target) -> None:
    """Render the helpzavr screen. ``target`` is Message or CallbackQuery."""
    if isinstance(target, CallbackQuery):
        chat_id = target.message.chat.id  # type: ignore[union-attr]
        user_id = target.from_user.id if target.from_user else None
    else:
        chat_id = target.chat.id
        user_id = target.from_user.id if target.from_user else None
    enabled = is_enabled(chat_id)
    groq = get_groq_key()
    oro = get_openrouter_key()
    has_keys = bool(groq) and bool(oro)
    # API-key admin gating: guests in public mode see status but not the
    # key-setter buttons. Bot owner / co-owners / full_public users see
    # the full screen as before.
    from ...access import can_admin

    is_admin = can_admin(user_id)
    if is_admin:
        text = SCREEN_TEXT.format(
            state=("ВКЛ ✅" if enabled else "ВЫКЛ ⛔") if has_keys else "нужны ключи",
            groq=_mask(groq),
            gm=get_groq_model(),
            oro=_mask(oro),
            om=get_openrouter_model(),
        )
    else:
        text = (
            "<b>🤖 Helpzavr</b> — помощник по скриншотам\n\n"
            "Что умеет:\n"
            "• <b>Скриншот формы</b> с вопросом «что вписать в email?» — "
            "обведу нужное поле красной рамкой, нарисую стрелку, "
            "подскажу что вписать и приведу пример.\n"
            "• <b>Любая картинка</b> с вопросом «где Releases?», «опиши», "
            "«реши капчу» — отвечу свободным текстом + пометкой на "
            "картинке если уместно.\n\n"
            f"Состояние: <b>{'ВКЛ ✅' if enabled else 'ВЫКЛ ⛔'}</b>"
            + ("" if has_keys else " <i>(нужны ключи — попроси владельца настроить)</i>")
        )
    kb = _kb_screen(enabled, has_keys, is_admin=is_admin)
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


# ---- pipeline ------------------------------------------------------------


async def _run_pipeline(
    message: Message,
    bot: Bot,
    img: Image.Image,
    question: str,
) -> None:
    """Vision → thinking → refine → annotate, with status routed
    through the user's chosen "Соображалка" style.

    Status updates go through :func:`addons.thinking_style.make_status_runner`
    so the user gets the same look they picked in the settings (white
    bubble / white + model / black-centered tile) regardless of which
    addon is running.
    """
    from ..thinking_style import make_status_runner
    from .annotate import Style, annotate
    from .refine import refine_bbox_async
    from .thinking import decide_what_to_write
    from .vision import analyze_screenshot, answer_about_image

    # In ⚪ Белая + модель the runner appends "· <model>" after every
    # status line. The active model here is OpenRouter's
    # (vision → groq for image OCR, thinking → openrouter for replies).
    # We surface the OpenRouter model since that's the one writing the
    # user-facing answer.
    def _model_hint() -> str:
        return get_openrouter_model()

    status_update, status_finish = make_status_runner(
        message, bot=bot, model_hint=_model_hint
    )
    await status_update("Смотрю на картинку…")

    def _intent(q: str) -> str:
        if not q:
            return "fill"
        fill = bool(_FILL_PATTERN.search(q))
        locate = bool(_LOCATE_PATTERN.search(q))
        return "general" if locate and not fill else "fill"

    intent = _intent(question)
    groq_key = get_groq_key()
    oro_key = get_openrouter_key()
    groq_model = get_groq_model()
    oro_model = get_openrouter_model()
    # Diagnostic INFO log so the bot owner can tell from logs which
    # branch the pipeline took for a given user message (general =
    # text-only Q&A, fill = annotated-image flow). Helps debug "why
    # didn't I get arrows/box back?" complaints.
    logger.info(
        "helpzavr pipeline: intent=%s question=%r image=%dx%d",
        intent,
        (question or "")[:80],
        img.width,
        img.height,
    )

    # general-Q&A path: short flow, no field detection.
    async def _general_answer() -> None:
        await status_update("Смотрю что на картинке…")
        try:
            ans = await answer_about_image(img, question or "опиши картинку", groq_key, groq_model)
        except Exception as e:  # noqa: BLE001
            logger.exception("answer_about_image failed")
            await status_finish()
            await message.answer(f"Ошибка зрения (Groq): {e}")
            return
        await status_finish()
        text = ans.text if hasattr(ans, "text") else str(ans)
        await message.answer(text[:4000] or "(пустой ответ)")

    if intent == "general":
        await _general_answer()
        return

    # 1) vision
    try:
        await status_update("Анализирую расположение полей… (Groq)")
        vision = await analyze_screenshot(img, question, groq_key, groq_model)
    except Exception as e:  # noqa: BLE001
        logger.exception("analyze_screenshot failed")
        await status_finish()
        await message.answer(f"Ошибка зрения (Groq): {e}")
        return

    logger.info(
        "helpzavr vision: %d candidates",
        len(vision.candidates) if vision and vision.candidates else 0,
    )

    if not vision.candidates:
        # No form fields found → drop to text answer, but tell the user
        # we did so. Previously a silent fallback made it look like the
        # bot "ignored" the fill request entirely.
        await message.answer(
            "ℹ️ Не нашёл на картинке полей формы для разметки — "
            "отвечаю текстом. Если ждал картинку со стрелкой, "
            "пришли скрин с явным полем ввода и вопросом «что вписать в …»."
        )
        await _general_answer()
        return

    # 2) thinking
    try:
        await status_update("Думаю, что туда написать…")
        decision = await decide_what_to_write(
            vision,
            question,
            oro_key,
            oro_model,
            fallback_groq_key=groq_key,
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("decide_what_to_write failed")
        await status_finish()
        await message.answer(f"Ошибка размышления (OpenRouter): {e}")
        return

    chosen = next((c for c in vision.candidates if c.id == decision.chosen_id), None)
    if chosen is None:
        await status_finish()
        await message.answer(
            "Модель не выбрала ни одного поля. Попробуй переформулировать вопрос."
        )
        return

    approx_bbox_px = chosen.to_pixel_bbox(img.width, img.height)

    # 3) refine
    try:
        await status_update("Уточняю границы поля…")
        bbox_px = await refine_bbox_async(
            img,
            approx_bbox_px,
            label=chosen.label,
            kind=chosen.kind,
            nearby_text=chosen.nearby_text,
            api_key=groq_key,
            model=groq_model,
        )
    except Exception:  # noqa: BLE001
        logger.exception("refine_bbox_async failed; using approx bbox")
        bbox_px = approx_bbox_px

    # 4) draw
    await status_update("Рисую…")
    style = Style()
    annotated = annotate(
        img,
        bbox_px,
        instruction=decision.instruction or chosen.context,
        example=decision.example or "",
        style=style,
    )
    out = io.BytesIO()
    annotated.save(out, format="PNG", optimize=True)
    out.seek(0)

    def _esc(s: str) -> str:
        return (
            s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )

    caption_lines = [
        f"<b>Поле:</b> {_esc(chosen.label)}",
        f"<b>Что написать:</b> {_esc(decision.instruction)}".strip(),
    ]
    if decision.example:
        caption_lines.append(
            f"<b>Пример:</b> <code>{_esc(decision.example)}</code>"
        )
    caption = "\n".join(caption_lines)
    if len(caption) > 1024:
        caption = caption[:1020] + "…"

    await message.answer_photo(
        BufferedInputFile(out.getvalue(), filename="annotated.png"),
        caption=caption,
    )
    await status_finish()


# ---- router --------------------------------------------------------------


def build_helpzavr_router() -> Router:
    router = Router(name="helpzavr_addon")

    @router.callback_query(F.data == "hz:on")
    async def on_enable(query: CallbackQuery) -> None:
        if query.message is None:
            await query.answer()
            return
        if not (get_groq_key() and get_openrouter_key()):
            await query.answer("Сначала введи оба ключа", show_alert=True)
            return
        _set_enabled(query.message.chat.id, True)
        await query.answer("Режим включён")
        await show_screen(query)

    @router.callback_query(F.data == "hz:off")
    async def on_disable(query: CallbackQuery) -> None:
        if query.message is None:
            await query.answer()
            return
        _set_enabled(query.message.chat.id, False)
        await query.answer("Режим выключен")
        await show_screen(query)

    @router.callback_query(F.data == "hz:back")
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
            logger.debug("hz: back failed", exc_info=True)
        await query.answer()

    @router.callback_query(F.data.startswith("hz:setkey:"))
    async def on_setkey(query: CallbackQuery) -> None:
        if query.data is None:
            await query.answer()
            return
        # API keys are admin-only: a guest in public mode must not be
        # able to overwrite the Groq / OpenRouter keys.
        from ...access import can_admin

        if not can_admin(query.from_user.id if query.from_user else None):
            await query.answer("Только владелец.", show_alert=True)
            return
        provider = query.data.split(":", 2)[2]
        # Per-chat scoping: ``awaiting_key`` previously lived as a global
        # ``addon_state`` flag, which meant the next plain-text message
        # from ANY chat would be saved as this user's API key. Pin the
        # await-flag to ``chat_id`` so only this chat's next text reply
        # is captured.
        if query.message is None:
            await query.answer()
            return
        addon_state.chat_set(
            "helpzavr", query.message.chat.id, "awaiting_key", provider
        )
        await query.message.answer(
            f"Пришли мне ключ <b>{provider.title()}</b> одним сообщением.\n"
            "Я его сохраню в state.json и удалю твоё сообщение."
        )
        await query.answer()

    async def _awaiting_key_filter(message: Message) -> bool:
        return bool(
            addon_state.chat_get(
                "helpzavr", message.chat.id, "awaiting_key", ""
            )
        )

    @router.message(
        StateFilter(None),
        F.text & ~F.text.startswith("/"),
        _awaiting_key_filter,
    )
    async def on_key_input(message: Message, bot: Bot) -> None:
        chat_id = message.chat.id
        provider = addon_state.chat_get(
            "helpzavr", chat_id, "awaiting_key", ""
        )
        if not provider:
            return
        key = (message.text or "").strip()
        if not key:
            await message.answer("Пустой ключ — попробуй ещё раз.")
            return
        # Validate before saving so the user can't accidentally lock the
        # addon into a permanent 401 by pasting a model name (or any
        # other non-key string) into the «🔑 Groq API key» prompt.
        # ``awaiting_key`` stays set so the next message gets another
        # shot without the user having to re-open the screen.
        if provider == "groq" and not _looks_like_groq_key(key):
            with contextlib.suppress(Exception):
                await message.delete()
            await message.answer(
                "Это не похоже на ключ Groq. Настоящий ключ начинается с "
                "<code>gsk_</code> и не содержит «/». Имя модели сюда "
                "вписывать не нужно — оно задано по умолчанию.\n\n"
                "Возьми ключ на "
                "<a href=\"https://console.groq.com\">console.groq.com</a> "
                "→ API Keys → Create."
            )
            return
        if provider == "openrouter" and not _looks_like_openrouter_key(key):
            with contextlib.suppress(Exception):
                await message.delete()
            await message.answer(
                "Это не похоже на ключ OpenRouter. Настоящий ключ "
                "начинается с <code>sk-or-</code>. Имя модели сюда "
                "вписывать не нужно — оно задано по умолчанию.\n\n"
                "Возьми ключ на "
                "<a href=\"https://openrouter.ai\">openrouter.ai</a> "
                "→ Keys → Create."
            )
            return
        if provider == "groq":
            _set_groq_key(key)
        elif provider == "openrouter":
            _set_openrouter_key(key)
        # Clear the per-chat flag (mirrors ``mailbox._stop_awaiting``).
        addon_state.chat_set("helpzavr", chat_id, "awaiting_key", "")
        with contextlib.suppress(Exception):
            await message.delete()
        await message.answer(f"Ключ <b>{provider}</b> сохранён.")
        await show_screen(message)

    # ---- awaiting-prompt FSM (tg_bot13 parity) --------------------------
    #
    # When ``photo_chooser`` routed a photo here but no caption was
    # present, it parks the FSM in ``HelpzavrStates.awaiting_prompt``
    # (see :func:`request_prompt_for_photo` above). The next free-text
    # message becomes the question and we run the pipeline.

    @router.message(
        StateFilter(HelpzavrStates.awaiting_prompt),
        F.text & ~F.text.startswith("/"),
    )
    async def on_prompt_text(
        message: Message, bot: Bot, state: FSMContext
    ) -> None:
        data = await state.get_data()
        file_id = data.get("hz_file_id")
        await state.clear()
        if not file_id:
            await message.answer(
                "Не нашёл привязанной картинки. Пришли фото ещё раз."
            )
            return
        question = (message.text or "").strip()
        if not (get_groq_key() and get_openrouter_key()):
            await message.answer(
                "Сначала задай Groq и OpenRouter ключи в "
                "<b>🤖 Helpzavr</b> на /start."
            )
            return
        buf = io.BytesIO()
        try:
            await bot.download(file_id, destination=buf)
        except Exception as exc:  # noqa: BLE001
            logger.exception("hz: prompt-after-photo download failed")
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
            await _run_pipeline(message, bot, img, question)
        except Exception as exc:  # noqa: BLE001
            logger.exception("hz: prompt-after-photo pipeline failed")
            await message.answer(f"Ошибка пайплайна Helpzavr: {exc}")

    # NOTE: Helpzavr used to register its own ``F.photo`` /
    # ``F.document.mime_type.startswith("image/")`` handlers here. They
    # have been removed in favour of the single chooser in
    # :mod:`bot.addons.photo_chooser` which asks the user which mode
    # should handle the photo before either pipeline reacts. The
    # chooser invokes ``_run_pipeline`` directly with the cached
    # ``file_id`` so this module's pipeline still runs the same way —
    # it just no longer races with the media generation flow.

    return router
