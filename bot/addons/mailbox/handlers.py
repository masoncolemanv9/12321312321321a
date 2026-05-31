"""Mailbox addon — aiogram router for IMAP setup + manual check + notifications.

Three top-level screens behind the «📬 Проверка почты» main-menu button:

* **Главный экран** — статус подключения, кнопка «Проверить сейчас»,
  входы в подменю.
* **⚙️ Настройки почты** — email / app-password / сервер / папка;
  каждое поле редактируется кнопкой → пользователь шлёт одно сообщение
  → бот удаляет его сообщение и сохраняет.
* **⚙️ Уведомления** — переключатель «📨 все письма» (push нового
  письма в этот чат) и «🟡 По силам ИИ» / «🟢 Точно по силам»
  (заглушки для будущих LLM-фильтров — кнопки видны, но требуют
  активного исполнителя).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

from aiogram import F, Router
from aiogram.filters import StateFilter
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from .. import state as addon_state
from .credentials import is_configured, load_credentials, save_credentials

logger = logging.getLogger(__name__)


_AWAITING_FIELD_KEY = "awaiting_field"  # one of: email | password | host | port | folder


# ---- UI helpers ----------------------------------------------------------


def _kb_main() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="📥 Проверить сейчас", callback_data="mb:check")],
            [
                InlineKeyboardButton(
                    text="⚙️ Уведомления", callback_data="mb:notify"
                )
            ],
            [
                InlineKeyboardButton(
                    text="🤖 Исполнитель", callback_data="mb:exec"
                )
            ],
            [
                InlineKeyboardButton(
                    text="⚙️ Настройки почты", callback_data="mb:settings"
                )
            ],
            [InlineKeyboardButton(text="↩️ Назад", callback_data="mb:back")],
        ]
    )


def _kb_settings() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="✉️ Email", callback_data="mb:set:email")],
            [InlineKeyboardButton(text="🔑 Пароль", callback_data="mb:set:password")],
            [InlineKeyboardButton(text="🌐 Сервер", callback_data="mb:set:host")],
            [InlineKeyboardButton(text="🔌 Порт", callback_data="mb:set:port")],
            [InlineKeyboardButton(text="📂 Папка", callback_data="mb:set:folder")],
            [InlineKeyboardButton(text="🧹 Очистить", callback_data="mb:set:clear")],
            [InlineKeyboardButton(text="↩️ Назад", callback_data="mb:home")],
        ]
    )


def _kb_notify() -> InlineKeyboardMarkup:
    def _emoji(name: str) -> str:
        return "🟢" if addon_state.get("mailbox", f"notify_{name}", False) else "⚪"
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=f"{_emoji('all_emails')} 📨 Все письма",
                    callback_data="mb:notify:all_emails",
                )
            ],
            [
                InlineKeyboardButton(
                    text=f"{_emoji('probable')} 🟡 По силам ИИ (75-95%)",
                    callback_data="mb:notify:probable",
                )
            ],
            [
                InlineKeyboardButton(
                    text=f"{_emoji('certain')} 🟢 Точно по силам (≥95%)",
                    callback_data="mb:notify:certain",
                )
            ],
            [InlineKeyboardButton(text="↩️ Назад", callback_data="mb:home")],
        ]
    )


def _mask(s: str) -> str:
    if not s:
        return "—"
    if len(s) <= 4:
        return "***"
    return f"{s[:2]}…{s[-2:]}"


def _settings_summary() -> str:
    from .config import get_settings
    settings = get_settings()
    creds = load_credentials()
    return (
        f"<b>⚙️ Настройки почты</b>\n\n"
        f"<b>Где взять пароль приложения:</b>\n"
        f"• Yandex: <a href=\"https://id.yandex.ru/security/app-passwords\">id.yandex.ru/security/app-passwords</a>\n"
        f"• Mail.ru: <a href=\"https://account.mail.ru/user/2-step-auth/passwords\">account.mail.ru/...</a>\n"
        f"• Gmail: <a href=\"https://myaccount.google.com/apppasswords\">myaccount.google.com/apppasswords</a>\n"
        f"<i>Это специальный пароль ТОЛЬКО для приложений, не основной от ящика.</i>\n\n"
        f"<b>Текущее:</b>\n"
        f"• Email: <code>{creds.email if creds else '—'}</code>\n"
        f"• Пароль: <code>{_mask(creds.password if creds else '')}</code>\n"
        f"• Сервер: <code>{settings.mailbox_imap_host}:{settings.mailbox_imap_port}</code>\n"
        f"• Папка: <code>{settings.mailbox_imap_folder}</code>\n"
    )


def _main_summary() -> str:
    creds = load_credentials()
    if not creds:
        return (
            "<b>📬 Проверка почты</b>\n\n"
            "⚠ Почта не настроена. Открой <b>⚙️ Настройки почты</b> и "
            "введи email + пароль приложения."
        )
    return (
        "<b>📬 Проверка почты</b>\n\n"
        f"Подключён ящик: <code>{creds.email}</code>\n\n"
        "Что умею:\n"
        "• 📥 «Проверить сейчас» — тяну последние ~50 писем, "
        "распознаю YouDo-задания, шлю карточками.\n"
        "• ⚙️ Уведомления — пуш на каждое новое письмо в чат "
        "(фоновый опрос каждые 5 минут).\n"
        "• ⚙️ Настройки почты — сменить email/пароль/сервер."
    )


# ---- helpers -------------------------------------------------------------


def _start_awaiting(chat_id: int, field: str) -> None:
    addon_state.chat_set("mailbox", chat_id, _AWAITING_FIELD_KEY, field)


def _stop_awaiting(chat_id: int) -> None:
    addon_state.chat_set("mailbox", chat_id, _AWAITING_FIELD_KEY, "")


def _awaiting(chat_id: int) -> str:
    return addon_state.chat_get("mailbox", chat_id, _AWAITING_FIELD_KEY, "") or ""


def _set_env_override(key: str, value: str) -> None:
    """Override an env-var-backed setting via addon_state."""
    addon_state.set_("mailbox", f"env_override_{key}", value)


def _get_env_override(key: str, default: str) -> str:
    return addon_state.get("mailbox", f"env_override_{key}", "") or default


# Monkey-patch get_settings to consider env overrides — avoids editing
# the env at runtime.
def _patch_settings() -> None:
    from . import config as _cfg

    _orig = _cfg.get_settings

    def patched() -> Any:
        s = _orig()
        host = _get_env_override("host", s.mailbox_imap_host)
        port = int(_get_env_override("port", str(s.mailbox_imap_port)) or s.mailbox_imap_port)
        folder = _get_env_override("folder", s.mailbox_imap_folder)
        if host == s.mailbox_imap_host and port == s.mailbox_imap_port and folder == s.mailbox_imap_folder:
            return s
        from dataclasses import replace
        return replace(s, mailbox_imap_host=host, mailbox_imap_port=port, mailbox_imap_folder=folder)

    _cfg.get_settings = patched  # type: ignore[assignment]


_patch_settings()


# ---- show screen ---------------------------------------------------------


async def show_main(target) -> None:
    """Render the mailbox top-level screen via Message or CallbackQuery."""
    text = _main_summary()
    kb = _kb_main()
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


# ---- letter formatting ---------------------------------------------------


def _email_card(email_obj) -> str:
    """Render an EmailContent (from imap_client) as a Telegram card.

    Falls back to plain text when parsers haven't matched.
    """
    subj = getattr(email_obj, "subject", "") or "(без темы)"
    sender = getattr(email_obj, "from_", "") or ""
    body = getattr(email_obj, "text", "") or getattr(email_obj, "body", "") or ""
    body = body.strip()
    if len(body) > 1200:
        body = body[:1200] + "…"

    def _esc(s: str) -> str:
        return (
            (s or "")
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    return (
        f"<b>{_esc(subj)}</b>\n"
        f"<i>от {_esc(sender)}</i>\n\n"
        f"<code>{_esc(body)}</code>"
    )


async def _do_check(message_target) -> str:
    """Manual check: fetch up to 50 recent emails, render summary."""
    from .config import get_settings
    from .imap_client import MailboxNotConfiguredError, fetch_emails_multi_folder
    try:
        settings = get_settings()
        emails = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: fetch_emails_multi_folder(
                folders=list(settings.default_scan_folders),
                limit_per_folder=15,
            ),
        )
    except MailboxNotConfiguredError:
        return "⚠ Почта не настроена. Открой ⚙️ Настройки почты."
    except Exception as e:  # noqa: BLE001
        logger.exception("manual check failed")
        return f"Ошибка IMAP: {e}"

    if not emails:
        return "Свежих писем не найдено."

    # Send the latest 10 as separate cards
    return f"Нашёл {len(emails)} писем. Показываю последние 10:", emails[:10]  # type: ignore[return-value]


# ---- router --------------------------------------------------------------


def build_mailbox_router() -> Router:
    router = Router(name="mailbox_addon")

    @router.callback_query(F.data == "mb:back")
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
            logger.debug("mb: back failed", exc_info=True)
        await query.answer()

    @router.callback_query(F.data == "mb:home")
    async def on_home(query: CallbackQuery) -> None:
        await query.answer()
        await show_main(query)

    @router.callback_query(F.data == "mb:settings")
    async def on_settings(query: CallbackQuery) -> None:
        if query.message is None:
            await query.answer()
            return
        try:
            await query.message.edit_text(
                _settings_summary(),
                reply_markup=_kb_settings(),
                disable_web_page_preview=True,
            )
        except Exception:  # noqa: BLE001
            await query.message.answer(
                _settings_summary(),
                reply_markup=_kb_settings(),
                disable_web_page_preview=True,
            )
        await query.answer()

    @router.callback_query(F.data == "mb:notify")
    async def on_notify(query: CallbackQuery) -> None:
        if query.message is None:
            await query.answer()
            return
        text = (
            "<b>⚙️ Уведомления</b>\n\n"
            "Когда какой-то из режимов <b>ВКЛ</b> — бот в фоне каждые "
            "5 минут опрашивает ящик и шлёт сюда соответствующие письма "
            "как карточки в реальном времени.\n\n"
            "<i>По силам ИИ / Точно по силам</i> — заглушка под будущий "
            "LLM-фильтр; пока работает только «Все письма»."
        )
        try:
            await query.message.edit_text(text, reply_markup=_kb_notify())
        except Exception:  # noqa: BLE001
            await query.message.answer(text, reply_markup=_kb_notify())
        await query.answer()

    @router.callback_query(F.data.startswith("mb:notify:"))
    async def on_notify_toggle(query: CallbackQuery) -> None:
        if query.message is None or query.data is None:
            await query.answer()
            return
        kind = query.data.split(":", 2)[2]
        if kind not in {"all_emails", "probable", "certain"}:
            await query.answer()
            return
        cur = bool(addon_state.get("mailbox", f"notify_{kind}", False))
        addon_state.set_("mailbox", f"notify_{kind}", not cur)
        if not cur:
            # Snapshot chat id so the poller knows where to push.
            addon_state.set_("mailbox", "notify_chat_id", query.message.chat.id)
        text = (
            "<b>⚙️ Уведомления</b>\n\n"
            "Когда какой-то из режимов <b>ВКЛ</b> — бот в фоне каждые "
            "5 минут опрашивает ящик и шлёт сюда соответствующие письма "
            "как карточки в реальном времени.\n\n"
            "<i>По силам ИИ / Точно по силам</i> — заглушка под будущий "
            "LLM-фильтр; пока работает только «Все письма»."
        )
        with contextlib.suppress(Exception):
            await query.message.edit_text(text, reply_markup=_kb_notify())
        await query.answer(f"{'✓ ВКЛ' if not cur else '⏸ ВЫКЛ'}: {kind}")

    @router.callback_query(F.data == "mb:exec")
    async def on_exec(query: CallbackQuery) -> None:
        if query.message is None:
            await query.answer()
            return
        active = addon_state.get("mailbox", "active_executor", "devin-builtin")
        text = (
            "<b>🤖 Исполнитель</b>\n\n"
            f"Активный: <b>{active}</b>\n\n"
            "Этот пункт — заглушка под выбор кастомного LLM-исполнителя "
            "для авто-оценки задач. Пока используется встроенный Devin."
        )
        kb = InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text="↩️ Назад", callback_data="mb:home")]
            ]
        )
        try:
            await query.message.edit_text(text, reply_markup=kb)
        except Exception:  # noqa: BLE001
            await query.message.answer(text, reply_markup=kb)
        await query.answer()

    @router.callback_query(F.data == "mb:check")
    async def on_check(query: CallbackQuery) -> None:
        if query.message is None:
            await query.answer()
            return
        # Route progress through the shared "Соображалка" runner so the
        # user's chosen thinking style applies to mailbox fetches too.
        # In the ⬛ "чёрная по центру" mode no chat bubble is rendered
        # and the user only sees Telegram's native indicator at the top.
        from ..thinking_style import make_status_runner

        status_update, status_finish = make_status_runner(query.message)
        await status_update("Подключаюсь к IMAP…")
        try:
            await status_update("Тяну свежие письма…")
            result = await _do_check(query.message)
        except Exception as e:  # noqa: BLE001
            await status_finish()
            await query.message.answer(f"Ошибка: {e}")
            await query.answer()
            return
        await status_finish()
        if isinstance(result, str):
            await query.message.answer(result)
            await query.answer()
            return
        header, emails = result
        await query.message.answer(header)
        for em in emails:
            try:
                await query.message.answer(_email_card(em))
            except Exception:  # noqa: BLE001
                logger.exception("send email card failed")
        await query.answer()

    @router.callback_query(F.data.startswith("mb:set:"))
    async def on_set_field(query: CallbackQuery) -> None:
        if query.message is None or query.data is None:
            await query.answer()
            return
        field = query.data.split(":", 2)[2]
        if field == "clear":
            addon_state.delete("mailbox", "email")
            addon_state.delete("mailbox", "password")
            with contextlib.suppress(Exception):
                await query.message.edit_text(
                    _settings_summary(),
                    reply_markup=_kb_settings(),
                    disable_web_page_preview=True,
                )
            await query.answer("Учётка очищена")
            return
        _start_awaiting(query.message.chat.id, field)
        labels = {
            "email": "Пришли свой email одним сообщением. Пример: <code>alex@yandex.ru</code>",
            "password": (
                "Пришли <b>пароль приложения</b> одним сообщением.\n"
                "<i>Это не основной пароль от почты! Создай в "
                "id.yandex.ru/security/app-passwords для Yandex.</i>"
            ),
            "host": "Пришли IMAP-сервер. Yandex = <code>imap.yandex.com</code>, Mail.ru = <code>imap.mail.ru</code>, Gmail = <code>imap.gmail.com</code>",
            "port": "Пришли порт IMAP (обычно <code>993</code> для SSL).",
            "folder": "Пришли имя папки IMAP (обычно <code>INBOX</code>).",
        }
        await query.message.answer(labels.get(field, f"Пришли значение для {field}"))
        await query.answer()

    async def _is_awaiting(message: Message) -> bool:
        return bool(_awaiting(message.chat.id))

    @router.message(
        StateFilter(None),
        F.text & ~F.text.startswith("/"),
        _is_awaiting,
    )
    async def on_field_input(message: Message) -> None:
        chat_id = message.chat.id
        field = _awaiting(chat_id)
        value = (message.text or "").strip()
        if not value:
            await message.answer("Пустое значение. Попробуй ещё раз.")
            return
        try:
            if field == "email":
                cur = load_credentials()
                save_credentials(value, cur.password if cur else "")
            elif field == "password":
                cur = load_credentials()
                save_credentials(cur.email if cur else "", value)
            elif field == "host":
                _set_env_override("host", value)
            elif field == "port":
                try:
                    int(value)
                except ValueError:
                    await message.answer("Порт должен быть числом.")
                    return
                _set_env_override("port", value)
            elif field == "folder":
                _set_env_override("folder", value)
            else:
                await message.answer(f"Неизвестное поле: {field}")
                _stop_awaiting(chat_id)
                return
        except Exception as e:  # noqa: BLE001
            logger.exception("save field failed")
            await message.answer(f"Ошибка сохранения: {e}")
            return
        _stop_awaiting(chat_id)
        # delete the user's message (it contains the secret)
        if field == "password":
            with contextlib.suppress(Exception):
                await message.delete()
        await message.answer(
            f"Сохранено: <b>{field}</b>.",
            reply_markup=_kb_settings(),
            disable_web_page_preview=True,
        )

    return router


# ---- background poller ---------------------------------------------------


async def poller_loop(bot, *, get_chat_id) -> None:  # type: ignore[no-untyped-def]
    """Background asyncio task: every N seconds, push new emails when notify is on.

    Started from main.py side-by-side with aiogram polling. Idle if no
    notify mode is enabled or credentials are missing.
    """
    from .config import get_settings
    from .imap_client import MailboxNotConfiguredError, fetch_emails_multi_folder

    seen_uids: set[str] = set()
    settings = get_settings()
    interval = max(60, settings.bot_poll_interval_seconds)

    # Warm-up: skip the first batch (avoid spamming on restart).
    warmed = False

    while True:
        try:
            notify_any = (
                bool(addon_state.get("mailbox", "notify_all_emails", False))
                or bool(addon_state.get("mailbox", "notify_probable", False))
                or bool(addon_state.get("mailbox", "notify_certain", False))
            )
            if not notify_any or not is_configured():
                await asyncio.sleep(interval)
                continue
            settings_loop = get_settings()
            emails = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda s=settings_loop: fetch_emails_multi_folder(
                    folders=list(s.default_scan_folders),
                    limit_per_folder=10,
                ),
            )
            chat_id = addon_state.get("mailbox", "notify_chat_id", None) or get_chat_id()
            if not chat_id:
                await asyncio.sleep(interval)
                continue
            for em in emails:
                uid = str(getattr(em, "uid", "") or getattr(em, "id", ""))
                if uid in seen_uids:
                    continue
                seen_uids.add(uid)
                if not warmed:
                    continue
                if not addon_state.get("mailbox", "notify_all_emails", False):
                    # probable / certain require LLM scoring — not wired yet.
                    continue
                try:
                    await bot.send_message(chat_id, _email_card(em))
                except Exception:  # noqa: BLE001
                    logger.exception("push email card failed")
            warmed = True
        except MailboxNotConfiguredError:
            pass
        except Exception:  # noqa: BLE001
            logger.exception("mailbox poller iteration failed")
        await asyncio.sleep(interval)
