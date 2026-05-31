from __future__ import annotations

import contextlib
import logging
from typing import Any

from aiogram import F, Router
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from ...uniqueization.randomization import (
    DEBATE_PCT,
    DEBATE_RANGES,
    MAX_UNIQUENESS_PCT,
    SLIDER_STEPS,
    clamp_pct,
    describe_zone,
)
from .. import state as addon_state

logger = logging.getLogger(__name__)

router = Router(name="editor_agent")

_ADDON = "editor_agent"

_VERSIONS: tuple[str, ...] = ("v1", "v2", "v2.1", "v6")
_PROFILES: tuple[str, ...] = ("light", "medium", "heavy")

_VERSION_LABEL = {
    "v1": "v1",
    "v2": "v2",
    "v2.1": "v2.1",
    "v6": "v6",
}
_PROFILE_LABEL = {
    "light": "light",
    "medium": "medium",
    "heavy": "heavy",
}


# ---------------------------------------------------------------------------
# Active source (set when user clicks 🎞 Монтажёр under a downloaded video).
# Stored per-addon so the next "▶️ Запустить" press picks it up. Cleared
# automatically after a successful run.
# ---------------------------------------------------------------------------


def _set_active_source(source_path: str, *, job_id: int | None, chat_id: int | None,
                       duration_s: float | None, language: str | None) -> None:
    addon_state.set_(_ADDON, "active_source_path", source_path)
    addon_state.set_(_ADDON, "active_source_job_id", job_id)
    addon_state.set_(_ADDON, "active_source_chat_id", chat_id)
    addon_state.set_(_ADDON, "active_source_duration_s", duration_s)
    addon_state.set_(_ADDON, "active_source_language", language)


def _get_active_source() -> dict[str, Any] | None:
    path = addon_state.get(_ADDON, "active_source_path")
    if not isinstance(path, str) or not path:
        return None
    return {
        "source_path": path,
        "job_id": addon_state.get(_ADDON, "active_source_job_id"),
        "chat_id": addon_state.get(_ADDON, "active_source_chat_id"),
        "duration_s": addon_state.get(_ADDON, "active_source_duration_s"),
        "language": addon_state.get(_ADDON, "active_source_language"),
    }


def _clear_active_source() -> None:
    for key in (
        "active_source_path",
        "active_source_job_id",
        "active_source_chat_id",
        "active_source_duration_s",
        "active_source_language",
    ):
        with contextlib.suppress(Exception):
            addon_state.set_(_ADDON, key, None)


def _live_config_attr(attr: str, default: Any) -> Any:
    try:
        from bot import config as bot_config
    except Exception:  # noqa: BLE001
        return default
    return getattr(bot_config, attr, default)


def current_version() -> str:
    """Return the effective editor version (storage override > env)."""
    override = addon_state.get(_ADDON, "version_override")
    if isinstance(override, str) and override.strip().lower() in _VERSIONS:
        return override.strip().lower()
    raw = str(_live_config_attr("EDITOR_VERSION", "v1") or "v1").lower().strip()
    return raw if raw in _VERSIONS else "v1"


def current_profile() -> str:
    """Return the effective intensity profile (storage override > env)."""
    override = addon_state.get(_ADDON, "profile")
    if isinstance(override, str) and override.strip().lower() in _PROFILES:
        return override.strip().lower()
    raw = str(_live_config_attr("EDITOR_PROFILE", "light") or "light").lower().strip()
    return raw if raw in _PROFILES else "light"


def is_v6_enabled() -> bool:
    override = addon_state.get(_ADDON, "v6_enabled")
    if isinstance(override, bool):
        return override
    return bool(_live_config_attr("EDITOR_V6_ENABLED", False))


def current_uniqueness_pct() -> int:
    override = addon_state.get(_ADDON, "uniqueness_pct")
    if isinstance(override, int):
        return clamp_pct(override)
    raw = _live_config_attr("EDITOR_UNIQUENESS_PCT", 0) or 0
    try:
        return clamp_pct(int(raw))
    except Exception:  # noqa: BLE001
        return 0


def _apply_runtime(
    version: str | None,
    profile: str | None,
    v6: bool | None,
    uniqueness_pct: int | None = None,
) -> None:
    """Push the effective values into ``bot.config`` so the factory sees them."""
    try:
        from bot import config as bot_config
    except Exception:  # noqa: BLE001
        return
    if version is not None:
        bot_config.EDITOR_VERSION = version
    if profile is not None:
        bot_config.EDITOR_PROFILE = profile
    if v6 is not None:
        bot_config.EDITOR_V6_ENABLED = v6
    if uniqueness_pct is not None:
        bot_config.EDITOR_UNIQUENESS_PCT = clamp_pct(uniqueness_pct)


def _short_path(path: str, *, max_len: int = 60) -> str:
    if len(path) <= max_len:
        return path
    return "…" + path[-(max_len - 1):]


def _screen_body() -> str:
    version = current_version()
    profile = current_profile()
    v6 = is_v6_enabled()
    pct = current_uniqueness_pct()

    active = _get_active_source()
    active_block = ""
    if active:
        active_block = (
            "<b>📎 Активный файл:</b> "
            f"<code>{_short_path(active['source_path'])}</code>\n"
            "После выбора настроек жми <b>▶️ Запустить</b> — пайплайн "
            "пойдёт с этим файлом.\n\n"
        )

    return (
        "<b>🎞 Монтажёр</b>\n\n"
        + active_block
        + "Конфигурация пайплайна редактора видео — её читает воркер "
        "<code>edit</code> при каждом запуске. Меняй прямо здесь, "
        "перезапуск не нужен.\n\n"
        f"• <b>Версия:</b> <code>{version}</code> ({_VERSION_LABEL[version]})\n"
        f"• <b>Профиль:</b> <code>{profile}</code> ({_PROFILE_LABEL[profile]})\n"
        f"• <b>v6 creative planner:</b> {'ВКЛ ✅' if v6 else 'ВЫКЛ ⛔'}\n"
        f"• <b>Уникальность:</b> <code>{pct}%</code> "
        f"({describe_zone(pct)})\n\n"
        "<b>Что делает каждая версия:</b>\n"
        "• <b>v1</b> — legacy: crop+scale + опциональный логотип. "
        "Дефолт продакшна.\n"
        "• <b>v2</b> — face-aware zoom, mirror, colorgrade, loudnorm, "
        "манифест.\n"
        "• <b>v2.1</b> — поверх v2: cuts, subtitles (ASS, safe-area), "
        "blur_fill, hook_emphasis, unique_distance.\n"
        "• <b>v6</b> — поверх v2.1: creative planner "
        "(beat sheet → zoom → cuts → mirror → subtitle → audio → "
        "color → blur_fill → hook → manifest reasoning → intensity).\n\n"
        "<b>Профиль:</b> верхний потолок интенсивности креатив-планнера. "
        "Light не позволяет планнеру тратить более 40% бюджета "
        "изменений, medium — 60%, heavy — 80%. На v1/v2 не влияет.\n\n"
        "<b>v6 toggle:</b> мастер-выключатель v6 dispatch. Когда выключен "
        "и версия = v6, воркер падает в v2.1 path (без агрессивного "
        "переосмысления раскладки).\n\n"
        "<b>Уникальность:</b> per-job рандомизация четырёх ffmpeg-параметров "
        "(zoom / effects_opacity / audio_fx_wet / mirror_duration_s) внутри "
        "диапазонов, задокументированных в "
        "<code>docs/DEBATE_TOPICS/editor-agent-v2.md</code>. При 0% — без "
        f"вариации (byte-equivalent). Cap = {MAX_UNIQUENESS_PCT}%. "
        "Жми кнопку чтобы открыть слайдер.\n\n"
        "<i>Сброс</i> возвращает все четыре параметра к значениям из "
        "env-переменных (<code>EDITOR_VERSION</code>, "
        "<code>EDITOR_PROFILE</code>, <code>EDITOR_V6_ENABLED</code>, "
        "<code>EDITOR_UNIQUENESS_PCT</code>)."
    )


def _kb_screen() -> InlineKeyboardMarkup:
    version = current_version()
    profile = current_profile()
    v6 = is_v6_enabled()
    active = _get_active_source()

    rows: list[list[InlineKeyboardButton]] = []

    # Run button at the top if we have an active source.
    if active:
        rows.append(
            [
                InlineKeyboardButton(
                    text="▶️ Запустить",
                    callback_data="editor_agent:run",
                )
            ]
        )

    # Version row (4 buttons, single-select).
    rows.append(
        [
            InlineKeyboardButton(
                text=("▶ " if version == v else "") + _VERSION_LABEL[v],
                callback_data=f"editor_agent:ver:{v}",
            )
            for v in _VERSIONS
        ]
    )
    # Profile row (3 buttons, single-select).
    rows.append(
        [
            InlineKeyboardButton(
                text=("▶ " if profile == p else "") + _PROFILE_LABEL[p],
                callback_data=f"editor_agent:prof:{p}",
            )
            for p in _PROFILES
        ]
    )
    # v6 toggle.
    rows.append(
        [
            InlineKeyboardButton(
                text=("⏸ Выключить v6" if v6 else "▶ Включить v6"),
                callback_data=("editor_agent:v6:off" if v6 else "editor_agent:v6:on"),
            )
        ]
    )
    # Reset to env-var defaults.
    rows.append(
        [
            InlineKeyboardButton(
                text="🔄 Сбросить к env-defaults",
                callback_data="editor_agent:reset",
            )
        ]
    )
    # Uniqueness sub-screen entry button.
    rows.append(
        [
            InlineKeyboardButton(
                text=f"🎯 Уникальность: {current_uniqueness_pct()}%",
                callback_data="editor_agent:uniq:open",
            )
        ]
    )
    # If we have an active file, also offer a "clear active file" affordance.
    if active:
        rows.append(
            [
                InlineKeyboardButton(
                    text="❌ Снять активный файл",
                    callback_data="editor_agent:run:clear",
                )
            ]
        )
    rows.append(
        [InlineKeyboardButton(text="↩️ Назад", callback_data="editor_agent:back")]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


async def show_screen(
    message_or_query: Message | CallbackQuery,
    *,
    source_path: str | None = None,
    job_id: int | None = None,
    chat_id: int | None = None,
    duration_s: float | None = None,
    language: str | None = None,
) -> None:
    """Render the editor-agent config screen.

    Works for both the slash-command entry point (a ``Message``) and
    the main-menu callback (a ``CallbackQuery``). When called from a
    callback we edit the existing message in-place so the menu doesn't
    pile up.

    If ``source_path`` is provided, it is stored as the "active file"
    so the user can pick version/profile/v6 settings and then press
    "▶️ Запустить" to launch analyze on it.
    """
    if source_path:
        _set_active_source(
            source_path,
            job_id=job_id,
            chat_id=chat_id,
            duration_s=duration_s,
            language=language,
        )

    text = _screen_body()
    kb = _kb_screen()
    if isinstance(message_or_query, CallbackQuery):
        if message_or_query.message is None:
            await message_or_query.answer()
            return
        msg = message_or_query.message
        try:
            await msg.edit_text(text, reply_markup=kb)  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            await msg.answer(text, reply_markup=kb)
    else:
        await message_or_query.answer(text, reply_markup=kb)


# ---------------------------------------------------------------------------
# Callback handlers
# ---------------------------------------------------------------------------


@router.callback_query(F.data.startswith("editor_agent:ver:"))
async def on_set_version(query: CallbackQuery) -> None:
    value = (query.data or "").split(":")[-1].strip().lower()
    if value not in _VERSIONS:
        await query.answer("Неизвестная версия", show_alert=True)
        return
    addon_state.set_(_ADDON, "version_override", value)
    _apply_runtime(version=value, profile=None, v6=None)
    await query.answer(f"Версия → {value}")
    await show_screen(query)


@router.callback_query(F.data.startswith("editor_agent:prof:"))
async def on_set_profile(query: CallbackQuery) -> None:
    value = (query.data or "").split(":")[-1].strip().lower()
    if value not in _PROFILES:
        await query.answer("Неизвестный профиль", show_alert=True)
        return
    addon_state.set_(_ADDON, "profile", value)
    _apply_runtime(version=None, profile=value, v6=None)
    await query.answer(f"Профиль → {value}")
    await show_screen(query)


@router.callback_query(F.data == "editor_agent:v6:on")
async def on_v6_on(query: CallbackQuery) -> None:
    addon_state.set_(_ADDON, "v6_enabled", True)
    _apply_runtime(version=None, profile=None, v6=True)
    await query.answer("v6 включён")
    await show_screen(query)


@router.callback_query(F.data == "editor_agent:v6:off")
async def on_v6_off(query: CallbackQuery) -> None:
    addon_state.set_(_ADDON, "v6_enabled", False)
    _apply_runtime(version=None, profile=None, v6=False)
    await query.answer("v6 выключен")
    await show_screen(query)


@router.callback_query(F.data == "editor_agent:reset")
async def on_reset(query: CallbackQuery) -> None:
    import os as _os
    for key in ("version_override", "profile", "v6_enabled", "uniqueness_pct"):
        with contextlib.suppress(Exception):
            addon_state.set_(_ADDON, key, None)
    try:
        from bot import config as bot_config
        bot_config.EDITOR_VERSION = _os.environ.get("EDITOR_VERSION", "v1")
        bot_config.EDITOR_PROFILE = _os.environ.get("EDITOR_PROFILE", "light")
        bot_config.EDITOR_V6_ENABLED = _os.environ.get("EDITOR_V6_ENABLED", "").lower() in ("1", "true", "yes", "on")
        try:
            bot_config.EDITOR_UNIQUENESS_PCT = int(_os.environ.get("EDITOR_UNIQUENESS_PCT", "0") or 0)
        except Exception:  # noqa: BLE001
            bot_config.EDITOR_UNIQUENESS_PCT = 0
    except Exception:  # noqa: BLE001
        pass
    await query.answer("Сброшено к env-defaults")
    await show_screen(query)


@router.callback_query(F.data == "editor_agent:uniq:open")
async def on_uniq_open(query: CallbackQuery) -> None:
    await show_uniqueness_screen(query)


@router.callback_query(F.data == "editor_agent:uniq:back")
async def on_uniq_back(query: CallbackQuery) -> None:
    await show_screen(query)


@router.callback_query(F.data.startswith("editor_agent:uniq:set:"))
async def on_uniq_set(query: CallbackQuery) -> None:
    try:
        value = int((query.data or "").split(":")[-1])
    except (TypeError, ValueError):
        await query.answer("Битый шаг", show_alert=True)
        return
    if value not in SLIDER_STEPS:
        await query.answer("Шаг вне допустимых", show_alert=True)
        return
    addon_state.set_(_ADDON, "uniqueness_pct", value)
    _apply_runtime(version=None, profile=None, v6=None, uniqueness_pct=value)
    await query.answer(f"Уникальность → {value}%")
    await show_uniqueness_screen(query)


@router.callback_query(F.data == "editor_agent:back")
async def on_back(query: CallbackQuery) -> None:
    from ...wizard import _kb_main_after_claim  # type: ignore
    if query.message is None:
        await query.answer()
        return
    with contextlib.suppress(Exception):
        await query.message.edit_text(  # type: ignore[union-attr]
            "Главное меню", reply_markup=_kb_main_after_claim()
        )
    await query.answer()


@router.callback_query(F.data == "editor_agent:run:clear")
async def on_run_clear(query: CallbackQuery) -> None:
    _clear_active_source()
    await query.answer("Активный файл снят")
    await show_screen(query)


@router.callback_query(F.data == "editor_agent:run")
async def on_run(query: CallbackQuery) -> None:
    """Launch analyze on the active source file with current config settings."""
    from ..access import can_use  # type: ignore

    if not can_use(query.from_user.id if query.from_user else None):
        await query.answer("Нет доступа.", show_alert=True)
        return

    active = _get_active_source()
    if not active:
        await query.answer("Нет активного файла", show_alert=True)
        return

    chat_id = active.get("chat_id")
    if chat_id is None and query.message is not None:
        chat_id = query.message.chat.id
    if chat_id is None:
        await query.answer("Нет чата", show_alert=True)
        return

    try:
        from ...jobs import get_default_queue  # type: ignore
    except Exception:  # noqa: BLE001
        try:
            from bot.jobs import get_default_queue  # type: ignore
        except Exception:  # noqa: BLE001
            await query.answer("Очередь недоступна", show_alert=True)
            return

    queue = get_default_queue()
    await queue.enqueue(
        "analyze",
        {
            "source_path": active["source_path"],
            "duration_s": active.get("duration_s") or 0.0,
            "language": active.get("language"),
            "editor_version": current_version(),
            "editor_profile": current_profile(),
            "editor_v6_enabled": is_v6_enabled(),
            "editor_uniqueness_pct": current_uniqueness_pct(),
        },
        chat_id=chat_id,
        parent_id=active.get("job_id"),
    )

    version = current_version()
    profile = current_profile()
    v6 = is_v6_enabled()
    pct = current_uniqueness_pct()

    _clear_active_source()
    await query.answer("Монтажёр запущен")
    if query.message is not None:
        with contextlib.suppress(Exception):
            await query.message.edit_text(  # type: ignore[union-attr]
                "🎞 <b>Монтажёр запущен</b>\n\n"
                f"• Версия: <code>{version}</code>\n"
                f"• Профиль: <code>{profile}</code>\n"
                f"• v6: {'ВКЛ' if v6 else 'ВЫКЛ'}\n"
                f"• Уникальность: <code>{pct}%</code>\n\n"
                "<i>анализ → нарезка → SEO → публикация</i>\n\n"
                "Когда нарезки будут готовы — пришлю их сюда же. "
                "Прогресс — команда <code>/jobs</code>.",
            )


# ---------------------------------------------------------------------------
# Uniqueness sub-screen
# ---------------------------------------------------------------------------


def _uniq_body() -> str:
    pct = current_uniqueness_pct()
    return (
        "<b>🎯 Уникальность</b>\n\n"
        f"Текущее значение: <code>{pct}%</code> ({describe_zone(pct)})\n\n"
        "Per-job рандомизация четырёх ffmpeg-параметров "
        "(zoom / effects_opacity / audio_fx_wet / mirror_duration_s) внутри "
        "диапазонов из <code>docs/DEBATE_TOPICS/editor-agent-v2.md</code>.\n\n"
        f"Дебатный центр: <code>{DEBATE_PCT}%</code>. "
        f"Cap: <code>{MAX_UNIQUENESS_PCT}%</code>."
    )


def _kb_uniq() -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    current = current_uniqueness_pct()
    # Steps split into rows of 4.
    row: list[InlineKeyboardButton] = []
    for step in SLIDER_STEPS:
        label = f"{'▶ ' if step == current else ''}{step}%"
        row.append(
            InlineKeyboardButton(
                text=label, callback_data=f"editor_agent:uniq:set:{step}"
            )
        )
        if len(row) == 4:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    rows.append(
        [InlineKeyboardButton(text="↩️ Назад", callback_data="editor_agent:uniq:back")]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


async def show_uniqueness_screen(query: CallbackQuery) -> None:
    if query.message is None:
        await query.answer()
        return
    msg = query.message
    text = _uniq_body()
    kb = _kb_uniq()
    try:
        await msg.edit_text(text, reply_markup=kb)  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        await msg.answer(text, reply_markup=kb)
