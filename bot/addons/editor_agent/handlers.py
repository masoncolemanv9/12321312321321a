"""Main-menu screen + callback router for the Editor Agent.

The addon is read-mostly: the editor pipeline lives in
``bot/uniqueization/`` and ``bot/workers/editor_v2.py`` (Parts 1-11)
and is invoked by the worker queue on ``edit`` jobs. The UI here only
flips the three runtime knobs the pipeline reads on every invocation:

* ``EDITOR_VERSION``   — selects which worker the factory hands out.
* ``EDITOR_PROFILE``   — selects which intensity ceiling the planner
  uses (light = 0.40, medium = 0.60, heavy = 0.80).
* ``EDITOR_V6_ENABLED`` — master switch for the v6 dispatch path.

State sits in lilush's ``_settings.addons.editor_agent.*`` so it
survives restarts without an extra file for ops to back up. When a
key isn't set, we fall back to the live ``bot.config`` attribute
(which itself defaults to the env var the operator set in the
deploy). This keeps env-var behaviour as the source of truth for
fresh deploys and only overrides it once the user has explicitly
clicked a toggle in the chat UI.
"""

from __future__ import annotations

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

_ADDON = "editor_agent"

# Valid values, kept in sync with bot/workers/__init__.py:build_editor_worker.
_VERSIONS: tuple[str, ...] = ("v1", "v2", "v2.1", "v6")
_PROFILES: tuple[str, ...] = ("light", "medium", "heavy")

# Slider steps come straight from the randomization module so the
# UI and the worker can never disagree on what's a legal slider value.
_UNIQUENESS_STEPS: tuple[int, ...] = SLIDER_STEPS

_VERSION_LABEL = {
    "v1": "v1 — legacy",
    "v2": "v2 — face-aware",
    "v2.1": "v2.1 — cuts/subs",
    "v6": "v6 — creative planner",
}
_PROFILE_LABEL = {
    "light": "light · 0.40",
    "medium": "medium · 0.60",
    "heavy": "heavy · 0.80",
}


# ---- effective-state helpers --------------------------------------------


def _live_config_attr(name: str, default: Any) -> Any:
    """Return ``bot.config.<name>`` if present, else ``default``.

    We import lazily (and tolerate ImportError) so this module stays
    safe to import in test environments that monkey out bot.config.
    """
    try:
        from bot import config as bot_config
    except Exception:  # noqa: BLE001
        return default
    return getattr(bot_config, name, default)


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
    """Return the effective v6 master switch (storage override > env)."""
    override = addon_state.get(_ADDON, "v6_enabled")
    if isinstance(override, bool):
        return override
    return bool(_live_config_attr("EDITOR_V6_ENABLED", False))


def current_uniqueness_pct() -> int:
    """Return the effective uniqueness percent (storage override > env).

    The result is clamped to ``[0, MAX_UNIQUENESS_PCT]``. The cap
    (``4 × DEBATE_PCT``) is the user-specified hard maximum — above it
    variation exceeds the debate-vetted envelope and the UI marks the
    zone red.
    """
    override = addon_state.get(_ADDON, "uniqueness_pct")
    if isinstance(override, int | float):
        return clamp_pct(int(override))
    return clamp_pct(_live_config_attr("EDITOR_UNIQUENESS_PCT", 0))


def _apply_runtime(
    version: str | None,
    profile: str | None,
    v6: bool | None,
    uniqueness_pct: int | None = None,
) -> None:
    """Push the effective values into ``bot.config`` so the factory sees them.

    The factory at ``bot.workers.build_editor_worker`` reads
    ``sys.modules["bot.config"].EDITOR_VERSION`` on every call (see
    Part 3 commit). Mirroring storage into the live module makes the
    UI toggle take effect immediately, without requiring a process
    restart.
    """
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


# ---- screen --------------------------------------------------------------


def _kb_screen() -> InlineKeyboardMarkup:
    version = current_version()
    profile = current_profile()
    v6 = is_v6_enabled()

    rows: list[list[InlineKeyboardButton]] = []

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
    rows.append(
        [InlineKeyboardButton(text="↩️ Назад", callback_data="editor_agent:back")]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _kb_uniqueness_screen() -> InlineKeyboardMarkup:
    """Build the keyboard for the slider sub-screen.

    Buttons are taken from :data:`SLIDER_STEPS`. The currently active
    step is prefixed with ▶, green-band steps with 🟢, red-band ones
    with 🔴, so the user sees the safety zone at a glance.
    """
    current = current_uniqueness_pct()

    def _label(step: int) -> str:
        if step == 0:
            prefix = "·"
        elif step <= DEBATE_PCT:
            prefix = "🟢"
        else:
            prefix = "🔴"
        marker = "▶ " if step == current else ""
        return f"{marker}{prefix} {step}%"

    button_for = lambda step: InlineKeyboardButton(  # noqa: E731
        text=_label(step),
        callback_data=f"editor_agent:uniq:set:{step}",
    )
    # Split SLIDER_STEPS into row groups of 5 for a phone-friendly
    # 3-row layout.
    rows: list[list[InlineKeyboardButton]] = []
    chunk = 5
    for i in range(0, len(_UNIQUENESS_STEPS), chunk):
        rows.append([button_for(s) for s in _UNIQUENESS_STEPS[i : i + chunk]])
    rows.append(
        [
            InlineKeyboardButton(
                text="↩️ Назад в Монтажёр",
                callback_data="editor_agent:uniq:back",
            )
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _uniqueness_screen_body() -> str:
    """Render the Russian-language explanation for the slider screen.

    Pulls debate-vetted ranges from :data:`DEBATE_RANGES` so the UI
    and the worker can never drift apart — if someone changes the
    range table, both update simultaneously.
    """
    pct = current_uniqueness_pct()
    zone = describe_zone(pct)

    range_lines: list[str] = []
    for spec in DEBATE_RANGES:
        range_lines.append(
            f"• <code>{spec.field}</code> — "
            f"debate envelope <b>{spec.debate_lo:g}…{spec.debate_hi:g}</b>, "
            f"physical bound <b>{spec.physical_lo:g}…{spec.physical_hi:g}</b>\n"
            f"   <i>{spec.debate_ref}</i>"
        )

    ranges_block = "\n".join(range_lines)

    return (
        "<b>🎯 Уникальность</b>\n\n"
        f"Текущее значение: <b>{pct}%</b> — {zone}\n\n"
        "<b>Что делает слайдер</b>\n"
        "При каждом <code>edit</code> job воркер варьирует четыре "
        "параметра ffmpeg внутри debate-vetted envelope (трёх профилей "
        "<code>light → medium → heavy</code>, утверждённых D1+D2 после "
        "5 раундов дебатов). Сид выводится из <code>job_id</code> + "
        "<code>upload_ts</code> + <code>clip_index</code> — один и тот же "
        "ролик, залитый в 100 разных jobs, получит 100 разных вариантов.\n\n"
        "<b>Per-job формула</b>\n"
        "<code>v = uniform(0.6 × X, X)</code> — фактическая вариация для этого job.\n"
        "При <code>X=5</code> → <code>v ∈ [3, 5]</code> (пример).\n"
        "Затем <code>spread = debate_width × v / 100</code>, "
        "<code>offset = uniform(-spread/2, +spread/2)</code>, "
        "<code>value = clamp(center + offset, physical_lo, physical_hi)</code>.\n\n"
        f"<b>0%</b> — без вариации (Part 1-11 baseline, byte-equivalent).\n"
        f"<b>{DEBATE_PCT}%</b> — ровно одна debate envelope (граница зелёного).\n"
        f"<b>{MAX_UNIQUENESS_PCT}%</b> — 4× debate envelope (максимум, красный).\n\n"
        "<b>Debate-vetted параметры</b>\n"
        f"{ranges_block}\n\n"
        "<b>Зоны слайдера</b>\n"
        "• <b>0</b> — без рандома (debate-default = текущая реализация Part 1-11)\n"
        f"• <b>1 — {DEBATE_PCT}%</b> — 🟢 внутри debate-envelope\n"
        f"• <b>{DEBATE_PCT + 1} — {MAX_UNIQUENESS_PCT}%</b> — "
        "🔴 <b>выходим из рамок дебатов</b>: вариация превышает "
        "утверждённый D1+D2 envelope (cap = 4× по требованию пользователя).\n\n"
        f"Слайдер заперт на {MAX_UNIQUENESS_PCT}% — выше дебатовые границы "
        "уже не действуют. Сид и фактические выборы пишутся в "
        "<code>uniqueization.json</code> → "
        "<code>extra.uniqueness_randomization</code>."
    )


async def show_uniqueness_screen(message_or_query: Message | CallbackQuery) -> None:
    """Render the 🎯 Уникальность slider sub-screen."""
    text = _uniqueness_screen_body()
    kb = _kb_uniqueness_screen()
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


def _screen_body() -> str:
    version = current_version()
    profile = current_profile()
    v6 = is_v6_enabled()

    pct = current_uniqueness_pct()
    return (
        "<b>🎞 Монтажёр</b>\n\n"
        "Конфигурация пайплайна редактора видео — её читает воркер "
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


async def show_screen(message_or_query: Message | CallbackQuery) -> None:
    """Render the editor-agent config screen.

    Works for both the slash-command entry point (a ``Message``) and
    the main-menu callback (a ``CallbackQuery``). When called from a
    callback we edit the existing message in-place so the menu doesn't
    pile up.
    """
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


# ---- router --------------------------------------------------------------


def build_editor_agent_router() -> Router:
    router = Router(name="editor_agent_addon")

    @router.callback_query(F.data.startswith("editor_agent:ver:"))
    async def on_set_version(query: CallbackQuery) -> None:
        value = (query.data or "").split(":", 2)[2]
        if value not in _VERSIONS:
            await query.answer("Неизвестная версия", show_alert=True)
            return
        addon_state.set_(_ADDON, "version_override", value)
        _apply_runtime(version=value, profile=None, v6=None)
        await query.answer(f"Версия: {value}")
        await show_screen(query)

    @router.callback_query(F.data.startswith("editor_agent:prof:"))
    async def on_set_profile(query: CallbackQuery) -> None:
        value = (query.data or "").split(":", 2)[2]
        if value not in _PROFILES:
            await query.answer("Неизвестный профиль", show_alert=True)
            return
        addon_state.set_(_ADDON, "profile", value)
        _apply_runtime(version=None, profile=value, v6=None)
        await query.answer(f"Профиль: {value}")
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
        import os

        addon_state.delete(_ADDON, "version_override")
        addon_state.delete(_ADDON, "profile")
        addon_state.delete(_ADDON, "v6_enabled")
        addon_state.delete(_ADDON, "uniqueness_pct")
        # Reload env-default values into the live config module too so
        # the factory sees them.
        try:
            from bot import config as bot_config

            bot_config.EDITOR_VERSION = (
                os.environ.get("EDITOR_VERSION", "v1").strip().lower()
            )
            bot_config.EDITOR_PROFILE = (
                os.environ.get("EDITOR_PROFILE", "light").strip().lower()
            )
            bot_config.EDITOR_V6_ENABLED = (
                os.environ.get("EDITOR_V6_ENABLED", "false").strip().lower()
                == "true"
            )
            bot_config.EDITOR_UNIQUENESS_PCT = clamp_pct(
                os.environ.get("EDITOR_UNIQUENESS_PCT", "0")
            )
        except Exception:  # noqa: BLE001
            logger.exception("editor_agent reset failed to refresh config")
        await query.answer("Сброшено к env-defaults")
        await show_screen(query)

    @router.callback_query(F.data == "editor_agent:uniq:open")
    async def on_uniq_open(query: CallbackQuery) -> None:
        await query.answer()
        await show_uniqueness_screen(query)

    @router.callback_query(F.data == "editor_agent:uniq:back")
    async def on_uniq_back(query: CallbackQuery) -> None:
        await query.answer()
        await show_screen(query)

    @router.callback_query(F.data.startswith("editor_agent:uniq:set:"))
    async def on_uniq_set(query: CallbackQuery) -> None:
        raw = (query.data or "").split(":", 3)[3]
        try:
            value = clamp_pct(int(raw))
        except (TypeError, ValueError):
            await query.answer("Некорректный процент", show_alert=True)
            return
        if value not in _UNIQUENESS_STEPS:
            await query.answer("Шаг вне сетки", show_alert=True)
            return
        addon_state.set_(_ADDON, "uniqueness_pct", value)
        _apply_runtime(version=None, profile=None, v6=None, uniqueness_pct=value)
        await query.answer(f"Уникальность: {value}%")
        await show_uniqueness_screen(query)

    @router.callback_query(F.data == "editor_agent:back")
    async def on_back(query: CallbackQuery) -> None:
        try:
            from ...wizard import _kb_main_after_claim

            uid = query.from_user.id if query.from_user else None
            kb = _kb_main_after_claim(uid)
            if query.message is not None:
                msg = query.message
                try:
                    await msg.edit_text(  # type: ignore[union-attr]
                        "Главное меню. Выбери что сделать:", reply_markup=kb
                    )
                except Exception:  # noqa: BLE001
                    await msg.answer(
                        "Главное меню. Выбери что сделать:", reply_markup=kb
                    )
        except Exception:  # noqa: BLE001
            logger.exception("editor_agent back failed to render main menu")
        await query.answer()

    return router
