"""First-run setup wizard for the bot.

Flow when a fresh container starts up:

1. *Anyone* may click ``/start``. The first user who does so is locked in
   as the **owner** (single-tenant). All later messages from anyone else
   get a polite refusal.

2. Owner is shown a 3-button menu to pick a brain:

   - ``[OpenRouter]``  → opens API/Model/URL config sub-menu
   - ``[Devin.ai]``    → emits a copy-pasteable prompt for a new Devin session
   - ``[Другое]``      → same sub-menu as OpenRouter (custom OpenAI-compatible
     endpoint such as Together, Groq, vLLM, LM Studio, etc.)

3. In the API/Model/URL sub-menu each button starts an FSM step where the
   bot asks for that single field. Sensitive values (API key) get the
   user's message deleted right after capture.

The wizard handlers live in their own ``Router`` so ``handlers.py`` stays
focused on day-to-day commands. The wizard router is included by
``main.py`` *before* the main router so its callback queries take
priority.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
from pathlib import Path

from aiogram import F, Router
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from .access import can_admin, can_use, is_owner
from .persona import get_persona, list_personas
from .storage import storage

logger = logging.getLogger(__name__)
wizard_router = Router(name="wizard")

# Filter for FSM text capture — commands like /cancel are filtered out so
# the user can bail mid-wizard. Defined here (early) because several
# handlers below use it as a decorator argument and decorators are
# evaluated top-down.
_NOT_A_COMMAND = F.text & ~F.text.startswith("/")


# ---- FSM states ----------------------------------------------------------


class SetupStates(StatesGroup):
    awaiting_api_key = State()
    awaiting_model = State()
    awaiting_url = State()
    # External research/scraping tools (Apify / Firecrawl / Tavily / ...).
    # We stash the chosen tool name in FSM data so the capture handler
    # knows which key to save.
    awaiting_external_tool_key = State()
    # «Прослушать» in the «Запись голоса» menu: bot waits for a
    # text message and synthesises it with the currently selected
    # voice (or the user's cloned voice).
    awaiting_voice_preview_text = State()
    # «🔴 Запись» — bot waits for an audio/voice file the user
    # wants to clone. The file is saved as the user's "recording"
    # sample; converting it to a Voice-Builder JSON style is a
    # separate manual step (see the Voice Builder info screen).
    awaiting_voice_recording = State()
    # «📥 Импорт» — bot waits for a Voice-Builder JSON document so
    # it can be saved as the user's active cloned voice.
    awaiting_voice_json_import = State()
    # «📚 Узнать о функционале» — bot routes the next text message(s)
    # through the main LLM with a tutorial framing that lists every
    # known capability. Stays active until the user types /cancel so
    # they can ask follow-up questions without re-opening the menu.
    awaiting_learn_question = State()
    # ☁️ ElevenLabs TTS provider — two prompts to capture the API key
    # and the voice_id (copied from the user's ElevenLabs Voice
    # Library page). Both saved per-bot, never per-chat.
    awaiting_elevenlabs_api_key = State()
    awaiting_elevenlabs_voice_id = State()
    # 🔌 «API» — batch import: one giant string with everything
    # separated by ``$$$``. Parsed by :func:`_apply_batch_api_blob`.
    awaiting_batch_api_blob = State()
    # «🍪 Загрузить cookies» on the /dl screen — bot waits for a
    # Netscape-format cookies.txt file (exported via «Get
    # cookies.txt LOCALLY» or similar). Saved to
    # ``data/yt_cookies.txt`` and picked up automatically by the
    # yt-dlp downloader on the next «скачай <URL>».
    awaiting_yt_cookies_file = State()


class AccessStates(StatesGroup):
    # Waiting for the user to send the Telegram id of the account that
    # should be granted full admin access (co-owner).
    awaiting_co_owner_id = State()


# ---- Keyboards -----------------------------------------------------------


def _kb_claim() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🚀 Запустить и стать владельцем", callback_data="claim_owner")]
        ]
    )


_GROQ_URL_HINT = "groq.com"


def _provider_hint_from(provider: str, base_url: str) -> str:
    """Map (provider, base_url) → short label: «OpenRouter», «Groq», ...

    Used to render the dynamic per-slot button labels in the brain menu.
    Empty / unset returns ``""`` so the caller can append nothing.
    """
    base = (base_url or "").lower()
    if _GROQ_URL_HINT in base:
        return "Groq"
    if provider == "openrouter":
        return "OpenRouter"
    if provider == "custom" and base:
        return "свой endpoint"
    return ""


def _slot_config(slot: str) -> tuple[str, str, str, str]:
    """Return (provider, api_key, base_url, model) for the requested slot.

    Slot 1 reads from the legacy top-level fields so existing users keep
    their config untouched. Slot 2 reads from the nested ``brain_slot2``
    dict added in Phase 2.
    """
    if slot == "2":
        s = storage.get_brain_slot2()
        return (
            s.get("provider", "custom") or "custom",
            s.get("api_key", "") or "",
            s.get("base_url", "") or "",
            s.get("model", "") or "",
        )
    return (
        storage.get_provider() or "openrouter",
        storage.get_provider_key("openrouter") or "",
        storage.get_base_url() or "",
        storage.get_model() or "",
    )


def _slot_label(slot: str) -> str:
    """Dynamic ``Мозг N`` label suffixed with the detected provider.

    Slot 1 is the chat brain and always shows the ▶ marker when
    configured. Slot 2 is reserved for voice/photo (Groq override) and
    gets a «voice/photo» suffix so the user knows it never picks up
    chat traffic.

    Examples: ``▶ Мозг 1 (OpenRouter)``, ``Мозг 2 (Groq, voice/photo)``,
    ``Мозг 2 (не задан)`` if no API key is set yet.
    """
    provider, api_key, base_url, _ = _slot_config(slot)
    hint = _provider_hint_from(provider, base_url)
    if not api_key:
        return f"Мозг {slot} (не задан)"
    if slot == "1":
        marker = "▶ "
        suffix = f" ({hint})" if hint else ""
        return f"{marker}Мозг 1{suffix}"
    # slot 2 — voice/photo only, never chat.
    bits = [b for b in (hint, "voice/photo") if b]
    return f"Мозг 2 ({', '.join(bits)})"



def _auto_brain_label() -> str:
    """Dynamic label for the top-level «Авто мозг» entry.

    Combines the two slots into a single string so the user can see at
    a glance which providers are wired up: ``Авто мозг (OpenRouter +
    Groq)``, ``Авто мозг (OpenRouter)``, ``Авто мозг`` when nothing is
    set yet, etc.
    """
    brain = storage.get_brain()
    if brain != "auto":
        return "🧠 Авто мозг"
    hints: list[str] = []
    for slot in ("1", "2"):
        provider, api_key, base_url, _ = _slot_config(slot)
        if not api_key:
            continue
        h = _provider_hint_from(provider, base_url)
        if h and h not in hints:
            hints.append(h)
    if hints:
        return f"🧠 Авто мозг ({' + '.join(hints)})"
    return "🧠 Авто мозг"


def _kb_brain() -> InlineKeyboardMarkup:
    hb_label = (
        "💓 Heartbeat OpenRouter: ON"
        if storage.get_heartbeat_enabled()
        else "💤 Heartbeat OpenRouter: OFF"
    )
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=_auto_brain_label(), callback_data="brain:openrouter")],
            [InlineKeyboardButton(text="🎯 Devin.ai (ручные ответы через шелл)", callback_data="brain:devin")],
            [InlineKeyboardButton(text="⚙️  Другое (свой OpenAI-совместимый endpoint)", callback_data="brain:other")],
            [InlineKeyboardButton(text="🛠 Внешние API (Apify, Firecrawl, Tavily, ...)", callback_data="ext:menu")],
            [InlineKeyboardButton(text="🎬 Мозг для видео и фото (deAPI, fal, ...)", callback_data="media:menu")],
            [InlineKeyboardButton(text=hb_label, callback_data="brain:heartbeat_toggle")],
            [InlineKeyboardButton(text="← Назад", callback_data="brain:back")],
        ]
    )


def _kb_auto_brain() -> InlineKeyboardMarkup:
    """Submenu shown after tapping «🧠 Авто мозг».

    Exposes the two brain slots side-by-side so the user can configure
    OpenRouter on one and Groq on the other (or any pair). Slot 1 is
    always the chat brain (marker ``▶``); slot 2 is reserved for the
    voice/photo Groq override.
    """
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=f"🧠 {_slot_label('1')}", callback_data="brain_slot:1")],
            [InlineKeyboardButton(text=f"🧠 {_slot_label('2')}", callback_data="brain_slot:2")],
            [InlineKeyboardButton(text="← Вернуться", callback_data="auto_brain:back")],
        ]
    )


def _kb_roles_menu() -> InlineKeyboardMarkup:
    """Settings → «🎭 Роли» submenu — pick role + gate ON/OFF toggle.

    The gate is the historical «без выбранной роли проектная часть
    молчит» behaviour; it defaults to OFF now, so the user has to flip
    it on explicitly if they want it back.
    """
    enabled = storage.get_role_gate_enabled()
    toggle_label = (
        "🔒 Требование роли: ВКЛ" if enabled else "🔓 Требование роли: ВЫКЛ"
    )
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🎭 Сменить роль", callback_data="roles:pick")],
            [InlineKeyboardButton(text=toggle_label, callback_data="roles:toggle")],
            [InlineKeyboardButton(text="← Назад", callback_data="roles:back")],
        ]
    )


def _kb_tts_menu() -> InlineKeyboardMarkup:
    """Settings → «🔊 Голосовой ответчик» submenu — on/off + voice / record + back.

    Active mode gets a ▶ marker. Adds the «Выбрать голос», «Запись
    голоса» and «☁️ ElevenLabs» submenus the user asked for. The
    currently-used provider is marked with ✅ in its own button so
    the user always sees which source is speaking.
    """
    enabled = storage.get_tts_enabled()
    on_label = ("▶ 🔊 Авто включить" if enabled else "🔊 Авто включить")
    off_label = ("▶ 🔇 Выключить" if not enabled else "🔇 Выключить")
    provider = storage.get_tts_provider()
    voice_label = _voice_picker_button_label()
    record_label = _record_menu_button_label()
    if provider == "local":
        voice_label = "✅ " + voice_label
    elif provider == "clone":
        record_label = "✅ " + record_label
    el_marker = "✅ " if provider == "elevenlabs" else ""
    el_label = f"{el_marker}☁️ ElevenLabs (облачный TTS)"
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=on_label, callback_data="tts:on")],
            [InlineKeyboardButton(text=off_label, callback_data="tts:off")],
            [InlineKeyboardButton(text=voice_label, callback_data="tts:voice")],
            [InlineKeyboardButton(text=record_label, callback_data="tts:rec")],
            [InlineKeyboardButton(text=el_label, callback_data="tts:el")],
            [InlineKeyboardButton(text="← Назад", callback_data="tts:back")],
        ]
    )


def _provider_pretty(p: str) -> str:
    return {
        "local": "🎙 Встроенный (Supertonic)",
        "clone": "🎤 Записанный клон",
        "elevenlabs": "☁️ ElevenLabs",
    }.get(p, "🎙 Встроенный (Supertonic)")


def _kb_elevenlabs_menu() -> InlineKeyboardMarkup:
    """Settings → 🔊 Голосовой ответчик → ☁️ ElevenLabs.

    Three actions: paste API key, paste voice_id, mark this provider
    as active. URL is hardcoded to ``api.elevenlabs.io`` — no field.
    Back button label is «Перенастроить мозг» per user request, but
    it lands back on the TTS menu (one screen up).
    """
    provider = storage.get_tts_provider()
    has_key = bool(storage.get_elevenlabs_api_key())
    has_voice = bool(storage.get_elevenlabs_voice_id())
    api_label = "🔑 API ключ" + (" ✅" if has_key else "")
    voice_label = "🎤 Voice ID" + (" ✅" if has_voice else "")
    if provider == "elevenlabs":
        use_label = "✅ Используется"
    else:
        use_label = "▶ Использовать ElevenLabs"
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=api_label, callback_data="tts:el:api")],
            [InlineKeyboardButton(text=voice_label, callback_data="tts:el:voiceid")],
            [InlineKeyboardButton(text=use_label, callback_data="tts:el:use")],
            [InlineKeyboardButton(text="← Перенастроить мозг", callback_data="tts:el:back")],
        ]
    )


def _voice_picker_button_label() -> str:
    """Dynamic "М1"-style suffix so the user sees the current pick."""
    if storage.get_tts_custom_voice_path():
        return "🎙 Выбрать голос (сейчас: мой клон)"
    voice = storage.get_tts_voice() or "M1"
    return f"🎙 Выбрать голос (сейчас: {voice})"


def _record_menu_button_label() -> str:
    """Dynamic label — surfaces whether a clone is already saved."""
    if storage.get_tts_custom_voice_path():
        return "🎤 Запись голоса (клон готов)"
    return "🎤 Запись голоса"


def _kb_voice_picker_root() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="👨 Мужской", callback_data="tts:voice:male")],
            [InlineKeyboardButton(text="👩 Женский", callback_data="tts:voice:female")],
            [InlineKeyboardButton(text="← Вернуться", callback_data="tts:voice:back")],
        ]
    )


def _kb_voice_picker_list(gender: str) -> InlineKeyboardMarkup:
    """5 male voices (M1-M5) or 5 female voices (F1-F5) + back.

    The active voice gets a ▶ marker. An «▶ Использовать
    встроенные» / «✅ Используются» row lets the user mark the
    on-device Supertonic engine as the active TTS provider explicitly
    — important when ElevenLabs is also configured.
    """
    from .tts import BUILTIN_FEMALE_VOICES, BUILTIN_MALE_VOICES

    voices = BUILTIN_MALE_VOICES if gender == "male" else BUILTIN_FEMALE_VOICES
    current = storage.get_tts_voice() or "M1"
    has_custom = bool(storage.get_tts_custom_voice_path())
    provider = storage.get_tts_provider()
    rows: list[list[InlineKeyboardButton]] = []
    for v in voices:
        marker = "▶ " if (not has_custom and v == current) else ""
        rows.append(
            [
                InlineKeyboardButton(
                    text=f"{marker}{v}", callback_data=f"tts:voice:set:{v}"
                )
            ]
        )
    use_label = (
        "✅ Встроенные используются"
        if provider == "local"
        else "▶ Использовать встроенный голос"
    )
    rows.append([InlineKeyboardButton(text=use_label, callback_data="tts:use:local")])
    rows.append(
        [InlineKeyboardButton(text="← Вернуться", callback_data="tts:voice")]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _kb_record_menu() -> InlineKeyboardMarkup:
    """«Запись голоса» submenu — record / listen / save / import / export / VB.

    The Listen / Save / «Использовать клон» buttons only show once
    a recording exists so the user isn't tempted to click them on an
    empty slot. The «Использовать клон» button is what marks the
    Supertonic-clone path as the active TTS provider.
    """
    rec_path = storage.get_tts_custom_voice_path()
    has_recording = bool(rec_path)
    provider = storage.get_tts_provider()
    rows: list[list[InlineKeyboardButton]] = [
        [InlineKeyboardButton(text="🔴 Запись", callback_data="tts:rec:record")],
    ]
    if has_recording:
        rows.append(
            [InlineKeyboardButton(text="🔊 Прослушать", callback_data="tts:rec:listen")]
        )
        rows.append(
            [InlineKeyboardButton(text="💾 Сохранить как активный", callback_data="tts:rec:save")]
        )
        use_clone_label = (
            "✅ Клон используется"
            if provider == "clone"
            else "▶ Использовать клон"
        )
        rows.append(
            [InlineKeyboardButton(text=use_clone_label, callback_data="tts:use:clone")]
        )
    rows.extend([
        [InlineKeyboardButton(text="📥 Импорт (JSON)", callback_data="tts:rec:import")],
        [InlineKeyboardButton(text="📤 Экспорт (JSON)", callback_data="tts:rec:export")],
        [InlineKeyboardButton(text="🎶 Voice Builder", callback_data="tts:rec:vb")],
        [InlineKeyboardButton(text="← Вернуться", callback_data="tts:menu")],
    ])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _kb_main_after_claim(user_id: int | None = None) -> InlineKeyboardMarkup:
    """Top-level menu shown after ``/start`` once an owner exists.

    Layout depends on the caller's privileges:

    * Admins (owner, co-owners, full-public mode) see the full menu
      including settings-y actions: download, role, brain, tokens.
    * Guests (public mode, non-owner / non-co-owner) see ONLY the
      feature buttons; nothing that lets them change keys or models.

    ``⚙️ Настройки`` is rendered as the very last row per user
    request — easier to find by scrolling to the bottom of the menu.
    """
    rows: list[list[InlineKeyboardButton]] = [
        [InlineKeyboardButton(text="🤖 Helpzavr", callback_data="main:helpzavr")],
        [InlineKeyboardButton(text="✨ Красивый текст", callback_data="main:pretty")],
        [InlineKeyboardButton(text="📬 Проверка почты", callback_data="main:mailbox")],
        [InlineKeyboardButton(text="🎬 Генерация фото и видео", callback_data="main:media_toggle")],
        [InlineKeyboardButton(text="📝 Markitdown", callback_data="main:markitdown")],
        [InlineKeyboardButton(text="🐙 GitHub проекты", callback_data="main:github")],
        [InlineKeyboardButton(text="🖥 Терминал", callback_data="main:terminal")],
    ]
    if can_admin(user_id):
        rows.extend(
            [
                [InlineKeyboardButton(text="📥 Скачать видео", callback_data="main:download")],
                [InlineKeyboardButton(text="🖥 Комп (localdesktop)", callback_data="main:install:comp")],
                [InlineKeyboardButton(text="⬇️ Скачать (bash2mp4)", callback_data="main:install:downloader")],
                [InlineKeyboardButton(text="🧪 Анализатор (agent-skills)", callback_data="main:install:analyzer")],
                [InlineKeyboardButton(text="🎞 Монтажёр", callback_data="main:editor")],
                [InlineKeyboardButton(text="🧠 Перенастроить мозг", callback_data="main:brain")],
                [InlineKeyboardButton(text="📊 Статистика токенов (/tokens)", callback_data="main:tokens")],
            ]
        )
    # Settings always sits at the bottom — the user explicitly wants
    # it as the last item in the list so it doesn't get buried.
    rows.append(
        [InlineKeyboardButton(text="⚙️ Настройки", callback_data="main:settings")]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _kb_settings_menu(user_id: int | None) -> InlineKeyboardMarkup:
    """Shared keyboard for the ⚙️ Настройки screen.

    Centralised because the same set of rows is rendered from at least
    four entry points (``/settings`` command, ``main:settings``
    callback, and the «Назад» arrows on the Roles / TTS / Access
    sub-screens). Keeping one definition avoids the rows drifting out
    of sync when buttons get added or renamed.
    """
    rows: list[list[InlineKeyboardButton]] = [
        [InlineKeyboardButton(text="🎭 Роли", callback_data="roles:menu")],
        [InlineKeyboardButton(text="🔊 Голосовой ответчик", callback_data="tts:menu")],
        [InlineKeyboardButton(text="🧠 Соображалка", callback_data="main:thinking")],
        [InlineKeyboardButton(text="💾 Память бота", callback_data="main:memory")],
        [InlineKeyboardButton(text="💻 RAM", callback_data="main:ram")],
        [InlineKeyboardButton(text="🧩 Алгоритм", callback_data="algo:menu")],
        [InlineKeyboardButton(text="🔌 API (массовый импорт)", callback_data="api:batch")],
        [InlineKeyboardButton(text="🐙 GitHub проекты", callback_data="gh:menu")],
        [InlineKeyboardButton(text="🖥 Терминал", callback_data="main:terminal")],
        [InlineKeyboardButton(text="📚 Узнать о функционале", callback_data="main:learn")],
    ]
    if user_id is not None and can_admin(user_id):
        rows.append(
            [InlineKeyboardButton(text="⛔ Доступ", callback_data="access:menu")]
        )
    rows.append(
        [InlineKeyboardButton(text="↩️ В главное меню", callback_data="settings:back")]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


# Predefined install batches for the three main-menu shortcuts. Each entry
# maps the callback suffix (``main:install:<key>``) to a human-readable
# title and the list of ``/work``-style commands that will be executed
# sequentially when the user taps the button. Keep titles short — they
# show up in chat as a banner before the first command runs.
INSTALL_BATCHES: dict[str, tuple[str, list[str]]] = {
    "comp": (
        "🖥 Комп — localdesktop",
        [
            "/clone https://github.com/localdesktop/localdesktop",
        ],
    ),
    "downloader": (
        "⬇️ Скачать — bash2mp4",
        [
            "/clone https://github.com/htr-tech/bash2mp4",
            "bash setup.sh",
        ],
    ),
    "analyzer": (
        "🧪 Анализатор — agent-skills",
        [
            "/clone https://github.com/apify/agent-skills",
        ],
    ),
}


def _kb_role_picker() -> InlineKeyboardMarkup:
    """Grid of every persona + 'Info' + 'Back' buttons.

    Each persona button switches the bot's role on click (with a confirm
    step). 20 entries arranged 2 per row keeps the keyboard compact on
    mobile.
    """
    personas = list_personas()
    rows: list[list[InlineKeyboardButton]] = []
    active = get_persona().key
    for i in range(0, len(personas), 2):
        row: list[InlineKeyboardButton] = []
        for p in personas[i : i + 2]:
            marker = "▶ " if p.key == active else ""
            row.append(
                InlineKeyboardButton(
                    text=f"{marker}{p.display_name}",
                    callback_data=f"role:pick:{p.key}",
                )
            )
        rows.append(row)
    rows.append(
        [
            InlineKeyboardButton(text="ℹ️ Инфо о ролях", callback_data="role:info"),
            InlineKeyboardButton(text="◀️ Назад", callback_data="role:back"),
        ]
    )
    # If an override is active, show a button to revert to env-var default.
    if storage.get_persona_override() is not None:
        rows.append(
            [
                InlineKeyboardButton(
                    text="🗑 Сбросить override (вернуться к BOT_PERSONA)",
                    callback_data="role:reset",
                )
            ]
        )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _kb_role_confirm(persona_key: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="✅ Использовать эту роль",
                    callback_data=f"role:use:{persona_key}",
                )
            ],
            [InlineKeyboardButton(text="◀️ Назад", callback_data="role:back")],
        ]
    )


def _kb_external_tools() -> InlineKeyboardMarkup:
    """Menu of external research/scraping tools the bot may need to log in to.

    Reads the list from ``KNOWN_EXTERNAL_TOOLS`` in storage so adding a new
    tool there automatically grows the keyboard.
    """
    from .storage import KNOWN_EXTERNAL_TOOLS

    keys_state = storage.list_external_tool_keys()
    rows: list[list[InlineKeyboardButton]] = []
    for tool, meta in KNOWN_EXTERNAL_TOOLS.items():
        state = keys_state.get(tool, {})
        source = state.get("source", "none")
        marker = "✅" if source == "telegram" else ("🌍" if source == "env" else "⚪️")
        label = meta["label"]
        rows.append(
            [
                InlineKeyboardButton(
                    text=f"{marker} {label}",
                    callback_data=f"ext:set:{tool}",
                )
            ]
        )
    rows.append([InlineKeyboardButton(text="↩️  Назад к мозгам", callback_data="ext:back")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _kb_llm_config(provider_label: str) -> InlineKeyboardMarkup:
    """Sub-menu shown after OpenRouter / Other.

    All three fields are optional individually but at minimum API + Model
    are needed to make a request. The wizard does not enforce that — the
    first message to the bot will surface a clear error if something's
    missing.
    """
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=f"🔑 API ключ ({provider_label})", callback_data="cfg:api")],
            [InlineKeyboardButton(text="🤖 Модель", callback_data="cfg:model")],
            [InlineKeyboardButton(text="🌐 URL (только для Другое)", callback_data="cfg:url")],
            [InlineKeyboardButton(text="✅ Готово, поехали", callback_data="cfg:done")],
            [InlineKeyboardButton(text="← Назад", callback_data="cfg:back")],
        ]
    )


def _kb_brain_slot_cfg(slot: str) -> InlineKeyboardMarkup:
    """Brain-1 / Brain-2 config screen: same shape as `_kb_llm_config`.

    Difference: the bottom button is «← Вернуться» (back to the Авто
    мозг submenu) instead of «← Назад» (back to the brain picker), and
    «✅ Готово» also marks this slot as active for chat / failover.
    """
    provider, _, _, _ = _slot_config(slot)
    label = "OpenRouter" if provider == "openrouter" else "custom"
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=f"🔑 API ключ ({label})", callback_data="cfg:api")],
            [InlineKeyboardButton(text="🤖 Модель", callback_data="cfg:model")],
            [InlineKeyboardButton(text="🌐 URL (только для Другое / Groq)", callback_data="cfg:url")],
            [InlineKeyboardButton(text="✅ Готово, поехали", callback_data="cfg:done")],
            [InlineKeyboardButton(text="← Вернуться", callback_data="cfg:back")],
        ]
    )


# ---- Helpers -------------------------------------------------------------


def _is_owner(user_id: int) -> bool:
    """Backward-compatible alias for "can change settings".

    Originally checked ``storage.get_owner_id() == user_id``. Now it
    delegates to :func:`bot.access.can_admin` so that co-owners (added
    via the «➕ Дать доступ» flow) and users in ``full_public`` access
    mode pass the same gate. The original single-claim ``/start``
    handler still uses :func:`bot.access.is_owner` directly so a
    co-owner can't "re-claim" the slot.
    """
    return can_admin(user_id)


def _provider_label(brain_choice: str) -> str:
    return "openrouter" if brain_choice == "openrouter" else "custom"


# ---- Handlers ------------------------------------------------------------


@wizard_router.message(Command("start"))
async def cmd_start_wizard(message: Message, state: FSMContext) -> None:
    """Owner-aware /start.

    - Unclaimed: anyone may claim. Show big ``Запустить`` button.
    - Owner: re-enter the brain picker (``/setup`` is an alias).
    - Stranger: deny with their id so the *real* owner can spot a leak.
    """
    user = message.from_user
    if user is None:
        return
    await state.clear()

    persona = get_persona()
    owner = storage.get_owner_id()
    if owner is None:
        await message.answer(
            f"<b>Привет! Я {persona.display_name}.</b>\n"
            f"<i>{persona.title}</i>\n\n"
            f"{_html_escape(persona.description)}\n\n"
            "Ты только что развернул мой контейнер. Я не знаю кому теперь подчиняться — "
            "первый человек, кто нажмёт кнопку ниже, станет владельцем (только он сможет писать мне дальше).\n\n"
            f"Твой Telegram id: <code>{user.id}</code>",
            reply_markup=_kb_claim(),
        )
        return

    if not can_use(user.id):
        # Private mode + stranger → polite refusal with their id so the
        # real owner can spot a leak.
        await message.answer(
            f"Я {persona.display_name}, и я уже привязан к другому "
            "владельцу. Этот бот работает в приватном режиме — доступ "
            "только у владельца и тех, кому он его явно дал.\n\n"
            f"Твой Telegram id: <code>{user.id}</code> (можешь "
            "переслать владельцу — он сможет добавить тебя через "
            "<b>⚙️ Настройки → ⛔ Доступ → ➕ Дать доступ</b>)."
        )
        return

    override = storage.get_persona_override()
    extra = (
        f"\n<i>(роль переопределена через /role — env BOT_PERSONA={_html_escape(os.environ.get('BOT_PERSONA', 'boss'))})</i>"
        if override is not None
        else ""
    )
    is_admin = can_admin(user.id)
    if is_admin:
        role_label = "владелец" if is_owner(user.id) else "соучастник (доступ от владельца)"
        body = (
            f"<b>{persona.display_name}</b> на связи. Ты {role_label}.{extra}\n\n"
            f"<i>{persona.title}</i>\n\n"
            "Что хочешь? Жми кнопку или используй команду:\n"
            "• <code>/dl &lt;url&gt;</code> — скачать видео в конвейер\n"
            "• <code>/role</code> — сменить роль этого бота\n"
            "• <code>/setup</code> — перенастроить мозг\n"
            "• <code>/tokens</code> — статистика токенов\n"
            "• <code>/help</code> — все команды"
        )
    else:
        # Guest in public mode — feature buttons only, no admin links.
        body = (
            f"<b>{persona.display_name}</b> на связи. Ты гость "
            "(бот открыт владельцем в публичном режиме).\n\n"
            f"<i>{persona.title}</i>\n\n"
            "Можешь пользоваться функциями ниже. Настройки бота (API-"
            "ключи, модели, статистика, доступ) — только у владельца."
        )
    await message.answer(body, reply_markup=_kb_main_after_claim(user.id))


@wizard_router.message(Command("setup"))
async def cmd_setup(message: Message, state: FSMContext) -> None:
    """Re-open the brain picker (owner only)."""
    user = message.from_user
    if user is None or not _is_owner(user.id):
        await message.answer("Только владелец может перенастраивать бот.")
        return
    await state.clear()
    await message.answer(
        "Перенастройка. Выбери мозг:",
        reply_markup=_kb_brain(),
    )


# ---- Top-level slash commands surfaced via setMyCommands (menu button) ----
#
# The Telegram menu button (≡ next to the input field) shows these
# commands in a popup. Each one is a direct shortcut into the
# corresponding feature screen — same as tapping the matching button
# from /start, just faster. ``/start`` and ``/setup`` are handled
# elsewhere; this block covers the rest of the menu list.


async def _open_addon_screen(message: Message, addon_module: str, *attrs: str) -> None:
    """Best-effort: import an addon and call the first matching entry-point.

    Addons (helpzavr, pretty_text, mailbox, media_toggle) all expose a
    top-level screen function, but the name isn't consistent —
    ``show_screen`` in most, ``show_main`` in mailbox. We accept a
    list of candidate attribute names and use the first one that
    exists. Failure is logged and a graceful fallback message tells
    the user how to open the screen manually.
    """
    try:
        import importlib

        mod = importlib.import_module(addon_module)
        fn = None
        for attr in attrs:
            fn = getattr(mod, attr, None)
            if fn is not None:
                break
        if fn is None:
            raise AttributeError(f"{addon_module}: none of {attrs} present")
        await fn(message)
    except Exception:  # noqa: BLE001
        logger.exception("failed to open addon screen %s (%s)", addon_module, attrs)
        await message.answer(
            "Не получилось открыть экран. Зайди через /start → нужный пункт меню."
        )


def _deny_use(message: Message) -> None:
    """Send the standard "no access" refusal for guests in private mode."""
    return None


@wizard_router.message(Command("settings"))
async def cmd_settings(message: Message, state: FSMContext) -> None:
    """Jump straight to the ⚙️ Настройки screen."""
    user = message.from_user
    if user is None or not can_use(user.id):
        await message.answer("Нет доступа.")
        return
    await state.clear()
    await message.answer(
        "<b>⚙️ Настройки</b>",
        reply_markup=_kb_settings_menu(user.id),
    )


@wizard_router.message(Command("github"))
async def cmd_github(message: Message, state: FSMContext) -> None:
    """Jump straight to the 🐙 GitHub screen."""
    user = message.from_user
    if user is None or not can_use(user.id):
        await message.answer("Нет доступа.")
        return
    await state.clear()
    await message.answer(_github_menu_body(), reply_markup=_kb_github_menu())


@wizard_router.message(Command("chat"))
async def cmd_chat(message: Message, state: FSMContext) -> None:
    """Hint how to chat with the LLM directly — text-only feature, no UI."""
    user = message.from_user
    if user is None or not can_use(user.id):
        await message.answer("Нет доступа.")
        return
    await state.clear()
    await message.answer(
        "<b>💬 Чат с LLM</b>\n\n"
        "Просто пиши обычный текст в этот чат — без слешей. Я отвечу.\n\n"
        "Если кинешь GitHub-ссылку и попросишь что-то посмотреть — "
        "сам склонирую и почитаю (если в /github → 🔁 Авто-клонирование "
        "не выключено).\n\n"
        "Дополнительные команды над чатом:\n"
        "• /role — сменить роль бота\n"
        "• /tokens — статистика расхода токенов\n"
        "• /reset — очистить историю разговора\n"
        "• /help — все команды"
    )


@wizard_router.message(Command("helpzavr"))
async def cmd_helpzavr(message: Message, state: FSMContext) -> None:
    """Open the Helpzavr addon screen."""
    user = message.from_user
    if user is None or not can_use(user.id):
        await message.answer("Нет доступа.")
        return
    await state.clear()
    await _open_addon_screen(message, "bot.addons.helpzavr.handlers", "show_screen")


@wizard_router.message(Command("pretty"))
async def cmd_pretty(message: Message, state: FSMContext) -> None:
    """Open the «Красивый текст» addon screen."""
    user = message.from_user
    if user is None or not can_use(user.id):
        await message.answer("Нет доступа.")
        return
    await state.clear()
    await _open_addon_screen(message, "bot.addons.pretty_text.handlers", "show_screen")


@wizard_router.message(Command("mailbox"))
async def cmd_mailbox(message: Message, state: FSMContext) -> None:
    """Open the «Проверка почты» addon screen."""
    user = message.from_user
    if user is None or not can_use(user.id):
        await message.answer("Нет доступа.")
        return
    await state.clear()
    await _open_addon_screen(message, "bot.addons.mailbox.handlers", "show_main", "show_screen")


@wizard_router.message(Command("media"))
async def cmd_media(message: Message, state: FSMContext) -> None:
    """Open the media generation (📥 / 🎬) screen."""
    user = message.from_user
    if user is None or not can_use(user.id):
        await message.answer("Нет доступа.")
        return
    await state.clear()
    await _open_addon_screen(message, "bot.addons.media_toggle.handlers", "show_screen")


@wizard_router.message(Command("editor"))
async def cmd_editor(message: Message, state: FSMContext) -> None:
    """Open the Editor Agent («🎞 Монтажёр») config screen."""
    user = message.from_user
    if user is None or not can_use(user.id):
        await message.answer("Нет доступа.")
        return
    await state.clear()
    await _open_addon_screen(message, "bot.addons.editor_agent.handlers", "show_screen")


@wizard_router.callback_query(F.data == "claim_owner")
async def cb_claim(query: CallbackQuery, state: FSMContext) -> None:
    user = query.from_user
    existing = storage.get_owner_id()
    if existing is not None:
        await query.answer("Владелец уже зарегистрирован.", show_alert=True)
        return
    storage.set_owner_id(user.id)
    await state.clear()
    if query.message is not None:
        await query.message.edit_text(
            f"<b>Готово.</b> Ты владелец (id <code>{user.id}</code>).\n\n"
            "Теперь выбери, кто будет «мозгами» — кто отвечает на сообщения от тебя:",
            reply_markup=_kb_brain(),
        )
    await query.answer("Ты теперь владелец 🎉")


@wizard_router.callback_query(F.data.startswith("brain:"))
async def cb_brain(query: CallbackQuery, state: FSMContext) -> None:
    user = query.from_user
    if not _is_owner(user.id):
        await query.answer("Только владелец.", show_alert=True)
        return

    choice = (query.data or "").split(":", 1)[1]
    await state.clear()

    if choice == "back":
        # «← Назад» from the brain picker — return to the main menu so
        # the user isn't stranded on the brain screen with no exit.
        await query.answer()
        uid = user.id if user else None
        if query.message is not None:
            with contextlib.suppress(Exception):
                await query.message.edit_text(
                    "Главное меню. Выбери что сделать:",
                    reply_markup=_kb_main_after_claim(uid),
                )
        return

    if choice == "heartbeat_toggle":
        new_state = not storage.get_heartbeat_enabled()
        storage.set_heartbeat_enabled(new_state)
        await query.answer(
            f"Heartbeat OpenRouter: {'ON' if new_state else 'OFF'}", show_alert=False
        )
        if query.message is not None:
            with contextlib.suppress(Exception):
                await query.message.edit_reply_markup(reply_markup=_kb_brain())
        return

    if choice == "devin":
        # Switch to brain=devin, show the prompt the owner gives to a fresh Devin session.
        storage.set_brain("devin")
        await query.answer("Brain = devin")
        prompt = _devin_handoff_prompt()
        if query.message is not None:
            await query.message.edit_text(
                "<b>Brain = devin</b>. Бот не отвечает сам — он только пишет входящие в "
                "<code>data/inbox.log</code>, а ты сидишь в <a href='https://app.devin.ai'>app.devin.ai</a>, "
                "читаешь их и отвечаешь через шелл-CLI <code>python -m bot.send</code>.\n\n"
                "Скопируй блок ниже и пришли первым сообщением в новую Devin-сессию — там всё про "
                "архитектуру и команды:",
            )
            # Send the prompt as a separate code block so it copies cleanly on mobile.
            await query.message.answer(f"<pre>{_html_escape(prompt)}</pre>")
            await query.message.answer(
                "Готов? /help покажет все команды бота. /setup — поменять мозг."
            )
        return

    # "openrouter" choice now opens the two-slot «Авто мозг» submenu so
    # the user can configure Brain 1 and Brain 2 independently. The
    # legacy "other" choice ("Другое") still goes straight into the
    # single-slot config screen — its behaviour is intentionally
    # untouched so the Phase-3 guard can rely on it.
    if choice == "openrouter":
        storage.set_brain("auto")
        await query.answer("Brain = auto")
        if query.message is not None:
            await query.message.edit_text(
                "<b>🧠 Авто мозг</b>\n\n"
                "Можно держать два мозга одновременно — например, OpenRouter "
                "для текста и Groq для голоса / фото. Активный отмечен ▶. "
                "Если активный не отвечает — бот автоматически переходит "
                "на второй.\n\n"
                "Выбери слот для настройки:",
                reply_markup=_kb_auto_brain(),
            )
        return

    # "Другое" — single-slot custom endpoint, behaviour preserved from
    # the original bot. Provider=custom flips the Phase-3 guard so the
    # Groq voice/photo overrides stay quiet.
    provider = "custom"
    storage.set_brain("auto")
    storage.set_provider(provider)
    label = "Кастом (свой endpoint)"
    await query.answer(f"Brain = {label}")
    if query.message is not None:
        await query.message.edit_text(
            f"<b>Brain = auto / {label}</b>\n\n"
            "Заполни конфиг по очереди — что не задано, то возьмётся по умолчанию:",
            reply_markup=_kb_llm_config(_provider_label(choice)),
        )


# --- /ext:* — external research/scraping tools sub-menu ------------------


def _external_tools_summary() -> str:
    """Pretty list of all known external tools and which are configured."""
    keys_state = storage.list_external_tool_keys()
    lines = ["<b>Внешние API</b> (для research/scraping):", ""]
    for _tool, info in keys_state.items():
        marker = (
            "✅"
            if info["source"] == "telegram"
            else ("🌍" if info["source"] == "env" else "⚪️")
        )
        masked = info["masked"] or "—"
        lines.append(
            f"{marker} <b>{info['label']}</b> — <code>{masked}</code>\n"
            f"   <i>{info['hint']}</i>"
        )
    lines.append("")
    lines.append(
        "Жми на инструмент — пришлёшь ключ одним сообщением, я его сохраню и "
        "удалю твоё сообщение. <b>⚪️</b> = не настроено, "
        "<b>🌍</b> = взято из env-var, <b>✅</b> = задано здесь."
    )
    return "\n".join(lines)


@wizard_router.callback_query(F.data == "ext:menu")
async def cb_ext_menu(query: CallbackQuery, state: FSMContext) -> None:
    if not _is_owner(query.from_user.id):
        await query.answer("Только владелец.", show_alert=True)
        return
    await state.clear()
    if query.message is not None:
        await query.message.edit_text(
            _external_tools_summary(),
            reply_markup=_kb_external_tools(),
            disable_web_page_preview=True,
        )
    await query.answer()


@wizard_router.callback_query(F.data == "ext:back")
async def cb_ext_back(query: CallbackQuery, state: FSMContext) -> None:
    if not _is_owner(query.from_user.id):
        await query.answer("Только владелец.", show_alert=True)
        return
    await state.clear()
    if query.message is not None:
        await query.message.edit_text(
            "Выбери мозг:",
            reply_markup=_kb_brain(),
        )
    await query.answer()


@wizard_router.callback_query(F.data.startswith("ext:set:"))
async def cb_ext_set(query: CallbackQuery, state: FSMContext) -> None:
    if not _is_owner(query.from_user.id):
        await query.answer("Только владелец.", show_alert=True)
        return
    from .storage import KNOWN_EXTERNAL_TOOLS

    tool = (query.data or "").split(":", 2)[-1]
    if tool not in KNOWN_EXTERNAL_TOOLS:
        await query.answer("Неизвестный инструмент.", show_alert=True)
        return
    meta = KNOWN_EXTERNAL_TOOLS[tool]
    await state.set_state(SetupStates.awaiting_external_tool_key)
    await state.update_data(external_tool=tool)
    if query.message is not None:
        await query.message.answer(
            f"Пришли ключ для <b>{meta['label']}</b> одним сообщением. "
            "Я удалю твоё сообщение как только сохраню.\n\n"
            f"Получить: <a href='{meta['url']}'>{meta['url']}</a>\n\n"
            f"<i>{meta['hint']}</i>",
            disable_web_page_preview=True,
        )
    await query.answer()


# --- /cfg:* — sub-menu inside OpenRouter / Other --------------------------


@wizard_router.callback_query(F.data == "settings:back")
async def cb_settings_back(query: CallbackQuery, state: FSMContext) -> None:
    """Back arrow on the Settings sub-screen → re-render the main menu."""
    user = query.from_user
    if not can_use(user.id if user else None):
        await query.answer("Нет доступа.", show_alert=True)
        return
    await state.clear()
    uid = user.id if user else None
    if query.message is not None:
        try:
            await query.message.edit_text(
                "Главное меню. Выбери что сделать:",
                reply_markup=_kb_main_after_claim(uid),
            )
        except Exception:  # noqa: BLE001
            await query.message.answer(
                "Главное меню. Выбери что сделать:",
                reply_markup=_kb_main_after_claim(uid),
            )
    await query.answer()


# ---- /roles:* — Settings → «🎭 Роли» submenu ----------------------------


def _roles_screen_body() -> str:
    """Top text for the «🎭 Роли» submenu — current persona + gate state."""
    persona = get_persona()
    override = storage.get_persona_override()
    source = "override (через /role)" if override is not None else "env BOT_PERSONA"
    gate_on = storage.get_role_gate_enabled()
    if gate_on:
        gate_text = (
            "<b>ВКЛ</b> — без выбранной роли /work, /exec, /clone и кнопки "
            "установки молчат."
        )
    else:
        gate_text = (
            "<b>ВЫКЛ</b> — все команды работают без выбора роли "
            "(по умолчанию)."
        )
    return (
        "<b>🎭 Роли</b>\n\n"
        f"Сейчас активна: <b>{persona.display_name}</b> "
        f"(<code>{persona.key}</code>, {source})\n"
        f"<i>{persona.title}</i>\n\n"
        f"<b>Требование роли:</b> {gate_text}"
    )


@wizard_router.callback_query(F.data == "roles:menu")
async def cb_roles_menu(query: CallbackQuery, state: FSMContext) -> None:
    """Open Settings → «🎭 Роли» submenu."""
    if not _is_owner(query.from_user.id if query.from_user else 0):
        await query.answer("Только владелец.", show_alert=True)
        return
    await state.clear()
    if query.message is not None:
        with contextlib.suppress(Exception):
            await query.message.edit_text(
                _roles_screen_body(), reply_markup=_kb_roles_menu()
            )
    await query.answer()


@wizard_router.callback_query(F.data == "roles:pick")
async def cb_roles_pick(query: CallbackQuery, state: FSMContext) -> None:
    """«Сменить роль» — open the existing role picker UI."""
    if not _is_owner(query.from_user.id if query.from_user else 0):
        await query.answer("Только владелец.", show_alert=True)
        return
    await state.clear()
    await query.answer()
    if query.message is not None:
        with contextlib.suppress(Exception):
            await query.message.edit_text(
                _role_picker_text(), reply_markup=_kb_role_picker()
            )


@wizard_router.callback_query(F.data == "roles:toggle")
async def cb_roles_toggle(query: CallbackQuery, state: FSMContext) -> None:
    """Flip the role gate ON ↔ OFF and re-render the submenu."""
    if not _is_owner(query.from_user.id if query.from_user else 0):
        await query.answer("Только владелец.", show_alert=True)
        return
    new_state = not storage.get_role_gate_enabled()
    storage.set_role_gate_enabled(new_state)
    await query.answer(
        f"Требование роли: {'ВКЛ' if new_state else 'ВЫКЛ'}", show_alert=False
    )
    if query.message is not None:
        with contextlib.suppress(Exception):
            await query.message.edit_text(
                _roles_screen_body(), reply_markup=_kb_roles_menu()
            )


@wizard_router.callback_query(F.data == "roles:back")
async def cb_roles_back(query: CallbackQuery, state: FSMContext) -> None:
    """«← Назад» from Роли submenu → re-render the Settings screen."""
    user = query.from_user
    if not can_use(user.id if user else None):
        await query.answer("Нет доступа.", show_alert=True)
        return
    await state.clear()
    uid = user.id if user else None
    if query.message is not None:
        with contextlib.suppress(Exception):
            await query.message.edit_text(
                "<b>⚙️ Настройки</b>",
                reply_markup=_kb_settings_menu(uid),
            )
    await query.answer()


# ---- /tts:* — Settings → «🔊 Голосовой ответчик» submenu --------------


def _tts_screen_body() -> str:
    """Top text for the TTS submenu — current state + short explainer."""
    enabled = storage.get_tts_enabled()
    state_line = (
        "<b>Сейчас:</b> 🔊 ВКЛЮЧЕНО — каждый текстовый ответ бота "
        "приходит ещё и голосом."
        if enabled
        else "<b>Сейчас:</b> 🔇 ВЫКЛЮЧЕНО — бот отвечает только текстом."
    )
    # Show which voice / provider is currently active so the user can
    # verify the picker actually took effect.
    provider = storage.get_tts_provider()
    custom = storage.get_tts_custom_voice_path()
    voice = storage.get_tts_voice() or "M1"
    if provider == "elevenlabs":
        active_line = "<b>Активный голос:</b> ☁️ ElevenLabs (облако)."
    elif provider == "clone" and custom:
        active_line = (
            "<b>Активный голос:</b> 🎤 кастомный клон (Supertonic)."
        )
    else:
        active_line = (
            f"<b>Активный голос:</b> 🎙 <code>{voice}</code> (Supertonic)"
        )
    # Surface why TTS doesn't work when the supertonic install is
    # broken (e.g. missing ffmpeg / model download failed). Without
    # this hint the user has no idea why the voice never plays.
    try:
        from .tts import get_unavailable_reason

        reason = get_unavailable_reason()
    except Exception:  # noqa: BLE001
        reason = None
    warn_line = (
        f"\n\n⚠️ <b>TTS недоступен:</b> {_html_escape(reason)}"
        if reason
        else ""
    )
    return (
        "<b>🔊 Голосовой ответчик</b>\n\n"
        f"{state_line}\n"
        f"{active_line}{warn_line}\n\n"
        "Озвучивает то, что бот сам тебе пишет (через on-device "
        "Supertonic TTS). Код, html-тэги и сырые ссылки в озвучке "
        "пропускаются; для github-ссылок читается «репо от создателей "
        "&lt;имя&gt;». Бот сам ничего в интернете не ищет — читает "
        "только то, что он уже сформулировал.\n\n"
        "Выбрал голос или импортировал JSON — озвучка включается "
        "автоматически. Хочешь выключить — жми «🔇 Выключить»."
    )


@wizard_router.callback_query(F.data == "tts:menu")
async def cb_tts_menu(query: CallbackQuery, state: FSMContext) -> None:
    """Open Settings → «🔊 Голосовой ответчик» submenu."""
    if not _is_owner(query.from_user.id if query.from_user else 0):
        await query.answer("Только владелец.", show_alert=True)
        return
    await state.clear()
    if query.message is not None:
        with contextlib.suppress(Exception):
            await query.message.edit_text(
                _tts_screen_body(), reply_markup=_kb_tts_menu()
            )
    await query.answer()


@wizard_router.callback_query(F.data == "tts:on")
async def cb_tts_on(query: CallbackQuery, state: FSMContext) -> None:
    """«Авто включить» — enable narration and re-render the submenu."""
    if not _is_owner(query.from_user.id if query.from_user else 0):
        await query.answer("Только владелец.", show_alert=True)
        return
    storage.set_tts_enabled(True)
    await query.answer("Голосовой ответчик: ВКЛ", show_alert=False)
    if query.message is not None:
        with contextlib.suppress(Exception):
            await query.message.edit_text(
                _tts_screen_body(), reply_markup=_kb_tts_menu()
            )


@wizard_router.callback_query(F.data == "tts:off")
async def cb_tts_off(query: CallbackQuery, state: FSMContext) -> None:
    """«Выключить» — disable narration and re-render the submenu."""
    if not _is_owner(query.from_user.id if query.from_user else 0):
        await query.answer("Только владелец.", show_alert=True)
        return
    storage.set_tts_enabled(False)
    await query.answer("Голосовой ответчик: ВЫКЛ", show_alert=False)
    if query.message is not None:
        with contextlib.suppress(Exception):
            await query.message.edit_text(
                _tts_screen_body(), reply_markup=_kb_tts_menu()
            )


@wizard_router.callback_query(F.data == "tts:back")
async def cb_tts_back(query: CallbackQuery, state: FSMContext) -> None:
    """«← Назад» from TTS submenu → re-render the Settings screen."""
    user = query.from_user
    if not can_use(user.id if user else None):
        await query.answer("Нет доступа.", show_alert=True)
        return
    await state.clear()
    uid = user.id if user else None
    if query.message is not None:
        with contextlib.suppress(Exception):
            await query.message.edit_text(
                "<b>⚙️ Настройки</b>",
                reply_markup=_kb_settings_menu(uid),
            )
    await query.answer()


# ---- /tts:use:* — provider switch (local / clone / elevenlabs) -----------


@wizard_router.callback_query(F.data.startswith("tts:use:"))
async def cb_tts_use(query: CallbackQuery, state: FSMContext) -> None:
    """Mark a TTS provider as active.

    Three callbacks: ``tts:use:local`` (Supertonic built-in voices),
    ``tts:use:clone`` (Supertonic clone), ``tts:use:elevenlabs``
    (ElevenLabs HTTP). Auto-enables narration so the user instantly
    hears the change instead of having to flip the on/off toggle too.
    """
    if not _is_owner(query.from_user.id if query.from_user else 0):
        await query.answer("Только владелец.", show_alert=True)
        return
    provider = (query.data or "").split(":")[-1]
    if provider not in ("local", "clone", "elevenlabs"):
        await query.answer("Неизвестный провайдер.", show_alert=True)
        return
    # Guard against picking "clone" without a saved clone — otherwise
    # storage.get_tts_provider() will silently downgrade to "local"
    # and the user will be confused.
    if provider == "clone" and not storage.get_tts_custom_voice_path():
        await query.answer(
            "Сначала запиши голос или импортируй JSON клона.",
            show_alert=True,
        )
        return
    if provider == "elevenlabs" and (
        not storage.get_elevenlabs_api_key()
        or not storage.get_elevenlabs_voice_id()
    ):
        await query.answer(
            "Сначала пришли API ключ ElevenLabs и Voice ID.",
            show_alert=True,
        )
        return
    storage.set_tts_provider(provider)
    storage.set_tts_enabled(True)
    label = {
        "local": "встроенный голос (Supertonic)",
        "clone": "клонированный голос",
        "elevenlabs": "ElevenLabs",
    }[provider]
    await query.answer(f"Используется: {label}", show_alert=False)
    # Re-render the same screen so the ✅ marker updates immediately.
    if query.message is not None:
        with contextlib.suppress(Exception):
            await query.message.edit_text(
                _tts_screen_body(),
                reply_markup=_kb_tts_menu(),
            )


# ---- /tts:el:* — ☁️ ElevenLabs subsection --------------------------------


def _elevenlabs_screen_body() -> str:
    """Top text for the ElevenLabs subsection — shows what's configured."""
    api_set = bool(storage.get_elevenlabs_api_key())
    voice_id = storage.get_elevenlabs_voice_id()
    provider = storage.get_tts_provider()
    lines = [
        "<b>☁️ ElevenLabs</b> — облачный TTS.",
        "",
        f"🔑 API ключ: {'✅ задан' if api_set else '❌ не задан'}",
        (
            f"🎤 Voice ID: <code>{_html_escape(voice_id)}</code>"
            if voice_id
            else "🎤 Voice ID: ❌ не задан"
        ),
        (
            "▶ Используется: <b>да</b>"
            if provider == "elevenlabs"
            else "▶ Используется: нет"
        ),
        "",
        "Voice ID берётся из <a href='https://elevenlabs.io/app/voice-library'>"
        "Voice Library</a> — открой нужный голос и скопируй id (длинная "
        "hex-строка типа <code>21m00Tcm4TlvDq8ikWAM</code>). URL хардкод "
        "<code>https://api.elevenlabs.io</code>, поле тебе вводить не нужно.",
    ]
    return "\n".join(lines)


@wizard_router.callback_query(F.data == "tts:el")
async def cb_tts_el_menu(query: CallbackQuery, state: FSMContext) -> None:
    """Open Settings → 🔊 Голосовой ответчик → ☁️ ElevenLabs."""
    if not _is_owner(query.from_user.id if query.from_user else 0):
        await query.answer("Только владелец.", show_alert=True)
        return
    await state.clear()
    if query.message is not None:
        with contextlib.suppress(Exception):
            await query.message.edit_text(
                _elevenlabs_screen_body(),
                reply_markup=_kb_elevenlabs_menu(),
                disable_web_page_preview=True,
            )
    await query.answer()


@wizard_router.callback_query(F.data == "tts:el:api")
async def cb_tts_el_api(query: CallbackQuery, state: FSMContext) -> None:
    if not _is_owner(query.from_user.id if query.from_user else 0):
        await query.answer("Только владелец.", show_alert=True)
        return
    await state.set_state(SetupStates.awaiting_elevenlabs_api_key)
    if query.message is not None:
        await query.message.answer(
            "Пришли <b>API ключ ElevenLabs</b> одним сообщением.\n\n"
            "Где взять: <a href='https://elevenlabs.io/app/settings/api-keys'>"
            "elevenlabs.io/app/settings/api-keys</a>. Я удалю твоё сообщение "
            "сразу как сохраню ключ, чтобы он не остался в истории чата."
        )
    await query.answer()


@wizard_router.callback_query(F.data == "tts:el:voiceid")
async def cb_tts_el_voiceid(query: CallbackQuery, state: FSMContext) -> None:
    if not _is_owner(query.from_user.id if query.from_user else 0):
        await query.answer("Только владелец.", show_alert=True)
        return
    await state.set_state(SetupStates.awaiting_elevenlabs_voice_id)
    if query.message is not None:
        await query.message.answer(
            "Пришли <b>Voice ID</b> одним сообщением.\n\n"
            "Где взять: <a href='https://elevenlabs.io/app/voice-library'>"
            "elevenlabs.io/app/voice-library</a> — открой нужный голос, "
            "там кнопка «Copy ID». Формат: длинная hex-строка типа "
            "<code>21m00Tcm4TlvDq8ikWAM</code>."
        )
    await query.answer()


@wizard_router.callback_query(F.data == "tts:el:use")
async def cb_tts_el_use(query: CallbackQuery, state: FSMContext) -> None:
    """Switch the active TTS provider to ElevenLabs.

    Refuses if API key or Voice ID is still empty so we don't trigger
    a silent fallback to Supertonic right after the user clicked
    «использовать».
    """
    if not _is_owner(query.from_user.id if query.from_user else 0):
        await query.answer("Только владелец.", show_alert=True)
        return
    if not storage.get_elevenlabs_api_key() or not storage.get_elevenlabs_voice_id():
        await query.answer(
            "Сначала задай API ключ и Voice ID — иначе озвучка не сработает.",
            show_alert=True,
        )
        return
    storage.set_tts_provider("elevenlabs")
    storage.set_tts_enabled(True)
    await query.answer("ElevenLabs активирован", show_alert=False)
    if query.message is not None:
        with contextlib.suppress(Exception):
            await query.message.edit_text(
                _elevenlabs_screen_body(),
                reply_markup=_kb_elevenlabs_menu(),
                disable_web_page_preview=True,
            )


@wizard_router.callback_query(F.data == "tts:el:back")
async def cb_tts_el_back(query: CallbackQuery, state: FSMContext) -> None:
    """«← Перенастроить мозг» — go up one level to the TTS submenu."""
    if not _is_owner(query.from_user.id if query.from_user else 0):
        await query.answer("Только владелец.", show_alert=True)
        return
    await state.clear()
    if query.message is not None:
        with contextlib.suppress(Exception):
            await query.message.edit_text(
                _tts_screen_body(), reply_markup=_kb_tts_menu()
            )
    await query.answer()


# ---- /api:batch — «🔌 API» mass-import button ----------------------------


def _api_batch_prompt() -> str:
    """Help text shown when the user opens the batch-import flow."""
    return (
        "<b>🔌 API — массовый импорт</b>\n\n"
        "Пришли одной строкой все ключи которые хочешь загрузить. "
        "Разделители:\n"
        "• <code>$$$</code> — между независимыми блоками\n"
        "• <code>;</code> — между полями в одном блоке\n"
        "• <code>:</code> — между target / key / value\n\n"
        "<b>Targets:</b> <code>brain1</code>, <code>brain2</code> (поля: "
        "<code>api</code>, <code>model</code>, <code>url</code>, "
        "<code>provider</code>), <code>elevenlabs</code> (<code>api</code>, "
        "<code>voice</code>), <code>tavily</code>, <code>brave</code>, "
        "<code>exa</code>, <code>firecrawl</code>, <code>apify</code>, "
        "<code>github</code> / <code>github_pat</code>.\n\n"
        "<b>Пример:</b>\n"
        "<code>brain1:provider:openrouter;brain1:api:sk-or-xxx;"
        "brain1:model:anthropic/claude-3.5-sonnet$$$brain2:provider:"
        "custom;brain2:api:gsk_yyy;brain2:model:whisper-large-v3;"
        "brain2:url:https://api.groq.com/openai/v1$$$tavily:api:tvly-zzz"
        "$$$elevenlabs:api:el-www;elevenlabs:voice:21m00Tcm4TlvDq8ikWAM</code>\n\n"
        "Я применю всё что распознаю, сообщения с ключами удалю из "
        "истории чата. /cancel — отмена."
    )


@wizard_router.callback_query(F.data == "api:batch")
async def cb_api_batch(query: CallbackQuery, state: FSMContext) -> None:
    """Open the 🔌 API mass-import flow."""
    if not _is_owner(query.from_user.id if query.from_user else 0):
        await query.answer("Только владелец.", show_alert=True)
        return
    await state.set_state(SetupStates.awaiting_batch_api_blob)
    if query.message is not None:
        await query.message.answer(
            _api_batch_prompt(), disable_web_page_preview=True
        )
    await query.answer()


@wizard_router.message(
    StateFilter(SetupStates.awaiting_batch_api_blob), _NOT_A_COMMAND
)
async def capture_batch_api_blob(message: Message, state: FSMContext) -> None:
    """Parse the pasted blob and apply every recognised field.

    Returns a single summary banner: applied vs. errors. Deletes the
    user's message so the secrets never sit in chat history.
    """
    if not _is_owner(message.from_user.id if message.from_user else 0):
        return
    blob = (message.text or "").strip()
    if not blob:
        await message.answer("Пустое сообщение, попробуй ещё раз.")
        return
    with contextlib.suppress(Exception):
        await message.delete()
    from .api_batch import parse_and_apply

    report = parse_and_apply(blob, storage)
    await state.clear()
    lines: list[str] = ["<b>🔌 API — массовый импорт</b>", ""]
    if report.applied:
        lines.append("✅ Применено: " + ", ".join(
            f"<code>{a}</code>" for a in report.applied
        ))
    else:
        lines.append("⚠️ Ничего не применено.")
    if report.errors:
        lines.append("")
        lines.append("Ошибки:")
        for err in report.errors:
            lines.append(f"• {err}")
    await message.answer(
        "\n".join(lines),
        reply_markup=_kb_settings_menu(
            message.from_user.id if message.from_user else None
        ),
        disable_web_page_preview=True,
    )


# ---- /tts:voice:* — voice picker submenu (M1..M5 / F1..F5) ---------------


def _voice_picker_body() -> str:
    current_voice = storage.get_tts_voice() or "M1"
    custom = storage.get_tts_custom_voice_path()
    active = (
        "ваш клонированный голос (JSON)"
        if custom
        else f"<code>{current_voice}</code>"
    )
    return (
        "<b>🎙 Выбрать голос</b>\n\n"
        f"Сейчас активен: {active}.\n\n"
        "В Supertonic 5 мужских голосов (M1–M5) и 5 женских (F1–F5). "
        "Выбери раздел — там 5 кнопок-семплов."
    )


@wizard_router.callback_query(F.data == "tts:voice")
async def cb_tts_voice_root(query: CallbackQuery, state: FSMContext) -> None:
    if not _is_owner(query.from_user.id if query.from_user else 0):
        await query.answer("Только владелец.", show_alert=True)
        return
    await state.clear()
    if query.message is not None:
        with contextlib.suppress(Exception):
            await query.message.edit_text(
                _voice_picker_body(), reply_markup=_kb_voice_picker_root()
            )
    await query.answer()


@wizard_router.callback_query(F.data == "tts:voice:back")
async def cb_tts_voice_back(query: CallbackQuery, state: FSMContext) -> None:
    if not _is_owner(query.from_user.id if query.from_user else 0):
        await query.answer("Только владелец.", show_alert=True)
        return
    await state.clear()
    if query.message is not None:
        with contextlib.suppress(Exception):
            await query.message.edit_text(
                _tts_screen_body(), reply_markup=_kb_tts_menu()
            )
    await query.answer()


@wizard_router.callback_query(F.data.in_({"tts:voice:male", "tts:voice:female"}))
async def cb_tts_voice_list(query: CallbackQuery, state: FSMContext) -> None:
    if not _is_owner(query.from_user.id if query.from_user else 0):
        await query.answer("Только владелец.", show_alert=True)
        return
    await state.clear()
    gender = (query.data or "").split(":")[-1]
    title = "👨 Мужской" if gender == "male" else "👩 Женский"
    body = (
        f"<b>{title}</b>\n\n"
        "Жми на кнопку — этот голос станет активным. После выбора можешь "
        "включить «🔊 Авто включить», и бот будет отвечать им."
    )
    if query.message is not None:
        with contextlib.suppress(Exception):
            await query.message.edit_text(
                body, reply_markup=_kb_voice_picker_list(gender)
            )
    await query.answer()


@wizard_router.callback_query(F.data.startswith("tts:voice:set:"))
async def cb_tts_voice_set(query: CallbackQuery, state: FSMContext) -> None:
    if not _is_owner(query.from_user.id if query.from_user else 0):
        await query.answer("Только владелец.", show_alert=True)
        return
    voice = (query.data or "").split(":")[-1].upper()

    from .tts import BUILTIN_VOICES

    if voice not in BUILTIN_VOICES:
        await query.answer("Неизвестный голос.", show_alert=True)
        return
    storage.set_tts_voice(voice)
    # Picking a built-in voice clears any custom Voice-Builder clone so
    # the next `synthesize_voice_ogg()` call picks up the built-in one,
    # and snaps the active provider back to ``local`` so we don't keep
    # routing through ElevenLabs after the user clicked a Supertonic
    # voice button.
    storage.clear_tts_custom_voice_path()
    storage.set_tts_provider("local")
    # Auto-enable the narrator — the user just told us "this is the
    # voice I want", so it's weird to make them flip a separate
    # toggle to actually hear it. Previously the menu accepted the
    # pick silently and the user thought "the voice doesn't work"
    # because TTS was still OFF by default.
    was_disabled = not storage.get_tts_enabled()
    storage.set_tts_enabled(True)
    notice = (
        f"Голос: {voice} · 🔊 Авто включить → ВКЛ"
        if was_disabled
        else f"Голос: {voice}"
    )
    await query.answer(notice, show_alert=False)
    # Re-render the picker list to update the ▶ marker.
    gender = "male" if voice.startswith("M") else "female"
    title = "👨 Мужской" if gender == "male" else "👩 Женский"
    if query.message is not None:
        with contextlib.suppress(Exception):
            await query.message.edit_text(
                f"<b>{title}</b>\n\nАктивный голос: <code>{voice}</code>\n"
                "🔊 Авто включить: <b>ВКЛ</b>",
                reply_markup=_kb_voice_picker_list(gender),
            )


# ---- /tts:rec:* — «Запись голоса» submenu --------------------------------


def _voice_dir() -> Path:
    """Where user voice recordings/JSON clones live."""
    from .config import DATA_DIR

    d = DATA_DIR / "tts_voices"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _user_recording_audio_path(user_id: int) -> Path:
    return _voice_dir() / f"{user_id}_recording.ogg"


def _user_voice_json_path(user_id: int) -> Path:
    return _voice_dir() / f"{user_id}_voice.json"


def _record_menu_body(user_id: int) -> str:
    rec_audio = _user_recording_audio_path(user_id)
    rec_json = _user_voice_json_path(user_id)
    has_audio = rec_audio.exists()
    has_json = rec_json.exists()
    saved = bool(storage.get_tts_custom_voice_path())
    lines = ["<b>🎤 Запись голоса (клонирование)</b>", ""]
    if has_audio:
        lines.append("🎵 Аудио-сэмпл записан.")
    else:
        lines.append("🎵 Аудио-сэмпла ещё нет.")
    if has_json:
        lines.append("📄 JSON-стиль голоса лежит на диске.")
    else:
        lines.append("📄 JSON-стиля голоса ещё нет.")
    if saved:
        lines.append("✅ Активный голос — клонированный.")
    lines.extend([
        "",
        "Как сделать свой голос:",
        "1. Нажми <b>🔴 Запись</b> и пришли voice/audio (10–60 сек чистой речи).",
        "2. Нажми <b>🎶 Voice Builder</b> — там инструкция, как из этого аудио "
        "получить JSON-стиль (бесплатный веб-сервис Supertonic).",
        "3. Нажми <b>📥 Импорт (JSON)</b> и пришли полученный JSON.",
        "4. <b>🔊 Прослушать</b> — синтезирует фразу твоим голосом.",
        "5. <b>💾 Сохранить как активный</b> — бот будет говорить твоим голосом.",
        "",
        "Уже есть JSON от Voice Builder? Жми <b>📥 Импорт</b> сразу.",
    ])
    return "\n".join(lines)


@wizard_router.callback_query(F.data == "tts:rec")
async def cb_tts_rec_menu(query: CallbackQuery, state: FSMContext) -> None:
    if not _is_owner(query.from_user.id if query.from_user else 0):
        await query.answer("Только владелец.", show_alert=True)
        return
    await state.clear()
    if query.message is not None:
        with contextlib.suppress(Exception):
            await query.message.edit_text(
                _record_menu_body(query.from_user.id),
                reply_markup=_kb_record_menu(),
            )
    await query.answer()


@wizard_router.callback_query(F.data == "tts:rec:record")
async def cb_tts_rec_record(query: CallbackQuery, state: FSMContext) -> None:
    if not _is_owner(query.from_user.id if query.from_user else 0):
        await query.answer("Только владелец.", show_alert=True)
        return
    await state.set_state(SetupStates.awaiting_voice_recording)
    if query.message is not None:
        await query.message.answer(
            "🎙 Жду аудио. Пришли voice-сообщение или файл (mp3 / m4a / ogg / wav). "
            "Чем чище запись и чем спокойнее темп речи — тем лучше склонируется. "
            "Длительность 10–60 секунд. Чтобы отменить — /cancel."
        )
    await query.answer()


@wizard_router.callback_query(F.data == "tts:rec:listen")
async def cb_tts_rec_listen(query: CallbackQuery, state: FSMContext) -> None:
    if not _is_owner(query.from_user.id if query.from_user else 0):
        await query.answer("Только владелец.", show_alert=True)
        return
    json_path = _user_voice_json_path(query.from_user.id)
    if not json_path.exists():
        await query.answer(
            "Сначала импортируй JSON или сохрани клон.", show_alert=True
        )
        return
    await state.set_state(SetupStates.awaiting_voice_preview_text)
    await state.update_data(preview_voice_json=str(json_path))
    if query.message is not None:
        await query.message.answer(
            "📝 Пришли текст — я озвучу его твоим клонированным голосом. "
            "Чтобы отменить — /cancel."
        )
    await query.answer()


@wizard_router.callback_query(F.data == "tts:rec:save")
async def cb_tts_rec_save(query: CallbackQuery, state: FSMContext) -> None:
    if not _is_owner(query.from_user.id if query.from_user else 0):
        await query.answer("Только владелец.", show_alert=True)
        return
    json_path = _user_voice_json_path(query.from_user.id)
    if not json_path.exists():
        await query.answer(
            "Сначала импортируй JSON или склонируй голос.", show_alert=True
        )
        return
    storage.set_tts_custom_voice_path(str(json_path))
    # Auto-enable narration so saving instantly translates into
    # spoken replies — same UX promise as «🔊 Авто включить» — and
    # mark the clone path as the active TTS provider so ElevenLabs
    # doesn't accidentally keep playing after the save.
    storage.set_tts_enabled(True)
    storage.set_tts_provider("clone")
    await query.answer(
        "Клонированный голос активирован · 🔊 Авто включить → ВКЛ",
        show_alert=False,
    )
    if query.message is not None:
        with contextlib.suppress(Exception):
            await query.message.edit_text(
                _record_menu_body(query.from_user.id),
                reply_markup=_kb_record_menu(),
            )


@wizard_router.callback_query(F.data == "tts:rec:export")
async def cb_tts_rec_export(query: CallbackQuery, state: FSMContext) -> None:
    if not _is_owner(query.from_user.id if query.from_user else 0):
        await query.answer("Только владелец.", show_alert=True)
        return
    json_path = _user_voice_json_path(query.from_user.id)
    if not json_path.exists():
        await query.answer(
            "JSON клона ещё не создан — сначала импортируй его.",
            show_alert=True,
        )
        return
    from aiogram.types import FSInputFile

    file = FSInputFile(str(json_path), filename=f"voice_{query.from_user.id}.json")
    if query.message is not None:
        await query.message.answer_document(
            file,
            caption=(
                "📤 Экспорт твоего голоса. Этот JSON можно импортировать в "
                "другом боте — там кнопка «📥 Импорт» в этом же меню."
            ),
        )
    await query.answer("Экспортировал.", show_alert=False)


@wizard_router.callback_query(F.data == "tts:rec:import")
async def cb_tts_rec_import(query: CallbackQuery, state: FSMContext) -> None:
    if not _is_owner(query.from_user.id if query.from_user else 0):
        await query.answer("Только владелец.", show_alert=True)
        return
    await state.set_state(SetupStates.awaiting_voice_json_import)
    if query.message is not None:
        await query.message.answer(
            "📥 Жду JSON-файл с голосом (то, что выдаёт Voice Builder либо "
            "экспортирован из другого бота). Просто пришли документ. "
            "Чтобы отменить — /cancel."
        )
    await query.answer()


@wizard_router.callback_query(F.data == "tts:rec:vb")
async def cb_tts_rec_vb(query: CallbackQuery, state: FSMContext) -> None:
    if not _is_owner(query.from_user.id if query.from_user else 0):
        await query.answer("Только владелец.", show_alert=True)
        return
    body = (
        "<b>🎶 Voice Builder — как сделать свой голос-JSON</b>\n\n"
        "Supertonic умеет говорить любым голосом, но сам аудио → JSON "
        "локально не конвертирует. Для этого есть бесплатный веб-сервис:\n\n"
        "🔗 <a href=\"https://supertonic.supertone.ai/voice_builder\">supertonic.supertone.ai/voice_builder</a>\n\n"
        "Шаги:\n"
        "1. Открой ссылку выше в браузере.\n"
        "2. Нажми «<b>+ New Voice</b>», дай имя.\n"
        "3. Загрузи свой аудио-сэмпл (10–60 сек, моно, без шума).\n"
        "   Если ты уже прислал сюда запись — она лежит в "
        "<code>data/tts_voices/&lt;твой_id&gt;_recording.ogg</code> "
        "на сервере; скачай её и закинь в Voice Builder.\n"
        "4. Дождись «Voice ready», нажми «<b>Export → JSON</b>».\n"
        "5. Вернись сюда → «📥 Импорт (JSON)» → пришли скачанный файл.\n"
        "6. «💾 Сохранить как активный» — и бот говорит твоим голосом.\n\n"
        "Если бот не догадывается, как читать получившийся JSON — пришли "
        "его как есть, я разберу и подскажу что не так."
    )
    kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="🌐 Открыть Voice Builder",
                    url="https://supertonic.supertone.ai/voice_builder",
                )
            ],
            [InlineKeyboardButton(text="← Назад", callback_data="tts:rec")],
        ]
    )
    if query.message is not None:
        with contextlib.suppress(Exception):
            await query.message.edit_text(
                body, reply_markup=kb, disable_web_page_preview=True
            )
    await query.answer()


# ---- audio / text / json capture handlers --------------------------------


def _is_audioish_message(msg: Message) -> tuple[str, str] | None:
    """Return (file_id, suffix) when the message has a voice / audio / .ogg doc."""
    if msg.voice is not None:
        return msg.voice.file_id, ".ogg"
    if msg.audio is not None:
        mime = (msg.audio.mime_type or "").lower()
        suffix = (
            ".mp3" if "mpeg" in mime else ".m4a" if "mp4" in mime else ".audio"
        )
        return msg.audio.file_id, suffix
    if msg.document is not None:
        mime = (msg.document.mime_type or "").lower()
        if mime.startswith("audio/") or mime == "video/ogg":
            ext = ".ogg"
            name = (msg.document.file_name or "").lower()
            for cand in (".mp3", ".m4a", ".wav", ".ogg"):
                if name.endswith(cand):
                    ext = cand
                    break
            return msg.document.file_id, ext
    return None


@wizard_router.message(
    StateFilter(SetupStates.awaiting_voice_recording),
    F.voice | F.audio | F.document,
)
async def capture_voice_recording(message: Message, state: FSMContext) -> None:
    if not _is_owner(message.from_user.id if message.from_user else 0):
        return
    target = _is_audioish_message(message)
    if target is None:
        await message.answer(
            "Не похоже на аудио. Пришли voice/audio или файл аудио. /cancel — отмена."
        )
        return
    file_id, suffix = target
    out_path = _user_recording_audio_path(message.from_user.id)
    if suffix != ".ogg":
        out_path = out_path.with_suffix(suffix)
    try:
        await message.bot.download(file_id, destination=str(out_path))
    except Exception as exc:  # noqa: BLE001
        await message.answer(
            f"Не получилось сохранить файл: <code>{_html_escape(str(exc))}</code>"
        )
        return
    await state.clear()
    await message.answer(
        "✅ Запись сохранена.\n"
        f"Файл: <code>{out_path}</code>\n\n"
        "Дальше — открой <b>🎶 Voice Builder</b> в меню записи, прогони "
        "файл через сервис и пришли мне получившийся JSON через "
        "<b>📥 Импорт</b>. Или, если JSON уже есть, импортируй его прямо сейчас.",
        reply_markup=_kb_record_menu(),
    )


@wizard_router.message(
    StateFilter(SetupStates.awaiting_voice_recording), _NOT_A_COMMAND
)
async def capture_voice_recording_wrong_type(
    message: Message, state: FSMContext
) -> None:
    """Anything other than audio while waiting for a recording — gentle nudge."""
    if not _is_owner(message.from_user.id if message.from_user else 0):
        return
    await message.answer(
        "Я жду аудио (voice/audio/.ogg/.mp3). /cancel — отмена."
    )


@wizard_router.message(
    StateFilter(SetupStates.awaiting_voice_json_import), F.document
)
async def capture_voice_json(message: Message, state: FSMContext) -> None:
    if not _is_owner(message.from_user.id if message.from_user else 0):
        return
    if message.document is None:
        return
    name = (message.document.file_name or "").lower()
    if not name.endswith(".json"):
        await message.answer("Жду JSON-файл (расширение .json). /cancel — отмена.")
        return
    out_path = _user_voice_json_path(message.from_user.id)
    try:
        await message.bot.download(message.document.file_id, destination=str(out_path))
    except Exception as exc:  # noqa: BLE001
        await message.answer(
            f"Не получилось сохранить JSON: <code>{_html_escape(str(exc))}</code>"
        )
        return
    # Quick sanity check — must be parseable JSON.
    import json as _json

    try:
        _json.loads(out_path.read_text())
    except Exception as exc:  # noqa: BLE001
        out_path.unlink(missing_ok=True)
        await message.answer(
            f"Файл не парсится как JSON: <code>{_html_escape(str(exc))}</code>"
        )
        return
    # Auto-activate the imported clone + auto-enable TTS + flip the
    # TTS provider selector to "clone". Without that last step the
    # bot would still route synthesis through whichever provider was
    # active before (e.g. ElevenLabs) and the freshly-uploaded clone
    # would sit on disk unused — most common "TTS не работает после
    # импорта" report.
    storage.set_tts_custom_voice_path(str(out_path))
    storage.set_tts_enabled(True)
    storage.set_tts_provider("clone")
    await state.clear()
    await message.answer(
        "✅ JSON голоса сохранён и включён как активный.\n"
        "🔊 Голосовой ответчик: <b>ВКЛ</b>, провайдер: <b>🎤 клон</b>.\n"
        "Все следующие текстовые ответы бота продублируются этим голосом.\n\n"
        "Если хочешь сначала послушать без активации — нажми "
        "<b>🔊 Прослушать</b>. Чтобы выключить озвучку — "
        "<b>🔇 Выключить</b> в меню «Голосовой ответчик».",
        reply_markup=_kb_record_menu(),
    )


@wizard_router.message(
    StateFilter(SetupStates.awaiting_voice_json_import), _NOT_A_COMMAND
)
async def capture_voice_json_wrong_type(
    message: Message, state: FSMContext
) -> None:
    if not _is_owner(message.from_user.id if message.from_user else 0):
        return
    await message.answer("Жду JSON-файл (документ .json). /cancel — отмена.")


@wizard_router.message(
    StateFilter(SetupStates.awaiting_voice_preview_text), _NOT_A_COMMAND
)
async def capture_voice_preview_text(
    message: Message, state: FSMContext
) -> None:
    if not _is_owner(message.from_user.id if message.from_user else 0):
        return
    text = (message.text or "").strip()
    if not text:
        await message.answer("Пустой текст. Пришли фразу.")
        return
    data = await state.get_data()
    json_path = str(data.get("preview_voice_json") or "")
    await state.clear()
    await message.bot.send_chat_action(message.chat.id, "record_voice")
    from .tts import synthesize_voice_ogg

    try:
        ogg = await synthesize_voice_ogg(text, custom_path=json_path or None)
    except Exception as exc:  # noqa: BLE001
        await message.answer(
            f"Озвучить не получилось: <code>{_html_escape(str(exc))}</code>"
        )
        return
    if not ogg:
        await message.answer(
            "Озвучить не получилось — Supertonic вернул пусто. Скорее всего "
            "JSON не валидный для Supertonic. Попробуй экспортировать заново "
            "через Voice Builder."
        )
        return
    from aiogram.types import BufferedInputFile

    voice_file = BufferedInputFile(ogg, filename="preview.ogg")
    await message.answer_voice(
        voice_file,
        caption="🔊 Прослушано твоим клонированным голосом.",
    )


# ---- «📚 Узнать о функционале» — LLM tutorial chat ----------------------


def _learn_intro_body() -> str:
    """Welcome message for the «Узнать о функционале» mode."""
    return (
        "<b>📚 Узнать о функционале</b>\n\n"
        "Я знаю всё о себе — спроси что угодно, отвечу через LLM (Мозг 1 "
        "или твой кастомный endpoint из ⚙️ Настройки → 🧠).\n\n"
        "Примеры вопросов:\n"
        "• как клонировать репо с github?\n"
        "• что умеет Helpzavr и как им пользоваться?\n"
        "• как включить голосовой ответчик?\n"
        "• как сменить роль / персону?\n"
        "• что делает «Соображалка»?\n"
        "• какие команды поддерживает Терминал?\n\n"
        "Пиши вопрос текстом — я отвечу. Чтобы выйти из режима — /cancel."
    )


def _learn_system_prompt() -> str:
    """Build a tutorial-mode system prompt with the full capability log."""
    from .capabilities import CAPABILITIES, CAPABILITIES_FILE

    blocks: list[str] = []
    if CAPABILITIES_FILE.exists():
        try:
            blocks.append(CAPABILITIES_FILE.read_text(errors="replace"))
        except Exception:  # noqa: BLE001
            logger.exception("could not read capabilities.md for /learn")
    if not blocks:
        # Fall back to the in-code list so the LLM is never empty-handed
        # (capabilities.md is generated on first boot only).
        for cap in CAPABILITIES:
            chunk = f"### {cap.title}\n{cap.description}"
            if cap.example:
                chunk += f"\n*Пример:* {cap.example}"
            blocks.append(chunk)
    capability_log = "\n\n".join(blocks)
    return (
        "Ты — встроенный помощник внутри Telegram-бота lilush. Твоя "
        "единственная задача — рассказывать пользователю, что умеет "
        "этот бот и как этим пользоваться. Отвечай по-русски, кратко, "
        "по делу, без воды. Если пользователь спрашивает не про бота "
        "(погода, история, философия) — вежливо верни разговор к "
        "функционалу бота.\n\n"
        "Главные возможности бота, на которые опирайся в ответах:\n\n"
        "• Главное меню после /start: 🤖 Helpzavr (скриншот-помощник), "
        "✨ Красивый текст (markdown → TG HTML), 📬 Проверка почты "
        "(IMAP), 🎬 Генерация фото и видео, 🐙 GitHub проекты "
        "(список склонированных репо), 🖥 Терминал (пакетный режим "
        "/work — список команд одним сообщением), и в самом низу "
        "⚙️ Настройки.\n"
        "• ⚙️ Настройки: 🎭 Роли (20 персон, переключатель «требовать "
        "роль»), 🔊 Голосовой ответчик (Supertonic TTS, выбор M1-M5 / "
        "F1-F5 или импорт собственного JSON-голоса через Voice Builder), "
        "🧠 Соображалка (3 стиля статус-индикатора), 💾 Память бота "
        "(размер истории, /reset), 💻 RAM (лимит и поведение при "
        "превышении), 🐙 GitHub проекты, 🖥 Терминал, 📚 Узнать о "
        "функционале (ты сейчас здесь), ⛔ Доступ (private/public/"
        "full_public + список со-владельцев).\n"
        "• Команды: /start, /chat, /setup, /role, /tokens, /reset, "
        "/work (пакетный терминал), /cancel (выйти из любой FSM-моды), "
        "/exec, /clone <url>, /git, /project, /cd, /github, /helpzavr, "
        "/pretty, /mailbox, /media, /settings.\n"
        "• Мозги: Мозг 1 — основной чат-LLM (OpenRouter / свой OpenAI-"
        "совместимый endpoint типа Groq, Together, vLLM, LM Studio). "
        "Мозг 2 — резерв для голосовых/фото перехватов (Groq Whisper / "
        "Vision). Auto-failover между слотами.\n"
        "• Внешние API: Apify, Firecrawl, Tavily, Brave Search, Exa, "
        "GitHub PAT — конфигурятся через /setup → 🛠 Внешние API.\n"
        "• GitHub: можешь кинуть ссылку в чат — бот сам склонирует "
        "репо (если включено авто-клонирование) и сможет читать "
        "файлы / запускать команды внутри.\n"
        "• Голосовой ответчик: чтобы заработал — выбрать голос (или "
        "импортнуть JSON) и нажать «🔊 Авто включить». На сервере "
        "нужны ffmpeg и пакет supertonic — оба ставятся "
        "автоматически при деплое (Dockerfile + requirements.txt).\n\n"
        "Если есть подробный лог возможностей — используй его:\n\n"
        f"{capability_log}"
    )


async def _answer_learn_question(message: Message, question: str) -> None:
    """Run a one-shot LLM call with the tutorial framing and reply."""
    from .agent import NoApiKeyError, oneshot_summary

    await message.bot.send_chat_action(message.chat.id, "typing")
    system = _learn_system_prompt()
    prompt = (
        f"{system}\n\n"
        f"Вопрос пользователя: {question}\n\n"
        "Ответь кратко и по делу, на русском."
    )
    try:
        answer = await oneshot_summary(prompt, purpose="learn")
    except NoApiKeyError as exc:
        await message.answer(
            "Не могу спросить LLM — Мозг 1 ещё не настроен.\n"
            f"<i>{_html_escape(str(exc))}</i>\n\n"
            "Открой ⚙️ Настройки → 🧠 Перенастроить мозг и заведи "
            "OpenRouter ключ (или свой OpenAI-совместимый endpoint)."
        )
        return
    except Exception as exc:  # noqa: BLE001
        logger.exception("learn mode failed")
        await message.answer(
            f"Не получилось спросить LLM: <code>{_html_escape(str(exc))}</code>"
        )
        return
    if not answer:
        await message.answer(
            "LLM ничего не ответил. Попробуй переформулировать вопрос."
        )
        return
    await message.answer(answer)


@wizard_router.message(
    StateFilter(SetupStates.awaiting_learn_question), _NOT_A_COMMAND
)
async def capture_learn_question(message: Message, state: FSMContext) -> None:
    """Route the user's question to the LLM tutorial helper.

    Stays in :class:`SetupStates.awaiting_learn_question` after each
    answer so the user can keep asking follow-up questions. They
    leave the mode with /cancel (handled by the global ``cmd_cancel``).
    """
    if not can_use(message.from_user.id if message.from_user else None):
        return
    text = (message.text or "").strip()
    if not text:
        await message.answer("Пустой вопрос. Напиши, что хочешь узнать.")
        return
    await _answer_learn_question(message, text)


async def _current_brain_slot(state: FSMContext) -> str:
    """Read the slot the user is currently configuring from FSM data.

    The slot is stashed by `cb_brain_slot` when the user enters a Brain
    1 / Brain 2 screen. Defaults to ``"1"`` when missing (e.g. user came
    through the legacy `/setup` → «Другое» path).
    """
    data = await state.get_data()
    raw = str(data.get("brain_cfg_slot") or "1")
    return raw if raw in ("1", "2") else "1"


@wizard_router.callback_query(F.data.startswith("brain_slot:"))
async def cb_brain_slot(query: CallbackQuery, state: FSMContext) -> None:
    """Open the per-slot config screen for Brain 1 or Brain 2.

    Sets ``brain_cfg_slot`` in FSM data so subsequent cfg:* callbacks
    and capture handlers know which slot to write into.
    """
    if not _is_owner(query.from_user.id):
        await query.answer("Только владелец.", show_alert=True)
        return
    slot = (query.data or "").split(":", 1)[1]
    if slot not in ("1", "2"):
        await query.answer("Неизвестный слот.", show_alert=True)
        return
    await state.update_data(brain_cfg_slot=slot)
    if slot == "1":
        storage.set_brain("auto")
        if storage.get_provider() not in ("openrouter", "custom"):
            storage.set_provider("openrouter")
    provider, api_key, base_url, model = _slot_config(slot)
    if slot == "1":
        role_line = "Назначение: основной мозг для чата.\n\n"
        cta = "Нажми «✅ Готово» когда закончишь."
    else:
        role_line = (
            "Назначение: голос/фото перехваты через Groq (Whisper / Vision). "
            "Чат всегда обслуживается Мозгом 1.\n\n"
        )
        cta = "Нажми «✅ Готово» когда закончишь — слот сохранится, активным мозгом чата остаётся Мозг 1."
    body = (
        f"<b>🧠 Мозг {slot}</b>\n\n"
        f"{role_line}"
        f"Провайдер: <code>{_html_escape(provider)}</code>\n"
        f"URL: <code>{_html_escape(base_url) if base_url else '(по умолчанию)'}</code>\n"
        f"Модель: <code>{_html_escape(model) if model else '(не задана)'}</code>\n"
        f"API ключ: {'✅ задан' if api_key else '⚠️ пусто'}\n\n"
        f"{cta}"
    )
    if query.message is not None:
        await query.message.edit_text(body, reply_markup=_kb_brain_slot_cfg(slot))
    await query.answer()


@wizard_router.callback_query(F.data == "auto_brain:back")
async def cb_auto_brain_back(query: CallbackQuery, state: FSMContext) -> None:
    """«← Вернуться» from the Мозг 1 / Мозг 2 submenu → brain picker."""
    if not _is_owner(query.from_user.id):
        await query.answer("Только владелец.", show_alert=True)
        return
    await state.clear()
    if query.message is not None:
        with contextlib.suppress(Exception):
            await query.message.edit_text("Выбери мозг:", reply_markup=_kb_brain())
    await query.answer()


@wizard_router.callback_query(F.data == "cfg:back")
async def cb_cfg_back(query: CallbackQuery, state: FSMContext) -> None:
    """Back arrow inside the per-slot / legacy config screen.

    Routes back to the Мозг 1 / Мозг 2 submenu if the user came in
    through a slot button (slot stashed in FSM data), otherwise back
    to the brain picker so the legacy «Другое» flow keeps working.
    """
    if not _is_owner(query.from_user.id):
        await query.answer("Только владелец.", show_alert=True)
        return
    data = await state.get_data()
    came_via_slot = bool(data.get("brain_cfg_slot"))
    await state.clear()
    if query.message is not None:
        if came_via_slot:
            await query.message.edit_text(
                "<b>🧠 Авто мозг</b>\n\nВыбери слот:",
                reply_markup=_kb_auto_brain(),
            )
        else:
            await query.message.edit_text(
                "Выбери мозг:",
                reply_markup=_kb_brain(),
            )
    await query.answer()


@wizard_router.callback_query(F.data == "cfg:done")
async def cb_cfg_done(query: CallbackQuery, state: FSMContext) -> None:
    if not _is_owner(query.from_user.id):
        await query.answer("Только владелец.", show_alert=True)
        return
    slot = await _current_brain_slot(state)
    await state.clear()
    summary = _config_summary()
    # Slot 1 — chat brain. Slot 2 is reserved for voice/photo via the
    # Groq override and is NEVER promoted to chat duty here. See
    # ``bot/agent.py::_slot_chain`` for the routing rule.
    if slot == "2":
        tail = (
            "Мозг 2 используется только для голосовых/фото перехватов "
            "(Groq Whisper / Vision). Чат по-прежнему идёт через Мозг 1."
        )
    else:
        tail = (
            "Теперь любое сообщение без <code>/</code> уйдёт в LLM. "
            "/help — все команды."
        )
    if query.message is not None:
        await query.message.edit_text(
            f"<b>Готово.</b>\n\n{summary}\n\n{tail}"
        )
    await query.answer()


@wizard_router.callback_query(F.data == "cfg:api")
async def cb_cfg_api(query: CallbackQuery, state: FSMContext) -> None:
    if not _is_owner(query.from_user.id):
        await query.answer("Только владелец.", show_alert=True)
        return
    slot = await _current_brain_slot(state)
    await state.update_data(brain_cfg_slot=slot)  # preserve across set_state
    await state.set_state(SetupStates.awaiting_api_key)
    if query.message is not None:
        provider, _, base_url, _ = _slot_config(slot)
        if provider == "openrouter":
            hint = "Получить: <a href='https://openrouter.ai/keys'>openrouter.ai/keys</a>"
        elif "groq.com" in (base_url or "").lower():
            hint = "Получить: <a href='https://console.groq.com/keys'>console.groq.com/keys</a>"
        else:
            hint = "Это твой ключ от того endpoint-а, который ты указал в URL"
        await query.message.answer(
            f"Пришли API-ключ для <b>Мозга {slot}</b> <b>одним сообщением</b>. "
            "Я удалю его из чата как только сохраню.\n\n" + hint
        )
    await query.answer()


# Quick-pick model presets for Brain 2 (Groq voice / vision slot). Tap a
# button = save the model directly; «✍️ Свой вариант» falls back to the
# original text-input flow for anything not on the list.
_BRAIN2_MODEL_PRESETS: dict[str, str] = {
    "whisper": "whisper-large-v3",
    "whisperturbo": "whisper-large-v3-turbo",
    "scout": "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama70": "llama-3.3-70b-versatile",
}


def _kb_brain2_model_picker() -> InlineKeyboardMarkup:
    """Quick-pick keyboard shown when the user opens «🤖 Модель» for
    Brain 2 — the slot for Groq STT (whisper-*) and vision/chat models.
    The current default (whisper-large-v3) is marked with ⭐.
    """
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="🎙 whisper-large-v3 ⭐ дефолт",
                    callback_data="cfg:m2p:whisper",
                )
            ],
            [
                InlineKeyboardButton(
                    text="🎙 whisper-large-v3-turbo",
                    callback_data="cfg:m2p:whisperturbo",
                )
            ],
            [
                InlineKeyboardButton(
                    text="🦙 llama-4-scout-17b-16e-instruct",
                    callback_data="cfg:m2p:scout",
                )
            ],
            [
                InlineKeyboardButton(
                    text="🦙 llama-3.3-70b-versatile",
                    callback_data="cfg:m2p:llama70",
                )
            ],
            [
                InlineKeyboardButton(
                    text="✍️ Свой вариант",
                    callback_data="cfg:model_freetext",
                )
            ],
            [
                InlineKeyboardButton(
                    text="← Перенастроить мозг",
                    callback_data="cfg:back",
                )
            ],
        ]
    )


@wizard_router.callback_query(F.data == "cfg:model")
async def cb_cfg_model(query: CallbackQuery, state: FSMContext) -> None:
    if not _is_owner(query.from_user.id):
        await query.answer("Только владелец.", show_alert=True)
        return
    slot = await _current_brain_slot(state)
    await state.update_data(brain_cfg_slot=slot)
    # Brain 2: show a quick-pick keyboard (Groq voice / vision presets)
    # before falling back to free-text input. Brain 1: open-ended text
    # since the OpenRouter / custom space is too large to enumerate.
    if slot == "2":
        if query.message is not None:
            await query.message.answer(
                "Выбери модель для <b>Мозга 2</b>. Кликни одну из "
                "популярных (сохранится сразу) или «✍️ Свой вариант», "
                "чтобы ввести вручную.",
                reply_markup=_kb_brain2_model_picker(),
            )
        await query.answer()
        return
    await state.set_state(SetupStates.awaiting_model)
    if query.message is not None:
        provider, _, base_url, _ = _slot_config(slot)
        if provider == "openrouter":
            hint = (
                "Примеры:\n"
                "• <code>nvidia/nemotron-3-super-120b-a12b:free</code> (бесплатно)\n"
                "• <code>anthropic/claude-opus-4.7</code> (платно, лучшее качество)\n"
                "• <code>openai/gpt-5</code>\n"
                "Полный список: <a href='https://openrouter.ai/models'>openrouter.ai/models</a>"
            )
        elif "groq.com" in (base_url or "").lower():
            hint = (
                "Примеры моделей Groq:\n"
                "• <code>meta-llama/llama-4-scout-17b-16e-instruct</code>\n"
                "• <code>llama-3.3-70b-versatile</code>\n"
                "Список: <a href='https://console.groq.com/docs/models'>console.groq.com/docs/models</a>"
            )
        else:
            hint = (
                "Имя модели для твоего endpoint-а. Например <code>llama3.1:70b</code>, "
                "<code>mistral-large</code>, <code>gpt-4o</code>."
            )
        await query.message.answer(
            f"Пришли имя модели для <b>Мозга {slot}</b> одним сообщением.\n\n" + hint
        )
    await query.answer()


@wizard_router.callback_query(F.data.startswith("cfg:m2p:"))
async def cb_brain2_model_preset(query: CallbackQuery, state: FSMContext) -> None:
    """Save the picked preset model for Brain 2 in one click.

    Mirrors the post-save banner produced by the text-input flow so the
    feedback feels identical regardless of which path the user took.
    """
    if not _is_owner(query.from_user.id):
        await query.answer("Только владелец.", show_alert=True)
        return
    key = (query.data or "").split(":", 2)[-1]
    model = _BRAIN2_MODEL_PRESETS.get(key)
    if not model:
        await query.answer("Неизвестная модель.", show_alert=True)
        return
    storage.set_brain_slot2_field("model", model)
    await state.update_data(brain_cfg_slot="2")
    summary = _config_summary()
    if query.message is not None:
        await query.message.answer(
            f"Модель для Мозга 2 сохранена: <code>{_html_escape(model)}</code>\n\n{summary}",
            reply_markup=_kb_brain_slot_cfg("2"),
        )
    await query.answer("Принято")


@wizard_router.callback_query(F.data == "cfg:model_freetext")
async def cb_brain2_model_freetext(query: CallbackQuery, state: FSMContext) -> None:
    """Brain 2 user clicked «✍️ Свой вариант» — fall through to the
    original free-text capture so they can paste any Groq model name.
    """
    if not _is_owner(query.from_user.id):
        await query.answer("Только владелец.", show_alert=True)
        return
    await state.update_data(brain_cfg_slot="2")
    await state.set_state(SetupStates.awaiting_model)
    if query.message is not None:
        await query.message.answer(
            "Пришли имя модели для <b>Мозга 2</b> одним сообщением.\n\n"
            "Примеры моделей Groq:\n"
            "• <code>meta-llama/llama-4-scout-17b-16e-instruct</code>\n"
            "• <code>llama-3.3-70b-versatile</code>\n"
            "• <code>whisper-large-v3</code>\n"
            "Список: <a href='https://console.groq.com/docs/models'>console.groq.com/docs/models</a>"
        )
    await query.answer()


@wizard_router.callback_query(F.data == "cfg:url")
async def cb_cfg_url(query: CallbackQuery, state: FSMContext) -> None:
    if not _is_owner(query.from_user.id):
        await query.answer("Только владелец.", show_alert=True)
        return
    slot = await _current_brain_slot(state)
    provider, _, _, _ = _slot_config(slot)
    if provider == "openrouter":
        # URL для OpenRouter не настраивается, чтобы юзер случайно не сломал коннект.
        await query.answer(
            "OpenRouter не требует URL. Чтобы задать свой endpoint — сначала "
            "переключи провайдер на custom через «Другое» или вставь Groq-URL.",
            show_alert=True,
        )
        return
    await state.update_data(brain_cfg_slot=slot)
    await state.set_state(SetupStates.awaiting_url)
    if query.message is not None:
        await query.message.answer(
            f"Пришли URL endpoint-а для <b>Мозга {slot}</b> одним сообщением. "
            "Должен быть OpenAI-совместимым (заканчивается на <code>/v1</code>).\n\n"
            "Примеры:\n"
            "• <code>https://api.together.xyz/v1</code>\n"
            "• <code>https://api.groq.com/openai/v1</code>\n"
            "• <code>http://localhost:11434/v1</code> (Ollama)\n"
            "• <code>http://10.0.0.5:8000/v1</code> (свой vLLM)"
        )
    await query.answer()


# --- text capture for FSM --------------------------------------------------
# Filter out commands so users can /cancel, /setup, /help mid-wizard without
# their command being treated as the requested input. (`_NOT_A_COMMAND` is
# also referenced by the voice-recording / JSON-import handlers above, so
# the definition must precede them — see the early definition near the
# top of this module.)


@wizard_router.message(StateFilter(SetupStates.awaiting_api_key), _NOT_A_COMMAND)
async def capture_api_key(message: Message, state: FSMContext) -> None:
    if not _is_owner(message.from_user.id if message.from_user else 0):
        return
    key = (message.text or "").strip()
    if not key:
        await message.answer("Пустое сообщение, попробуй ещё раз или нажми «← Назад».")
        return
    data = await state.get_data()
    slot = str(data.get("brain_cfg_slot") or "1")
    if slot not in ("1", "2"):
        slot = "1"
    # Save FIRST, then delete the message with the secret. If save fails we
    # want the user to be able to retry without re-typing the key blind.
    try:
        if slot == "2":
            storage.set_brain_slot2_field("api_key", key)
        else:
            # Brain 1 — historical top-level slot. The agent always reads the
            # API key from the "openrouter" slot regardless of provider
            # (custom endpoints reuse this slot — only base_url differs).
            storage.set_provider_key("openrouter", key)
    except Exception as exc:  # noqa: BLE001 — surface to the user, don't crash
        await message.answer(
            f"Не удалось сохранить ключ: <code>{_html_escape(str(exc))}</code>\n"
            "Попробуй ещё раз или напиши /cancel."
        )
        return
    with contextlib.suppress(Exception):
        await message.delete()
    came_via_slot = bool(data.get("brain_cfg_slot"))
    await state.clear()
    # Preserve slot context so subsequent cfg:* clicks stay on the same slot.
    if came_via_slot:
        await state.update_data(brain_cfg_slot=slot)
    summary = _config_summary()
    kb = _kb_brain_slot_cfg(slot) if came_via_slot else _kb_llm_config(_provider_label_from_storage())
    await message.answer(
        f"Ключ для Мозга {slot} сохранён, твоё сообщение удалено.\n\n"
        f"{summary}",
        reply_markup=kb,
    )


@wizard_router.message(StateFilter(SetupStates.awaiting_model), _NOT_A_COMMAND)
async def capture_model(message: Message, state: FSMContext) -> None:
    if not _is_owner(message.from_user.id if message.from_user else 0):
        return
    model = (message.text or "").strip()
    if not model:
        await message.answer("Пустое сообщение, попробуй ещё раз.")
        return
    data = await state.get_data()
    slot = str(data.get("brain_cfg_slot") or "1")
    if slot not in ("1", "2"):
        slot = "1"
    if slot == "2":
        storage.set_brain_slot2_field("model", model)
    else:
        storage.set_model(model)
    came_via_slot = bool(data.get("brain_cfg_slot"))
    await state.clear()
    if came_via_slot:
        await state.update_data(brain_cfg_slot=slot)
    summary = _config_summary()
    kb = _kb_brain_slot_cfg(slot) if came_via_slot else _kb_llm_config(_provider_label_from_storage())
    await message.answer(
        f"Модель для Мозга {slot} сохранена: <code>{_html_escape(model)}</code>\n\n{summary}",
        reply_markup=kb,
    )


@wizard_router.message(
    StateFilter(SetupStates.awaiting_external_tool_key), _NOT_A_COMMAND
)
async def capture_external_tool_key(message: Message, state: FSMContext) -> None:
    """Save an external-tool API key after the user has clicked one in the menu.

    The chosen tool name was stashed in FSM ``data["external_tool"]`` by
    ``cb_ext_set``; we read it here, persist the key, delete the user's
    message (to keep the secret out of TG history), and re-show the menu
    with the updated ✅ marker.
    """
    if not _is_owner(message.from_user.id if message.from_user else 0):
        return
    data = await state.get_data()
    tool = str(data.get("external_tool") or "").strip().lower()
    if not tool:
        await message.answer(
            "Не помню для какого инструмента ты задаёшь ключ — открой меню заново через /setup."
        )
        await state.clear()
        return
    key = (message.text or "").strip()
    if not key:
        await message.answer("Пустое сообщение, попробуй ещё раз.")
        return
    with contextlib.suppress(Exception):
        await message.delete()
    try:
        storage.set_external_tool_key(tool, key)
    except ValueError as exc:
        await message.answer(f"Не получилось сохранить ключ: {exc}")
        await state.clear()
        return
    await state.clear()
    await message.answer(
        f"Ключ для <b>{tool}</b> сохранён, твоё сообщение удалено.\n\n"
        + _external_tools_summary(),
        reply_markup=_kb_external_tools(),
        disable_web_page_preview=True,
    )


@wizard_router.message(
    StateFilter(SetupStates.awaiting_elevenlabs_api_key), _NOT_A_COMMAND
)
async def capture_elevenlabs_api_key(message: Message, state: FSMContext) -> None:
    """Save ElevenLabs API key from a paste, delete the user's message
    (so the secret never sits in chat history) and re-render the menu.
    """
    if not _is_owner(message.from_user.id if message.from_user else 0):
        return
    key = (message.text or "").strip()
    if not key:
        await message.answer("Пустое сообщение, попробуй ещё раз.")
        return
    with contextlib.suppress(Exception):
        await message.delete()
    storage.set_elevenlabs_api_key(key)
    await state.clear()
    await message.answer(
        "🔑 ElevenLabs API ключ сохранён, твоё сообщение удалено.\n\n"
        + _elevenlabs_screen_body(),
        reply_markup=_kb_elevenlabs_menu(),
        disable_web_page_preview=True,
    )


@wizard_router.message(
    StateFilter(SetupStates.awaiting_elevenlabs_voice_id), _NOT_A_COMMAND
)
async def capture_elevenlabs_voice_id(message: Message, state: FSMContext) -> None:
    """Save ElevenLabs Voice ID. Voice IDs are not secrets, but we
    still delete the user's message to keep the chat clean.
    """
    if not _is_owner(message.from_user.id if message.from_user else 0):
        return
    voice_id = (message.text or "").strip()
    if not voice_id:
        await message.answer("Пустое сообщение, попробуй ещё раз.")
        return
    with contextlib.suppress(Exception):
        await message.delete()
    storage.set_elevenlabs_voice_id(voice_id)
    await state.clear()
    await message.answer(
        f"🎤 Voice ID сохранён: <code>{_html_escape(voice_id)}</code>\n\n"
        + _elevenlabs_screen_body(),
        reply_markup=_kb_elevenlabs_menu(),
        disable_web_page_preview=True,
    )


@wizard_router.message(StateFilter(SetupStates.awaiting_url), _NOT_A_COMMAND)
async def capture_url(message: Message, state: FSMContext) -> None:
    if not _is_owner(message.from_user.id if message.from_user else 0):
        return
    url = (message.text or "").strip()
    if not url:
        await message.answer("Пустое сообщение, попробуй ещё раз.")
        return
    if not url.startswith(("http://", "https://")):
        await message.answer(
            "URL должен начинаться с <code>http://</code> или <code>https://</code>. Попробуй ещё раз."
        )
        return
    data = await state.get_data()
    slot = str(data.get("brain_cfg_slot") or "1")
    if slot not in ("1", "2"):
        slot = "1"
    if slot == "2":
        storage.set_brain_slot2_field("base_url", url)
        # If user sets a URL on slot 2 we assume it's a custom endpoint;
        # the provider field stays "custom" by default.
        storage.set_brain_slot2_field("provider", "custom")
    else:
        storage.set_base_url(url)
    came_via_slot = bool(data.get("brain_cfg_slot"))
    await state.clear()
    if came_via_slot:
        await state.update_data(brain_cfg_slot=slot)
    summary = _config_summary()
    kb = _kb_brain_slot_cfg(slot) if came_via_slot else _kb_llm_config(_provider_label_from_storage())
    await message.answer(
        f"URL для Мозга {slot} сохранён: <code>{_html_escape(url)}</code>\n\n{summary}",
        reply_markup=kb,
    )


# ---- /main:* — top-level menu callbacks shortcuts -----------------------


@wizard_router.callback_query(F.data.startswith("main:"))
async def cb_main_menu(query: CallbackQuery, state: FSMContext) -> None:
    """Handle clicks on the top-level main menu shown after /start.

    Auth model:
    * The action ``main:settings`` plus the four feature shortcuts
      (helpzavr / pretty / mailbox / media_toggle) are open to anyone
      who passes :func:`can_use` (so guests in public mode can use the
      bot's features).
    * The settings-y actions (download / role / brain / tokens) are
      admin-only (:func:`can_admin`) — guests don't see those buttons
      in :func:`_kb_main_after_claim` but we still gate the callback
      defensively in case the menu was rendered before access mode was
      flipped back to private.
    """
    user = query.from_user
    uid = user.id if user else None
    if not can_use(uid):
        await query.answer("Нет доступа.", show_alert=True)
        return
    action = (query.data or "").split(":", 1)[1]
    if action in {"download", "role", "brain", "tokens", "terminal", "learn", "editor"} and not can_admin(uid):
        await query.answer("Только владелец.", show_alert=True)
        return
    if action.startswith("install:") and not can_admin(uid):
        await query.answer("Только владелец.", show_alert=True)
        return
    await state.clear()

    if action.startswith("install:"):
        slug = action.split(":", 1)[1]
        batch = INSTALL_BATCHES.get(slug)
        if batch is None:
            await query.answer("Неизвестная установка", show_alert=True)
            return
        title, lines = batch
        await query.answer()
        if query.message is not None:
            from .config import DATA_DIR
            from .handlers import run_work_batch

            # bash2mp4 in particular: if the user already ran the install,
            # don't re-clone — show the settings screen with the
            # «использовать / удалить» toggle. The user explicitly asked
            # for a single «Использовать» button on each downloader
            # screen so that picking one disables the other.
            if slug == "downloader":
                project_dir = DATA_DIR / "projects" / "bash2mp4"
                if project_dir.is_dir():
                    await _show_bash2mp4_screen(query)
                    return

            # Pass the real clicker uid: ``query.message.from_user`` is the
            # bot itself for inline-keyboard messages, so the authorisation
            # check inside ``run_work_batch`` needs the user explicitly.
            await run_work_batch(query.message, lines, title=title, user_id=uid)

            # After bash2mp4 successfully clones, show the full settings
            # screen (status + «Использовать» button) instead of just
            # printing static help text.
            if slug == "downloader":
                await _show_bash2mp4_screen(query)
        return

    if action == "helpzavr":
        await query.answer()
        try:
            from .addons.helpzavr.handlers import show_screen as _hz_show
            await _hz_show(query)
        except Exception:  # noqa: BLE001
            logger.exception("helpzavr screen failed")
            if query.message is not None:
                await query.message.answer(
                    "Аддон Helpzavr не загружен. Проверь логи."
                )
        return

    if action == "markitdown":
        # Settings screen for the Markitdown auxiliary tool.
        # Five buttons: auto-on, auto-off, install (clone), uninstall, back.
        # "Auto" mode means when the user pastes a YouTube/Vimeo URL with
        # «дай данные <URL>» / «посмотри и расскажи о чём там <URL>», the
        # bot runs markitdown on it automatically instead of falling
        # through to the LLM agent.
        await query.answer()
        await _show_markitdown_screen(query)
        return

    if action == "pretty":
        await query.answer()
        try:
            from .addons.pretty_text.handlers import show_screen as _pt_show
            await _pt_show(query)
        except Exception:  # noqa: BLE001
            logger.exception("pretty_text screen failed")
            if query.message is not None:
                await query.message.answer(
                    "Аддон Красивый текст не загружен. Проверь логи."
                )
        return

    if action == "mailbox":
        await query.answer()
        try:
            from .addons.mailbox.handlers import show_main as _mb_show
            await _mb_show(query)
        except Exception:  # noqa: BLE001
            logger.exception("mailbox screen failed")
            if query.message is not None:
                await query.message.answer(
                    "Аддон Проверка почты не загружен. Проверь логи."
                )
        return

    if action == "media_toggle":
        await query.answer()
        try:
            from .addons.media_toggle.handlers import show_screen as _mt_show
            await _mt_show(query)
        except Exception:  # noqa: BLE001
            logger.exception("media_toggle screen failed")
            if query.message is not None:
                await query.message.answer(
                    "Аддон Генерация фото и видео не загружен. Проверь логи."
                )
        return

    if action == "github":
        # Shortcut from the main menu's «🐙 GitHub проекты» row —
        # opens the same screen as /github or Settings → 🐙 GitHub
        # проекты. Re-uses ``cb_gh_menu`` so the body / keyboard stay
        # in sync with the canonical screen.
        await cb_gh_menu(query, state)
        return

    if action == "terminal":
        # Shortcut from the main menu's «🖥 Терминал» row — enters the
        # batch-terminal FSM exactly like ``/work`` does. We need the
        # admin check because /work modifies the project workspace.
        if not can_admin(uid):
            await query.answer("Только владелец.", show_alert=True)
            return
        await query.answer()
        from .handlers import WorkStates

        await state.set_state(WorkStates.awaiting_commands)
        if query.message is not None:
            await query.message.answer(
                "<b>🖥 Терминал</b>\n\n"
                "Жду команды списком (по одной на строку). Пришли одним "
                "сообщением — выполню по порядку, ждя завершения каждой. "
                "Таймаут на команду снят (до 1 часа).\n\n"
                "Поддерживаю: <code>/exec</code>, <code>/clone</code>, "
                "<code>/project</code>, <code>/cd</code>, <code>/git</code>. "
                "Строка без <code>/</code> — bash в текущем проекте. Пустые "
                "строки и <code>#</code>-комментарии пропускаю.\n\n"
                "Выйти из режима: /cancel"
            )
        return

    if action == "learn":
        # Shortcut from Settings → «📚 Узнать о функционале» — opens a
        # tutorial chat with the main LLM. Subsequent text messages
        # are routed through ``cb_learn_question`` until the user
        # types /cancel.
        await query.answer()
        await state.set_state(SetupStates.awaiting_learn_question)
        if query.message is not None:
            await query.message.answer(_learn_intro_body())
        return

    if action == "settings":
        await query.answer()
        body_extra = ""
        if can_admin(uid):
            body_extra = (
                "\n\n<b>⛔ Доступ</b> — публичный / приватный режим бота, "
                "кому ты дал доступ."
            )
        kb = _kb_settings_menu(uid)
        gate_hint = (
            "ВКЛ — без /role проектные команды молчат"
            if storage.get_role_gate_enabled()
            else "ВЫКЛ — команды работают без выбора роли"
        )
        tts_hint = (
            "ВКЛ — каждый текстовый ответ дублируется голосом"
            if storage.get_tts_enabled()
            else "ВЫКЛ — бот молчит, отвечает только текстом"
        )
        if query.message is not None:
            try:
                await query.message.edit_text(
                    "<b>⚙️ Настройки</b>\n\n"
                    f"<b>🎭 Роли</b> — сменить персону и вкл/выкл требование "
                    f"роли для /work, /exec, /clone (сейчас: {gate_hint}).\n\n"
                    f"<b>🔊 Голосовой ответчик</b> — озвучка ответов бота "
                    f"через Supertonic (сейчас: {tts_hint}).\n\n"
                    "<b>🧠 Соображалка</b> — как бот показывает что он "
                    "думает (белый бабл / белый + модель / тёмная плитка "
                    "по центру).\n\n"
                    "<b>💾 Память бота</b> — сколько последних сообщений "
                    "помнит и кнопка очистить.\n\n"
                    "<b>🧩 Алгоритм</b> — 10 слотов с пошаговыми планами. "
                    "Бот сам составит план по новости/описанию, может "
                    "запускать его периодически (мин/час/день).\n\n"
                    "<b>🔌 API (массовый импорт)</b> — одной строкой "
                    "загрузи все ключи (Brain 1/2, ElevenLabs, Tavily, "
                    "Brave, Exa, Firecrawl, Apify, GitHub PAT). "
                    "Разделители: <code>$$$</code> между блоками, "
                    "<code>;</code> между полями.\n\n"
                    "<b>🐙 GitHub проекты</b> — склонированные репозитории "
                    "(список, описание, удаление) и переключатель "
                    "авто-клонирования по ссылке.\n\n"
                    "<b>🖥 Терминал</b> — пакетный режим /work: пришлёшь "
                    "список команд одним сообщением, я выполню по очереди.\n\n"
                    "<b>📚 Узнать о функционале</b> — спроси LLM-помощника "
                    "что умеет бот и как этим пользоваться." + body_extra,
                    reply_markup=kb,
                )
            except Exception:  # noqa: BLE001
                await query.message.answer(
                    "Настройки", reply_markup=kb
                )
        return

    if action == "thinking":
        await query.answer()
        try:
            from .addons.thinking_style import show_screen as _ts_show
            await _ts_show(query)
        except Exception:  # noqa: BLE001
            logger.exception("thinking_style screen failed")
            if query.message is not None:
                await query.message.answer(
                    "Аддон Соображалка не загружен. Проверь логи."
                )
        return

    if action == "memory":
        await query.answer()
        try:
            from .addons.memory.handlers import show_screen as _mem_show
            await _mem_show(query)
        except Exception:  # noqa: BLE001
            logger.exception("memory screen failed")
            if query.message is not None:
                await query.message.answer(
                    "Аддон Память бота не загружен. Проверь логи."
                )
        return

    if action == "ram":
        await query.answer()
        try:
            from .addons.ram_guard.handlers import show_screen as _ram_show
            await _ram_show(query)
        except Exception:  # noqa: BLE001
            logger.exception("ram screen failed")
            if query.message is not None:
                await query.message.answer(
                    "Аддон RAM не загружен. Проверь логи."
                )
        return

    if action == "download":
        await query.answer()
        await _show_download_screen(query)
        return

    if action == "role":
        await query.answer()
        await _show_role_picker(query, state)
        return

    if action == "brain":
        await query.answer()
        if query.message is not None:
            await query.message.edit_text(
                "Перенастройка. Выбери мозг:",
                reply_markup=_kb_brain(),
            )
        return

    if action == "editor":
        await query.answer()
        try:
            from .addons.editor_agent.handlers import show_screen as _ed_show
            await _ed_show(query)
        except Exception:  # noqa: BLE001
            logger.exception("editor_agent screen failed")
            if query.message is not None:
                await query.message.answer(
                    "Аддон Монтажёр не загружен. Проверь логи."
                )
        return

    if action == "tokens":
        await query.answer()
        if query.message is not None:
            from .token_tracker import format_token_stats

            await query.message.answer(format_token_stats())
        return

    await query.answer("Неизвестный пункт меню", show_alert=True)


# ---- /access:* — public/private mode + co-owner list --------------------


_ACCESS_LABELS = {
    "private": "🔒 Приватная",
    "public": "🌐 Публичная",
    "full_public": "🔓 Публичная полная",
}


def _kb_access_menu() -> InlineKeyboardMarkup:
    mode = storage.get_access_mode()
    rows: list[list[InlineKeyboardButton]] = []
    for key in ("public", "full_public", "private"):
        marker = "▶ " if mode == key else ""
        rows.append(
            [
                InlineKeyboardButton(
                    text=f"{marker}{_ACCESS_LABELS[key]}",
                    callback_data=f"access:set:{key}",
                )
            ]
        )
    rows.append(
        [InlineKeyboardButton(text="➕ Дать доступ", callback_data="access:add")]
    )
    if storage.get_co_owners():
        rows.append(
            [InlineKeyboardButton(text="👥 Список доступов", callback_data="access:list")]
        )
    rows.append(
        [InlineKeyboardButton(text="↩️ Вернуться", callback_data="access:back")]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _access_screen_body() -> str:
    mode = storage.get_access_mode()
    co_owners = storage.get_co_owners()
    co_owner_str = (
        ", ".join(f"<code>{cid}</code>" for cid in co_owners)
        if co_owners
        else "<i>пусто</i>"
    )
    return (
        "<b>⛔ Доступ</b>\n\n"
        f"Сейчас: <b>{_ACCESS_LABELS[mode]}</b>\n\n"
        "<b>🌐 Публичная</b> — пользоваться ботом могут все, "
        "но менять настройки (API-ключи, модели, статистика, импорт API, "
        "доступ) можешь только ты + те, кому ты дал доступ.\n\n"
        "<b>🔓 Публичная полная</b> — у всех полный доступ как у "
        "владельца, включая настройки и API-ключи.\n\n"
        "<b>🔒 Приватная</b> — только ты + те, кому ты дал доступ. "
        "Остальные получают вежливый отказ.\n\n"
        f"Соучастники (полный доступ как у владельца): {co_owner_str}"
    )


@wizard_router.callback_query(F.data == "access:menu")
async def cb_access_menu(query: CallbackQuery, state: FSMContext) -> None:
    user = query.from_user
    if not can_admin(user.id if user else None):
        await query.answer("Только владелец.", show_alert=True)
        return
    await state.clear()
    if query.message is not None:
        try:
            await query.message.edit_text(
                _access_screen_body(), reply_markup=_kb_access_menu()
            )
        except Exception:  # noqa: BLE001
            await query.message.answer(
                _access_screen_body(), reply_markup=_kb_access_menu()
            )
    await query.answer()


@wizard_router.callback_query(F.data.startswith("access:set:"))
async def cb_access_set(query: CallbackQuery, state: FSMContext) -> None:
    user = query.from_user
    if not can_admin(user.id if user else None):
        await query.answer("Только владелец.", show_alert=True)
        return
    # Only the real owner (not co-owners) may flip the access mode
    # so a co-owner can't accidentally lock the owner out by making
    # everything full-public or by demoting back to private.
    if not is_owner(user.id if user else None):
        await query.answer(
            "Менять режим доступа может только владелец.", show_alert=True
        )
        return
    mode = (query.data or "").split(":", 2)[2]
    if mode not in _ACCESS_LABELS:
        await query.answer("Неизвестный режим.", show_alert=True)
        return
    storage.set_access_mode(mode)
    await query.answer(f"Режим: {_ACCESS_LABELS[mode]}")
    if query.message is not None:
        with contextlib.suppress(Exception):
            await query.message.edit_text(
                _access_screen_body(), reply_markup=_kb_access_menu()
            )


@wizard_router.callback_query(F.data == "access:add")
async def cb_access_add(query: CallbackQuery, state: FSMContext) -> None:
    user = query.from_user
    if not can_admin(user.id if user else None):
        await query.answer("Только владелец.", show_alert=True)
        return
    await state.set_state(AccessStates.awaiting_co_owner_id)
    if query.message is not None:
        await query.message.answer(
            "Пришли <b>Telegram id</b> того, кому хочешь дать полный "
            "доступ. Это число (например <code>123456789</code>). "
            "Узнать свой id может любой через @userinfobot или прочитав "
            "сообщение от моего /start.\n\n"
            "Этот человек получит права как у тебя: сможет менять "
            "настройки, API-ключи, модели, статистику. Отозвать доступ "
            "можно в «👥 Список доступов».\n\n"
            "Чтобы отменить — пришли <code>/cancel</code>."
        )
    await query.answer()


@wizard_router.message(StateFilter(AccessStates.awaiting_co_owner_id))
async def capture_co_owner_id(message: Message, state: FSMContext) -> None:
    user = message.from_user
    if user is None or not can_admin(user.id):
        await state.clear()
        return
    raw = (message.text or "").strip()
    if raw.lower() in ("/cancel", "отмена", "cancel"):
        await state.clear()
        await message.answer("Отменил. Список доступов не изменён.")
        return
    try:
        new_id = int(raw)
    except ValueError:
        await message.answer(
            "Это не число. Пришли только Telegram id (например "
            "<code>123456789</code>) или /cancel чтобы отменить."
        )
        return
    if new_id == storage.get_owner_id():
        await state.clear()
        await message.answer(
            "Это твой собственный id — ты уже владелец, доп. доступ не нужен."
        )
        return
    added = storage.add_co_owner(new_id)
    await state.clear()
    if not added:
        await message.answer(
            f"Этот id (<code>{new_id}</code>) уже в списке доступов.",
            reply_markup=_kb_access_menu(),
        )
        return
    await message.answer(
        f"Готово. <code>{new_id}</code> теперь имеет полный доступ как у "
        "владельца. Чтобы отозвать — открой «👥 Список доступов».",
        reply_markup=_kb_access_menu(),
    )


@wizard_router.callback_query(F.data == "access:list")
async def cb_access_list(query: CallbackQuery, state: FSMContext) -> None:
    user = query.from_user
    if not can_admin(user.id if user else None):
        await query.answer("Только владелец.", show_alert=True)
        return
    owners = storage.get_co_owners()
    rows: list[list[InlineKeyboardButton]] = []
    for uid in owners:
        rows.append(
            [
                InlineKeyboardButton(
                    text=f"🗑 Отозвать {uid}",
                    callback_data=f"access:revoke:{uid}",
                )
            ]
        )
    rows.append(
        [InlineKeyboardButton(text="↩️ Назад", callback_data="access:menu")]
    )
    kb = InlineKeyboardMarkup(inline_keyboard=rows)
    body = (
        "<b>👥 Список доступов</b>\n\n"
        + (
            "\n".join(f"• <code>{uid}</code>" for uid in owners)
            if owners
            else "<i>пусто</i>"
        )
    )
    if query.message is not None:
        with contextlib.suppress(Exception):
            await query.message.edit_text(body, reply_markup=kb)
    await query.answer()


@wizard_router.callback_query(F.data.startswith("access:revoke:"))
async def cb_access_revoke(query: CallbackQuery, state: FSMContext) -> None:
    user = query.from_user
    if not can_admin(user.id if user else None):
        await query.answer("Только владелец.", show_alert=True)
        return
    try:
        target = int((query.data or "").split(":", 2)[2])
    except ValueError:
        await query.answer("Bad id", show_alert=True)
        return
    removed = storage.remove_co_owner(target)
    await query.answer(
        f"Доступ {'отозван' if removed else 'и так нет'}: {target}"
    )
    # Re-render the access:list screen so the row disappears.
    await cb_access_list(query, state)


@wizard_router.callback_query(F.data == "access:back")
async def cb_access_back(query: CallbackQuery, state: FSMContext) -> None:
    user = query.from_user
    uid = user.id if user else None
    if not can_use(uid):
        await query.answer("Нет доступа.", show_alert=True)
        return
    await state.clear()
    if query.message is not None:
        with contextlib.suppress(Exception):
            await query.message.edit_text(
                "<b>⚙️ Настройки</b>", reply_markup=_kb_settings_menu(uid)
            )
    await query.answer()


# ---- 🐙 GitHub settings (list / info / delete / auto-clone toggle) ----


def _kb_github_menu() -> InlineKeyboardMarkup:
    """Top of the GitHub settings screen.

    Shows one row per cloned repo (with name + short description as the
    button label) plus a row each for the "🔁 Авто-клонирование"
    toggle and back-navigation.
    """
    rows: list[list[InlineKeyboardButton]] = []
    repos = storage.list_github_repos()
    if not repos:
        rows.append(
            [
                InlineKeyboardButton(
                    text="📭 Пусто. Кинь мне ссылку на репо в чат — склонирую.",
                    callback_data="gh:noop",
                )
            ]
        )
    else:
        for repo in repos:
            label = f"📦 {repo['name']}"
            rows.append(
                [
                    InlineKeyboardButton(
                        text=label,
                        callback_data=f"gh:info:{repo['name']}",
                    )
                ]
            )
    auto = storage.get_github_auto_clone()
    rows.append(
        [
            InlineKeyboardButton(
                text=("🔁 Авто-клонирование: ВКЛ" if auto else "🔁 Авто-клонирование: ВЫКЛ"),
                callback_data="gh:auto",
            )
        ]
    )
    rows.append(
        [InlineKeyboardButton(text="↩️ В настройки", callback_data="main:settings")]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _github_menu_body() -> str:
    repos = storage.list_github_repos()
    auto = storage.get_github_auto_clone()
    head = (
        "<b>🐙 GitHub-проекты</b>\n\n"
        "Здесь живут все репо что я склонировал на сервер. Тап на "
        "репо — увижу описание и кнопку удалить (стираю и с диска, и "
        "из памяти бота).\n\n"
        f"<b>🔁 Авто-клонирование:</b> {'ВКЛ' if auto else 'ВЫКЛ'}.\n"
        "Если ВКЛ — кидаешь GitHub-ссылку в чат и просишь что-то "
        "посмотреть → я сам склонирую и почитаю. Если ВЫКЛ — буду "
        "ждать явной команды «склонируй»."
    )
    if not repos:
        return head + "\n\n<i>Пока ни одного репо не склонировано.</i>"
    return head + f"\n\nВсего: <b>{len(repos)}</b>."


@wizard_router.callback_query(F.data == "gh:menu")
async def cb_gh_menu(query: CallbackQuery, state: FSMContext) -> None:
    user = query.from_user
    if not can_use(user.id if user else None):
        await query.answer("Нет доступа.", show_alert=True)
        return
    await state.clear()
    if query.message is not None:
        with contextlib.suppress(Exception):
            await query.message.edit_text(
                _github_menu_body(), reply_markup=_kb_github_menu()
            )
    await query.answer()


@wizard_router.callback_query(F.data == "gh:auto")
async def cb_gh_auto(query: CallbackQuery, state: FSMContext) -> None:
    user = query.from_user
    if not can_admin(user.id if user else None):
        # Toggle is admin-only — guests in public mode can browse repos
        # but can't change the bot's behaviour.
        await query.answer("Только владелец может менять.", show_alert=True)
        return
    current = storage.get_github_auto_clone()
    storage.set_github_auto_clone(not current)
    new_state = "ВКЛЮЧЕНО" if not current else "ВЫКЛЮЧЕНО"
    await query.answer(f"Авто-клонирование: {new_state}", show_alert=False)
    if query.message is not None:
        with contextlib.suppress(Exception):
            await query.message.edit_text(
                _github_menu_body(), reply_markup=_kb_github_menu()
            )


@wizard_router.callback_query(F.data.startswith("gh:info:"))
async def cb_gh_info(query: CallbackQuery, state: FSMContext) -> None:
    user = query.from_user
    if not can_use(user.id if user else None):
        await query.answer("Нет доступа.", show_alert=True)
        return
    name = query.data.split(":", 2)[2] if query.data else ""
    meta = storage.get_github_repo(name)
    if meta is None:
        await query.answer("Репо больше нет на сервере.", show_alert=True)
        if query.message is not None:
            with contextlib.suppress(Exception):
                await query.message.edit_text(
                    _github_menu_body(), reply_markup=_kb_github_menu()
                )
        return
    description = meta.get("description") or "<i>описание не извлечено (нет README)</i>"
    url = meta.get("url") or ""
    body = (
        f"<b>📦 {_html_escape(name)}</b>\n\n"
        f"<b>Описание:</b> {_html_escape(description)}\n\n"
    )
    if url:
        body += f"<b>Источник:</b> <code>{_html_escape(url)}</code>\n\n"
    body += (
        "<b>Пример использования</b> (просто напиши боту в чат):\n"
        f"• «глянь что в проекте {_html_escape(name)}»\n"
        f"• «переключись на {_html_escape(name)} и покажи структуру»\n"
        f"• «в {_html_escape(name)} найди где обрабатывается X»\n\n"
        "Удалить — кнопка ниже. Сотрётся и с диска, и из памяти бота."
    )
    rows: list[list[InlineKeyboardButton]] = []
    if can_admin(user.id if user else None):
        rows.append(
            [
                InlineKeyboardButton(
                    text=f"🗑 Удалить {name}",
                    callback_data=f"gh:del:{name}",
                )
            ]
        )
    rows.append(
        [InlineKeyboardButton(text="↩️ Назад к списку", callback_data="gh:menu")]
    )
    kb = InlineKeyboardMarkup(inline_keyboard=rows)
    if query.message is not None:
        with contextlib.suppress(Exception):
            await query.message.edit_text(body, reply_markup=kb)
    await query.answer()


@wizard_router.callback_query(F.data.startswith("gh:del:"))
async def cb_gh_delete(query: CallbackQuery, state: FSMContext) -> None:
    user = query.from_user
    if not can_admin(user.id if user else None):
        await query.answer("Только владелец может удалять.", show_alert=True)
        return
    name = query.data.split(":", 2)[2] if query.data else ""
    if not name:
        await query.answer("?", show_alert=False)
        return
    # Two-step confirm so a misclick doesn't nuke a 5GB repo.
    rows = [
        [
            InlineKeyboardButton(
                text=f"🗑 Да, удалить {name}",
                callback_data=f"gh:delconfirm:{name}",
            )
        ],
        [
            InlineKeyboardButton(
                text="◀️ Отмена",
                callback_data=f"gh:info:{name}",
            )
        ],
    ]
    body = (
        f"<b>Удалить «{_html_escape(name)}»?</b>\n\n"
        "Стираю папку с диска (rm -rf) и забываю описание в state.json. "
        "Это безвозвратно — гитхаб-исходник не трогаю, но локальную "
        "копию (со всеми твоими изменениями если были) уже не вернёшь."
    )
    if query.message is not None:
        with contextlib.suppress(Exception):
            await query.message.edit_text(
                body, reply_markup=InlineKeyboardMarkup(inline_keyboard=rows)
            )
    await query.answer()


@wizard_router.callback_query(F.data.startswith("gh:delconfirm:"))
async def cb_gh_delete_confirm(query: CallbackQuery, state: FSMContext) -> None:
    user = query.from_user
    if not can_admin(user.id if user else None):
        await query.answer("Только владелец может удалять.", show_alert=True)
        return
    name = query.data.split(":", 2)[2] if query.data else ""
    from .tools import ToolError, delete_project

    try:
        existed = delete_project(name)
    except ToolError as exc:
        await query.answer(f"Ошибка: {exc}", show_alert=True)
        return
    msg = f"🗑 Удалено: <code>{_html_escape(name)}</code>." if existed else (
        "<i>На диске не было</i> — стёр только запись о репо."
    )
    await query.answer("Удалено.", show_alert=False)
    if query.message is not None:
        with contextlib.suppress(Exception):
            await query.message.edit_text(
                msg + "\n\n" + _github_menu_body(),
                reply_markup=_kb_github_menu(),
            )


@wizard_router.callback_query(F.data == "gh:noop")
async def cb_gh_noop(query: CallbackQuery, state: FSMContext) -> None:
    # Stub for the "📭 Пусто" placeholder row so taps don't error.
    await query.answer(
        "Кинь в чат ссылку, например: https://github.com/owner/repo",
        show_alert=False,
    )


# ---- /role — pick a persona for this bot --------------------------------


async def _show_role_picker(
    query: CallbackQuery | None = None, state: FSMContext | None = None
) -> None:
    """Render the role-picker UI; reused by /role command and main-menu button."""
    text = _role_picker_text()
    markup = _kb_role_picker()
    if query is not None and query.message is not None:
        # Try to edit the existing menu in place; if that fails (different
        # message type, etc.), send a fresh one.
        try:
            await query.message.edit_text(text, reply_markup=markup)
        except Exception:  # noqa: BLE001 — edit_text raises on inline-kb mismatch
            await query.message.answer(text, reply_markup=markup)


def _role_picker_text() -> str:
    """The header shown above the persona grid."""
    active = get_persona()
    override = storage.get_persona_override()
    env_persona = os.environ.get("BOT_PERSONA", "boss").strip().lower()
    if override is not None:
        source = (
            f"override (через /role) — env BOT_PERSONA=<code>{_html_escape(env_persona)}</code>"
        )
    else:
        source = "env BOT_PERSONA"
    return (
        "<b>🎭 Сменить роль этого бота</b>\n\n"
        f"Сейчас активна: <b>{active.display_name}</b> "
        f"(<code>{active.key}</code>, {source})\n"
        f"<i>{active.title}</i>\n\n"
        "Жми на роль чтобы её посмотреть и активировать. ▶ — текущая. "
        "После клика я попрошу подтвердить. Все 20 ролей ниже:"
    )


@wizard_router.message(Command("role"))
async def cmd_role(message: Message, state: FSMContext) -> None:
    """Show the role picker to the owner."""
    user = message.from_user
    if user is None or not _is_owner(user.id):
        await message.answer("Только владелец может менять роль.")
        return
    await state.clear()
    await message.answer(_role_picker_text(), reply_markup=_kb_role_picker())


@wizard_router.callback_query(F.data.startswith("role:"))
async def cb_role(query: CallbackQuery, state: FSMContext) -> None:
    if not _is_owner(query.from_user.id):
        await query.answer("Только владелец.", show_alert=True)
        return
    action = (query.data or "").split(":", 1)[1]
    await state.clear()

    if action == "back":
        await query.answer()
        if query.message is not None:
            persona = get_persona()
            await query.message.edit_text(
                f"<b>{persona.display_name}</b> на связи. Ты владелец.\n\n"
                f"<i>{persona.title}</i>\n\n"
                "Что хочешь?",
                reply_markup=_kb_main_after_claim(query.from_user.id if query.from_user else None),
            )
        return

    if action == "info":
        await query.answer()
        if query.message is not None:
            lines = ["<b>ℹ️ Все 20 ролей фермы</b>", ""]
            for p in list_personas():
                lines.append(
                    f"<b>{p.display_name}</b> (<code>{p.key}</code>) — "
                    f"{p.department}/{p.rank}"
                )
                lines.append(f"  <i>{p.title}</i>")
                lines.append(f"  {_html_escape(p.description)}")
                lines.append("")
            lines.append("◀️ Назад — кнопка ниже.")
            await query.message.answer(
                "\n".join(lines),
                reply_markup=InlineKeyboardMarkup(
                    inline_keyboard=[
                        [InlineKeyboardButton(text="◀️ Назад к ролям", callback_data="role:back_to_picker")]
                    ]
                ),
                disable_web_page_preview=True,
            )
        return

    if action == "back_to_picker":
        await query.answer()
        if query.message is not None:
            await query.message.answer(_role_picker_text(), reply_markup=_kb_role_picker())
        return

    if action == "reset":
        storage.clear_persona_override()
        await query.answer("Override снят, возвращаюсь к BOT_PERSONA из env")
        if query.message is not None:
            await query.message.edit_text(_role_picker_text(), reply_markup=_kb_role_picker())
        return

    if action.startswith("pick:"):
        persona_key = action.split(":", 1)[1]
        from .persona import get_persona as _get

        persona = _get(persona_key)
        await query.answer()
        if query.message is not None:
            current = get_persona()
            already = persona.key == current.key
            same_note = (
                "\n\n<i>Сейчас эта роль уже активна — нажми «Использовать» чтобы "
                "закрепить override.</i>"
                if already
                else ""
            )
            await query.message.edit_text(
                f"<b>{persona.display_name}</b>\n"
                f"<i>{persona.title}</i>\n"
                f"Отдел: <code>{persona.department}</code> • Ранг: <code>{persona.rank}</code>\n\n"
                f"{_html_escape(persona.description)}{same_note}",
                reply_markup=_kb_role_confirm(persona.key),
            )
        return

    if action.startswith("use:"):
        persona_key = action.split(":", 1)[1]
        storage.set_persona_override(persona_key)
        from .persona import get_persona as _get

        persona = _get(persona_key)
        await query.answer(f"Роль: {persona.display_name}")
        if query.message is not None:
            await query.message.edit_text(
                f"<b>✅ Готово. Теперь я — {persona.display_name}.</b>\n"
                f"<i>{persona.title}</i>\n\n"
                "Override сохранён в state.json. Чтобы вернуться к роли из "
                "BOT_PERSONA — открой <code>/role</code> → 🗑 Сбросить override.",
                reply_markup=_kb_main_after_claim(query.from_user.id if query.from_user else None),
            )
        return

    await query.answer("Неизвестная команда роли", show_alert=True)


# ---- Misc helpers --------------------------------------------------------


def _provider_label_from_storage() -> str:
    return storage.get_provider()


def _html_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _config_summary() -> str:
    """Pretty summary of the current LLM config, safe for HTML."""
    provider = storage.get_provider()
    keys = storage.list_provider_keys()
    label = "OpenRouter" if provider == "openrouter" else "Кастомный endpoint"
    key_state = keys.get("openrouter", {}).get("masked") or "не задан"
    if provider != "openrouter":
        key_state = keys.get("openrouter", {}).get("masked") or "не задан (нужен)"
    model = storage.get_model() or "—"
    base_url = storage.get_base_url()
    lines = [
        f"<b>{label}</b>",
        f"  Ключ: <code>{key_state}</code>",
        f"  Модель: <code>{_html_escape(model)}</code>",
    ]
    if provider != "openrouter":
        lines.append(f"  URL: <code>{_html_escape(base_url) if base_url else 'не задан'}</code>")
    return "\n".join(lines)


def _devin_handoff_prompt() -> str:
    """The text the owner pastes into a fresh Devin session.

    Devin will read this once and then live as the bot's brain — picking up
    inbox.log, replying via bot.send, etc.
    """
    return """Ты — мой удалённый ассистент с шелл-доступом к серверу, на котором живёт Telegram-бот.

ЧТО НА СЕРВЕРЕ
- Бот развёрнут как Docker-контейнер. Код — в моём GitHub-репо (точную ссылку
  я тебе дам отдельно, обычно это форк Lilush).
- Внутри контейнера: /app/bot/  (Python-пакет с handlers, wizard, agent и т.д.)
- Лог входящих сообщений: data/inbox.log  (бот пишет туда каждое моё сообщение)
- Состояние (ключи, brain, owner): data/state.json
- BOT_TOKEN — в .env рядом с ботом

ТВОЯ РАБОТА
Бот сейчас в режиме brain=devin: он НЕ отвечает мне сам. Когда я что-то пишу в TG, ты:
1. Читаешь свежие строки в data/inbox.log
2. Понимаешь что я хочу
3. Делаешь работу (правишь код, гонишь тесты, исследуешь, что угодно)
4. Отправляешь мне ответ обратно через CLI:
     set -a && source .env && set +a
     python -m bot.send <chat_id> "ответ"
   chat_id ты находишь в той же inbox.log строке.

ЕСЛИ ТЕБЕ НУЖЕН CONTEXT
Команды бота которые я могу запускать в TG (=ты их видишь в inbox.log как kind=cmd):
  /help, /start, /setup     — справка и перенастройка
  /brain, /setbrain auto|devin
  /clone <url>, /projects, /project <name>, /cd, /pwd
  /exec <bash>, /git <args>
  /keys, /setkey, /delkey, /models, /setmodel
  /enable, /disable, /reset

ВАЖНО
- Я (владелец) пингую тебя в нашем чате когда хочу чтобы ты отреагировал. Между пингами ты простаиваешь.
- Если просьба сложная — сначала разбери, потом ОДНИМ сообщением мне в TG пришли план; я скажу «ок» и ты делай.
- Если меняешь код — делай в ветке, делай PR, ссылку шли в TG.

Когда прочитал — ответь мне в этом чате (тут, у Devin) что готов, и я начну писать в TG."""


# ---- «скачай <URL>» / «🎞 Монтажёр» — natural-language download flow ----
#
# Matches a leading «скачай / скачать» followed by an HTTPS URL anywhere
# on the same line. Stays *outside* the slash-command space so the agent
# router below still gets first dibs on plain prose. The trailing URL
# is captured into group 1 — that's what we route into yt-dlp / bash2mp4.
#
# Examples:
#   «скачай https://www.instagram.com/reel/abc»
#   «Скачать https://www.youtube.com/shorts/xyz пожалуйста»
#   «  скачай  https://example.com/video.mp4  »
_DOWNLOAD_PHRASE_RE = re.compile(
    r"^\s*скача(?:й|ть)\s+(https?://\S+)",
    re.IGNORECASE,
)

# GitHub-flavoured URLs are handled by the *agent* (it auto-calls
# clone_repo). We only intercept video-ish URLs here.
_GITHUB_HOSTS = (
    "github.com",
    "gitlab.com",
    "bitbucket.org",
)


def _looks_like_video_url(url: str) -> bool:
    """Heuristic: True if URL is for a video site, not a code repo.

    Routing decision for «скачай <URL>»:
    - github / gitlab / bitbucket  → fall through to the agent (clone)
    - everything else (yt, ig, tiktok, vimeo, direct .mp4 …) → yt-dlp

    Kept deliberately loose — yt-dlp's site list covers 1500+ sources;
    we don't try to enumerate them.
    """
    u = url.strip().lower()
    if not (u.startswith("http://") or u.startswith("https://")):
        return False
    return not any(host in u for host in _GITHUB_HOSTS)


def _downloader_choice() -> str | None:
    """Return saved «/dl» vs «bash2mp4» preference, or ``None`` if unset.

    Stored under ``addons.downloader.choice`` so it survives restarts
    and survives the «изолированные тесты» pytest snapshot.
    """
    from .addons import state as _addon_state

    val = _addon_state.get("downloader", "choice", None)
    if val in ("dl", "bash2mp4"):
        return val
    return None


def _set_downloader_choice(choice: str) -> None:
    from .addons import state as _addon_state

    _addon_state.set_("downloader", "choice", choice)


def _store_pending_download(chat_id: int, url: str) -> None:
    """Stash the URL so the choice callback can read it.

    Callback data is capped at 64 bytes — too tight for full URLs. We
    persist the pending URL under ``addons.downloader.pending.<chat>``
    and just pass «dl:choose:dl|bash» in callback_data.
    """
    from .addons import state as _addon_state

    bucket = _addon_state.get("downloader", "pending", {}) or {}
    bucket[str(chat_id)] = url
    _addon_state.set_("downloader", "pending", bucket)


def _pop_pending_download(chat_id: int) -> str | None:
    from .addons import state as _addon_state

    bucket = _addon_state.get("downloader", "pending", {}) or {}
    url = bucket.pop(str(chat_id), None)
    _addon_state.set_("downloader", "pending", bucket)
    return url


def _kb_downloader_choice() -> InlineKeyboardMarkup:
    """First-time chooser: full pipeline vs simple bash2mp4 download."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="🎬 /dl (нарезка + SEO)",
                    callback_data="dl:choose:dl",
                )
            ],
            [
                InlineKeyboardButton(
                    text="⬇️ bash2mp4 (просто скачать)",
                    callback_data="dl:choose:bash2mp4",
                )
            ],
            [
                InlineKeyboardButton(
                    text="✖️ Отмена",
                    callback_data="dl:choose:cancel",
                )
            ],
        ]
    )


async def _enqueue_simple_download(chat_id: int, url: str) -> int:
    """Enqueue a download job with ``simple_only=True``. Returns job id."""
    from .jobs import get_default_queue

    queue = get_default_queue()
    return await queue.enqueue(
        "download",
        {"url": url, "simple_only": True},
        chat_id=chat_id,
    )


async def _enqueue_full_pipeline_download(chat_id: int, url: str) -> int:
    """Enqueue a normal /dl-flavoured download (chains to analyze/edit/...)."""
    from .jobs import get_default_queue

    queue = get_default_queue()
    return await queue.enqueue(
        "download",
        {"url": url},
        chat_id=chat_id,
    )


@wizard_router.message(F.text.regexp(_DOWNLOAD_PHRASE_RE))
async def on_download_intent(message: Message) -> None:
    """Intercept «скачай <video URL>» before the LLM agent sees it.

    Three paths depending on saved choice in ``addons.downloader.choice``:

    1. **Unset** — present a 2-button chooser («/dl полный конвейер» /
       «bash2mp4 простое скачивание»). The user's URL is stashed so the
       callback handler can act on it.
    2. **«dl»** — kick the existing 5-stage pipeline immediately
       (download → analyze → edit → seo → publish) via the standard
       ``/dl`` enqueue path.
    3. **«bash2mp4»** — enqueue a *simple-only* download. ``DownloaderWorker``
       sees ``simple_only=True`` and sends the resulting file back with a
       «🎞 Монтажёр» button instead of auto-chaining.

    GitHub / GitLab / Bitbucket URLs are intentionally NOT intercepted —
    they fall through to the agent which has its own ``clone_repo`` tool.
    """
    if not message.text:
        return
    if not can_use(message.from_user.id if message.from_user else None):
        return
    match = _DOWNLOAD_PHRASE_RE.match(message.text)
    if not match:
        return
    url = match.group(1)
    if not _looks_like_video_url(url):
        # Fall through to the default agent router — it knows how to
        # clone github repos.
        return

    chat_id = message.chat.id
    choice = _downloader_choice()

    if choice == "dl":
        await _enqueue_full_pipeline_download(chat_id, url)
        await message.answer(
            f"🎬 <b>Принято в полный конвейер</b>\n\n"
            f"<a href=\"{url}\">ссылка</a> → анализ → нарезка → SEO → "
            f"публикация.\n\n"
            f"Прогресс — команда <code>/jobs</code>. Когда готово — пришлю файл."
        )
        return

    if choice == "bash2mp4":
        await _enqueue_simple_download(chat_id, url)
        await message.answer(
            f"⬇️ <b>Скачиваю</b>\n\n"
            f"<a href=\"{url}\">ссылка</a> → пришлю файл + кнопку <b>🎞 Монтажёр</b>.\n\n"
            f"Минута-две на yt-dlp; если будет ошибка — расскажу здесь же."
        )
        return

    # First-time: ask which path to use.
    _store_pending_download(chat_id, url)
    await message.answer(
        "<b>Чем качать?</b>\n\n"
        "🎬 <b>/dl</b> — полный конвейер: "
        "<i>скачивание → нарезка → SEO → готовый клип</i>\n\n"
        "⬇️ <b>bash2mp4</b> — просто скачать файл, потом можно вручную нажать "
        "<b>🎞 Монтажёр</b>\n\n"
        "Выбор запомню, в следующий раз спрашивать не буду — "
        "сменить можно в ⚙️ Настройки.",
        reply_markup=_kb_downloader_choice(),
    )


@wizard_router.callback_query(F.data.startswith("dl:choose:"))
async def cb_dl_choose(query: CallbackQuery) -> None:
    """Handle the chooser buttons posted by ``on_download_intent``."""
    if not can_use(query.from_user.id if query.from_user else None):
        await query.answer("Нет доступа.", show_alert=True)
        return
    choice = (query.data or "").split(":", 2)[-1]
    if choice not in ("dl", "bash2mp4", "cancel"):
        await query.answer("Неизвестный выбор", show_alert=True)
        return

    chat_id = query.message.chat.id if query.message else None
    if chat_id is None:
        await query.answer("Нет чата", show_alert=True)
        return

    url = _pop_pending_download(chat_id)

    if choice == "cancel":
        await query.answer("Отменено.")
        if query.message is not None:
            with contextlib.suppress(Exception):
                await query.message.edit_text("✖️ Скачивание отменено.")
        return

    if not url:
        await query.answer("Ссылка не найдена — повтори «скачай <URL>».", show_alert=True)
        return

    _set_downloader_choice(choice)
    await query.answer(f"Запомнил: {choice}")

    if choice == "dl":
        await _enqueue_full_pipeline_download(chat_id, url)
        if query.message is not None:
            with contextlib.suppress(Exception):
                await query.message.edit_text(
                    f"🎬 <b>Принято в полный конвейер</b>\n\n"
                    f"<a href=\"{url}\">ссылка</a> → анализ → нарезка → "
                    f"SEO → публикация.\n\n"
                    f"Прогресс — команда <code>/jobs</code>."
                )
        return

    # bash2mp4
    await _enqueue_simple_download(chat_id, url)
    if query.message is not None:
        with contextlib.suppress(Exception):
            await query.message.edit_text(
                f"⬇️ <b>Скачиваю</b>\n\n"
                f"<a href=\"{url}\">ссылка</a> → пришлю файл + кнопку "
                f"<b>🎞 Монтажёр</b>.\n\n"
                f"Минута-две; если будет ошибка — расскажу здесь же."
            )


@wizard_router.callback_query(F.data.startswith("dl:edit:"))
async def cb_dl_edit(query: CallbackQuery) -> None:
    """«🎞 Монтажёр» button on the «видео скачано» message.

    Reads the original download job's result, then enqueues an
    ``analyze`` job pointing at the same ``source_path``. Analyze
    chains to edit → seo → publish via the normal worker pipeline, so
    the user gets the full edit treatment without re-downloading.
    """
    if not can_use(query.from_user.id if query.from_user else None):
        await query.answer("Нет доступа.", show_alert=True)
        return
    try:
        job_id = int((query.data or "").split(":")[-1])
    except (TypeError, ValueError):
        await query.answer("Битый ID джобы", show_alert=True)
        return

    from .jobs import get_default_queue

    queue = get_default_queue()
    job = await queue.get(job_id)
    if job is None:
        await query.answer("Джоба не найдена", show_alert=True)
        return
    if not job.result or not job.result.get("source_path"):
        await query.answer("У этой джобы нет файла", show_alert=True)
        return

    chat_id = query.message.chat.id if query.message else job.chat_id
    if chat_id is None:
        await query.answer("Нет чата", show_alert=True)
        return

    # Enqueue analyze with the existing source file — the pipeline
    # picks up from stage 2 and runs through edit / seo / publish.
    await queue.enqueue(
        "analyze",
        {
            "source_path": job.result["source_path"],
            "duration_s": job.result.get("duration_s") or 0.0,
            "language": job.result.get("language"),
        },
        chat_id=chat_id,
        parent_id=job.id,
    )
    await query.answer("Запускаю монтажёр…")
    if query.message is not None:
        with contextlib.suppress(Exception):
            await query.message.reply(
                "🎞 <b>Монтажёр запущен</b>\n\n"
                "<i>анализ → нарезка → SEO → публикация</i>\n\n"
                "Когда нарезки будут готовы — пришлю их сюда же. "
                "Прогресс — команда <code>/jobs</code>."
            )


def _bash2mp4_help_text() -> str:
    """Help message printed when the user clicks «⬇️ Скачать (bash2mp4)».

    Reused both after a fresh install and on subsequent clicks (when
    the install has already happened). Walks the user through the
    natural-language flow they asked for: «скачай <URL>» → bot picks
    bash2mp4 (or asks the first time) → file arrives with a «Монтажёр»
    button attached.
    """
    return (
        "<b>⬇️ bash2mp4 — как пользоваться</b>\n\n"
        "Просто пиши в чате <code>скачай</code> + ссылка. Бот сам поймёт "
        "что хочешь скачать видео и пришлёт файл с кнопкой "
        "<b>🎞 Монтажёр</b>.\n\n"
        "<b>Примеры:</b>\n"
        "• <code>скачай https://www.instagram.com/reel/DY7M7TDM-Lv</code>\n"
        "• <code>скачай https://www.youtube.com/shorts/2TEyTiQpYHU</code>\n"
        "• <code>скачай https://www.tiktok.com/@user/video/12345</code>\n\n"
        "<b>Что произойдёт дальше:</b>\n"
        "1. В первый раз бот спросит чем качать: "
        "<b>🎬 /dl</b> (полный конвейер с нарезкой) "
        "или <b>⬇️ bash2mp4</b> (просто скачать). Выбор запомнит.\n"
        "2. Когда скачает — пришлёт видео-файл прямо в чат.\n"
        "3. Под видео будет кнопка <b>🎞 Монтажёр</b> — "
        "нажмёшь → пайплайн нарезки и SEO запустится на этом же файле."
    )


# ---- 📝 Markitdown — settings screen + commands -------------------------
#
# Markitdown converts URLs (YouTube, Vimeo, PDFs, docx, …) to markdown.
# Lives under data/projects/markitdown/ once installed. The bot exposes
# it through two natural-language commands ("дай данные <URL>" and
# "посмотри и расскажи о чём там <URL>") and a settings screen reached
# from the main menu.

_MARKITDOWN_REPO = "https://github.com/microsoft/markitdown"


def _markitdown_installed() -> bool:
    """True iff ``data/projects/markitdown/`` exists with a pyproject."""
    from .config import DATA_DIR

    p = DATA_DIR / "projects" / "markitdown"
    # The repo has a pyproject inside packages/markitdown/ — that's the
    # marker we look for so a bare empty dir doesn't fool us.
    return (p / "packages" / "markitdown" / "pyproject.toml").is_file()


def _markitdown_auto_enabled() -> bool:
    """User toggle: when ON, «дай данные <URL>» runs markitdown directly.

    When OFF the commands still work, but the bot prefixes its reply
    with a hint that the feature is off and asks the user to enable it.
    Kept as a separate setting (rather than just relying on install
    state) because some users may want to keep the repo cloned but
    have the auto-handler quiet while debugging.
    """
    from .addons import state as _addon_state

    return bool(_addon_state.get("markitdown", "auto", False))


def _set_markitdown_auto(enabled: bool) -> None:
    from .addons import state as _addon_state

    _addon_state.set_("markitdown", "auto", bool(enabled))


def _kb_markitdown_menu() -> InlineKeyboardMarkup:
    """5-button screen the user asked for: on, off, install, uninstall, back."""
    installed = _markitdown_installed()
    auto_on = _markitdown_auto_enabled()
    auto_marker_on = "▶ " if auto_on else ""
    auto_marker_off = "" if auto_on else "▶ "
    install_label = (
        "✅ Установлен" if installed else "⬇️ Скачать с github"
    )
    rows: list[list[InlineKeyboardButton]] = [
        [
            InlineKeyboardButton(
                text=f"{auto_marker_on}✅ Авто-включение",
                callback_data="md:auto:on",
            )
        ],
        [
            InlineKeyboardButton(
                text=f"{auto_marker_off}⛔ Авто-выключение",
                callback_data="md:auto:off",
            )
        ],
        [
            InlineKeyboardButton(
                text=install_label,
                callback_data="md:install",
            )
        ],
    ]
    if installed:
        rows.append(
            [
                InlineKeyboardButton(
                    text="🗑 Удалить", callback_data="md:uninstall"
                )
            ]
        )
    rows.append(
        [InlineKeyboardButton(text="↩️ Назад", callback_data="md:back")]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _markitdown_screen_text() -> str:
    """Human-readable status + usage examples shown above the keyboard.

    Two free-text commands are advertised: «дай данные <URL>» dumps the
    raw markdown back to chat (as a file when long), «посмотри и
    расскажи о чём там <URL>» feeds the markdown into the LLM and
    summarises. Both rely on having markitdown installed AND auto-mode
    enabled — the status block at the top makes both states visible.
    """
    installed = _markitdown_installed()
    auto = _markitdown_auto_enabled()
    status_line = (
        f"Установка: {'✅ есть' if installed else '❌ нет'}    "
        f"Авто: {'✅ вкл' if auto else '⛔ выкл'}"
    )
    return (
        "<b>📝 Markitdown</b>\n\n"
        f"{status_line}\n\n"
        "<b>Команды</b> (когда установлено + авто-включено):\n\n"
        "• <code>дай данные https://www.youtube.com/shorts/XXX</code>\n"
        "  → бот шлёт сам markdown-файл с расшифровкой\n\n"
        "• <code>посмотри и расскажи о чём там https://www.youtube.com/shorts/XXX</code>\n"
        "  → бот сам прочитает markdown и расскажет своими словами\n\n"
        f"<i>Репозиторий: <a href=\"{_MARKITDOWN_REPO}\">microsoft/markitdown</a></i>"
    )


async def _show_markitdown_screen(query: CallbackQuery) -> None:
    """Render the markitdown settings screen on the user's chat."""
    text = _markitdown_screen_text()
    kb = _kb_markitdown_menu()
    if query.message is not None:
        with contextlib.suppress(Exception):
            await query.message.edit_text(
                text, reply_markup=kb, disable_web_page_preview=True
            )


@wizard_router.message(Command("markitdown"))
async def cmd_markitdown(message: Message) -> None:
    """Shortcut command to open the markitdown screen without the menu."""
    if not can_use(message.from_user.id if message.from_user else None):
        return
    await message.answer(
        _markitdown_screen_text(),
        reply_markup=_kb_markitdown_menu(),
        disable_web_page_preview=True,
    )


@wizard_router.callback_query(F.data.startswith("md:"))
async def cb_markitdown(query: CallbackQuery) -> None:
    """All «md:*» callbacks: auto toggle, install / uninstall, back.

    ``md:auto:on|off``   — flip the auto-handler flag
    ``md:install``       — clone microsoft/markitdown into data/projects
    ``md:uninstall``     — remove the cloned project dir
    ``md:back``          — return to the main menu
    """
    uid = query.from_user.id if query.from_user else None
    if not can_admin(uid):
        await query.answer("Только владелец.", show_alert=True)
        return
    action = (query.data or "").split(":", 1)[1]

    if action == "back":
        await query.answer()
        if query.message is not None:
            with contextlib.suppress(Exception):
                await query.message.edit_text(
                    "<b>Главное меню</b>",
                    reply_markup=_kb_main_menu(uid),
                )
        return

    if action == "auto:on":
        _set_markitdown_auto(True)
        await query.answer("Авто-режим включён.")
        await _show_markitdown_screen(query)
        return

    if action == "auto:off":
        _set_markitdown_auto(False)
        await query.answer("Авто-режим выключен.")
        await _show_markitdown_screen(query)
        return

    if action == "install":
        if _markitdown_installed():
            await query.answer("Уже установлен.")
            await _show_markitdown_screen(query)
            return
        await query.answer("Клонирую…")
        if query.message is not None:
            from .handlers import run_work_batch

            await run_work_batch(
                query.message,
                [f"/clone {_MARKITDOWN_REPO}"],
                title="📝 Markitdown — install",
                user_id=uid,
            )
            await query.message.answer(
                "Готово. Авто-режим можно включить кнопкой "
                "«✅ Авто-включение»."
            )
        return

    if action == "uninstall":
        from .config import DATA_DIR

        target = DATA_DIR / "projects" / "markitdown"
        if not target.is_dir():
            await query.answer("Уже удалён.")
            await _show_markitdown_screen(query)
            return
        await query.answer("Удаляю…")
        import shutil

        try:
            shutil.rmtree(target)
        except OSError as exc:
            logger.exception("markitdown uninstall failed")
            if query.message is not None:
                await query.message.answer(
                    f"Ошибка удаления: <code>{exc}</code>"
                )
            return
        if query.message is not None:
            await query.message.answer("🗑 Markitdown удалён.")
        await _show_markitdown_screen(query)
        return

    await query.answer("Неизвестное действие", show_alert=True)


# Pattern matchers for the two natural-language markitdown commands.
# Match the verb-phrase + a URL anywhere on the rest of the line so the
# user can pad with «пожалуйста / kindly / а потом …» without breaking
# routing. Both are routed only when ``addons.markitdown.auto`` is true
# (the user explicitly asked for an auto on/off switch).
_MD_DUMP_RE = re.compile(
    r"^\s*дай\s+данные\s+(https?://\S+)", re.IGNORECASE
)
_MD_SUMMARY_RE = re.compile(
    r"^\s*посмотри\s+и\s+расскажи(?:\s+о\s+чём\s+там)?\s+(https?://\S+)",
    re.IGNORECASE,
)


async def _run_markitdown(url: str) -> tuple[bool, str]:
    """Invoke markitdown on ``url`` and return ``(ok, body_or_error)``.

    Uses subprocess so a broken page can't crash the bot loop. Looks
    for the ``markitdown`` console script in the active venv first,
    falls back to ``python -m markitdown``. If neither is available
    the helper returns ``(False, instructions)`` so the caller can
    forward that text to the user.
    """
    import asyncio
    import shutil
    import sys

    md_bin = shutil.which("markitdown")
    if md_bin:
        cmd = [md_bin, url]
    else:
        cmd = [sys.executable, "-m", "markitdown", url]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=120
        )
    except asyncio.TimeoutError:
        return (False, "markitdown timed out after 120s")
    except FileNotFoundError:
        return (
            False,
            "markitdown не установлен. Запусти "
            "<code>pip install ./data/projects/markitdown/packages/markitdown[all]</code> "
            "в окружении бота.",
        )

    if proc.returncode != 0:
        err = (stderr or b"").decode("utf-8", "replace").strip()
        if "No module named" in err and "markitdown" in err:
            return (
                False,
                "Модуль markitdown отсутствует в окружении бота. "
                "Включи кнопку «⬇️ Скачать с github» и установи через "
                "<code>pip install -e ./data/projects/markitdown/packages/markitdown[all]</code>.",
            )
        return (False, f"markitdown упал: <code>{err[:500]}</code>")

    body = (stdout or b"").decode("utf-8", "replace").strip()
    if not body:
        return (False, "markitdown вернул пустой ответ.")
    return (True, body)


@wizard_router.message(F.text.regexp(_MD_DUMP_RE))
async def on_markitdown_dump(message: Message) -> None:
    """«дай данные <URL>» → send the raw markdown back as a file."""
    if not message.text:
        return
    if not can_use(message.from_user.id if message.from_user else None):
        return
    match = _MD_DUMP_RE.match(message.text)
    if not match:
        return
    url = match.group(1)
    if not _markitdown_auto_enabled():
        await message.answer(
            "<b>📝 Markitdown выключен.</b>\n\n"
            "Открой меню → <b>📝 Markitdown</b> → <b>✅ Авто-включение</b>, "
            "и потом повтори команду."
        )
        return
    if not _markitdown_installed():
        await message.answer(
            "<b>📝 Markitdown не установлен.</b>\n\n"
            "Открой меню → <b>📝 Markitdown</b> → <b>⬇️ Скачать с github</b>."
        )
        return

    await message.answer(f"📝 Получаю данные с <code>{url}</code>…")
    ok, body = await _run_markitdown(url)
    if not ok:
        await message.answer(f"❌ {body}")
        return
    await _send_markdown_dump(message, url, body)


@wizard_router.message(F.text.regexp(_MD_SUMMARY_RE))
async def on_markitdown_summary(message: Message) -> None:
    """«посмотри и расскажи о чём там <URL>» → markdown → LLM summary."""
    if not message.text:
        return
    if not can_use(message.from_user.id if message.from_user else None):
        return
    match = _MD_SUMMARY_RE.match(message.text)
    if not match:
        return
    url = match.group(1)
    if not _markitdown_auto_enabled():
        await message.answer(
            "<b>📝 Markitdown выключен.</b>\n\n"
            "Открой меню → <b>📝 Markitdown</b> → <b>✅ Авто-включение</b>, "
            "и потом повтори команду."
        )
        return
    if not _markitdown_installed():
        await message.answer(
            "<b>📝 Markitdown не установлен.</b>\n\n"
            "Открой меню → <b>📝 Markitdown</b> → <b>⬇️ Скачать с github</b>."
        )
        return

    await message.answer(f"🔍 Смотрю <code>{url}</code>…")
    ok, body = await _run_markitdown(url)
    if not ok:
        await message.answer(f"❌ {body}")
        return

    try:
        from .agent import oneshot_summary

        prompt = (
            "Кратко и по делу расскажи о чём этот материал. "
            "Раздели абзацы пустыми строками. Сохраняй URL-ы как "
            "кликабельные ссылки (HTML <a href>).\n\n"
            f"--- материал начало ---\n{body[:30000]}\n--- материал конец ---"
        )
        summary = await oneshot_summary(prompt, purpose="markitdown_summary")
    except Exception:  # noqa: BLE001
        logger.exception("markitdown summary LLM call failed")
        await message.answer(
            "Не смог получить саммари от LLM. Сырой markdown — пришлю файлом."
        )
        await _send_markdown_dump(message, url, body)
        return

    if not summary:
        await message.answer(
            "LLM вернул пустой ответ. Сырой markdown — файлом ниже."
        )
        await _send_markdown_dump(message, url, body)
        return

    await message.answer(summary, disable_web_page_preview=False)


async def _send_markdown_dump(
    message: Message, url: str, body: str
) -> None:
    """Pretty-print a markdown blob: inline if short, file if long.

    Telegram caps message bodies at 4096 chars. Anything bigger gets
    written to a tempfile and sent as a document attachment so the
    user has the full markdown they can grep through locally.
    """
    if len(body) <= 3500:
        await message.answer(
            f"<b>📝 Markdown</b>\n\n<pre>{_html_escape(body)}</pre>",
            disable_web_page_preview=True,
        )
        return

    import tempfile

    from aiogram.types import FSInputFile

    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".md", delete=False
    ) as fh:
        fh.write(f"# {url}\n\n")
        fh.write(body)
        path = fh.name
    try:
        await message.answer_document(
            FSInputFile(path, filename="markitdown.md"),
            caption=f"📝 markdown для <code>{url}</code>",
        )
    finally:
        os.unlink(path)


# ---- Mutually-exclusive downloader picker -------------------------------
#
# Each downloader («📥 /dl полный конвейер» and «⬇️ bash2mp4 простой») has
# its own screen with a «✅ Использовать» button. Pressing «Использовать»
# on one auto-disables the other because both write to the SAME setting
# (``addons.downloader.choice``) — there are only two valid values, so
# flipping one inherently turns the other off. That's the «одно
# отключает другого» behaviour the user asked for.

def _bash2mp4_installed() -> bool:
    """True iff ``data/projects/bash2mp4/`` exists with a setup.sh.

    Mirror of ``_markitdown_installed()`` — separate helper because the
    marker file is different (bash2mp4 has setup.sh, not pyproject).
    """
    from .config import DATA_DIR

    p = DATA_DIR / "projects" / "bash2mp4"
    return p.is_dir() and (
        (p / "setup.sh").is_file() or any(p.glob("*.sh"))
    )


def _yt_cookies_path() -> Path:
    """Filesystem location for user-uploaded YouTube cookies.txt.

    The downloader auto-picks up this file (via
    ``YT_COOKIES_DEFAULT_PATH`` in :mod:`bot.config`), so saving here
    is the only step required to flip yt-dlp into authenticated mode.
    """
    from .config import DATA_DIR

    return DATA_DIR / "yt_cookies.txt"


def _yt_cookies_present() -> bool:
    """True iff a cookies.txt has been uploaded by the user."""
    p = _yt_cookies_path()
    return p.is_file() and p.stat().st_size > 0


def _kb_dl_screen() -> InlineKeyboardMarkup:
    """Inline keyboard for the «📥 Скачать видео» (/dl) settings screen."""
    active = _downloader_choice() == "dl"
    has_cookies = _yt_cookies_present()
    rows: list[list[InlineKeyboardButton]] = []
    if active:
        rows.append(
            [
                InlineKeyboardButton(
                    text="✅ Активный режим",
                    callback_data="dl:noop",
                )
            ]
        )
    else:
        rows.append(
            [
                InlineKeyboardButton(
                    text="✅ Использовать /dl",
                    callback_data="dl:use:dl",
                )
            ]
        )
    # «🍪 Загрузить cookies» — opens upload flow. The button label
    # mirrors current state: «загрузить» if missing, «обновить» if
    # already present.
    if has_cookies:
        rows.append(
            [
                InlineKeyboardButton(
                    text="🍪 Cookies загружены (обновить)",
                    callback_data="dl:cookies:upload",
                ),
                InlineKeyboardButton(
                    text="🗑 Удалить cookies",
                    callback_data="dl:cookies:delete",
                ),
            ]
        )
    else:
        rows.append(
            [
                InlineKeyboardButton(
                    text="🍪 Загрузить cookies (для YouTube)",
                    callback_data="dl:cookies:upload",
                )
            ]
        )
    rows.append(
        [InlineKeyboardButton(text="↩️ Назад", callback_data="dl:back")]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _kb_bash2mp4_screen() -> InlineKeyboardMarkup:
    """Inline keyboard for the «⬇️ Скачать (bash2mp4)» settings screen."""
    installed = _bash2mp4_installed()
    active = _downloader_choice() == "bash2mp4"
    rows: list[list[InlineKeyboardButton]] = []
    if not installed:
        # Shouldn't normally be shown — install button on main menu
        # opens the cloning flow. Just in case render a re-install hint.
        rows.append(
            [
                InlineKeyboardButton(
                    text="⬇️ Установить bash2mp4",
                    callback_data="main:install:downloader",
                )
            ]
        )
    else:
        if active:
            rows.append(
                [
                    InlineKeyboardButton(
                        text="✅ Активный режим",
                        callback_data="dl:noop",
                    )
                ]
            )
        else:
            rows.append(
                [
                    InlineKeyboardButton(
                        text="✅ Использовать bash2mp4",
                        callback_data="dl:use:bash2mp4",
                    )
                ]
            )
        rows.append(
            [
                InlineKeyboardButton(
                    text="🗑 Удалить bash2mp4",
                    callback_data="dl:remove:bash2mp4",
                )
            ]
        )
    rows.append(
        [InlineKeyboardButton(text="↩️ Назад", callback_data="dl:back")]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _dl_screen_text() -> str:
    """Status + usage for the /dl screen."""
    active = _downloader_choice() == "dl"
    status_line = (
        "✅ <b>Активен</b>" if active else "⛔ <b>Неактивен</b>"
    )
    cookies_line = (
        "🍪 <b>cookies.txt</b>: загружен" if _yt_cookies_present()
        else "🍪 <b>cookies.txt</b>: не загружен"
    )
    return (
        "<b>📥 Скачать видео — /dl (yt-dlp пайплайн)</b>\n\n"
        f"Статус: {status_line}\n"
        f"{cookies_line}\n\n"
        "Полный конвейер: <i>скачивание → анализ → нарезка → SEO → "
        "публикация</i>. Поддерживает YouTube, Instagram, TikTok, "
        "Vimeo, прямые .mp4 — всё что глотает yt-dlp.\n\n"
        "<b>Как пользоваться:</b>\n"
        "• Команда: <code>/dl &lt;url&gt;</code>\n"
        "• Просто текстом: <code>скачай &lt;url&gt;</code> "
        "(подхватит активный режим)\n\n"
        "<b>Если YouTube ругается «Sign in to confirm you're not a bot»</b> "
        "— нажми <b>🍪 Загрузить cookies</b> и пришли файл "
        "<code>cookies.txt</code>, экспортированный из браузера "
        "(расширение «Get cookies.txt LOCALLY» в Chrome / Edge).\n\n"
        "Кнопка <b>✅ Использовать</b> делает этот режим активным "
        "(автоматически отключает bash2mp4)."
    )


def _bash2mp4_screen_text() -> str:
    """Status + usage for the bash2mp4 screen."""
    installed = _bash2mp4_installed()
    active = _downloader_choice() == "bash2mp4"
    install_line = (
        "✅ <b>Установлен</b>" if installed else "❌ <b>Не установлен</b>"
    )
    active_line = (
        "✅ <b>Активен</b>" if active else "⛔ <b>Неактивен</b>"
    )
    return (
        "<b>⬇️ Скачать видео — bash2mp4 (простой режим)</b>\n\n"
        f"Установка: {install_line}    Статус: {active_line}\n\n"
        "Простое скачивание: <i>файл сразу приходит в чат + кнопка "
        "«🎞 Монтажёр»</i>. Без авто-нарезки, без SEO, без "
        "публикации — только сырой файл.\n\n"
        "<b>Как пользоваться:</b>\n"
        "• Просто текстом: <code>скачай &lt;url&gt;</code>\n"
        "• Когда придёт файл, нажми <b>🎞 Монтажёр</b> чтобы "
        "запустить пайплайн нарезки на этом же файле.\n\n"
        "Кнопка <b>✅ Использовать</b> делает этот режим активным "
        "(автоматически отключает /dl)."
    )


async def _show_download_screen(query: CallbackQuery) -> None:
    """Render the /dl settings screen (status + Use button + back)."""
    if query.message is None:
        return
    with contextlib.suppress(Exception):
        await query.message.edit_text(
            _dl_screen_text(),
            reply_markup=_kb_dl_screen(),
            disable_web_page_preview=True,
        )


async def _show_bash2mp4_screen(query: CallbackQuery) -> None:
    """Render the bash2mp4 settings screen."""
    if query.message is None:
        return
    # ``edit_text`` fails if the previous message wasn't a text one
    # (e.g. it was the install log streamed via run_work_batch). Fall
    # back to ``answer`` in that case so the user always sees the
    # screen.
    try:
        await query.message.edit_text(
            _bash2mp4_screen_text(),
            reply_markup=_kb_bash2mp4_screen(),
            disable_web_page_preview=True,
        )
    except Exception:  # noqa: BLE001
        await query.message.answer(
            _bash2mp4_screen_text(),
            reply_markup=_kb_bash2mp4_screen(),
            disable_web_page_preview=True,
        )


@wizard_router.callback_query(F.data == "dl:back")
async def cb_dl_back(query: CallbackQuery) -> None:
    """«↩️ Назад» on a downloader screen → return to the main menu."""
    uid = query.from_user.id if query.from_user else None
    await query.answer()
    if query.message is not None:
        with contextlib.suppress(Exception):
            await query.message.edit_text(
                "<b>Главное меню</b>",
                reply_markup=_kb_main_menu(uid),
            )


@wizard_router.callback_query(F.data == "dl:noop")
async def cb_dl_noop(query: CallbackQuery) -> None:
    """The «✅ Активный режим» pseudo-button — just acknowledge."""
    await query.answer("Этот режим уже активен.")


@wizard_router.callback_query(F.data.startswith("dl:use:"))
async def cb_dl_use(query: CallbackQuery) -> None:
    """«✅ Использовать <name>» — switch the active downloader.

    Both screens share the same backing key
    (``addons.downloader.choice``), so activating one inherently
    deactivates the other. Refreshes the same screen so the user sees
    the «✅ Активен» indicator turn green.
    """
    uid = query.from_user.id if query.from_user else None
    if not can_admin(uid):
        await query.answer("Только владелец.", show_alert=True)
        return

    choice = (query.data or "").split(":", 2)[-1]
    if choice not in ("dl", "bash2mp4"):
        await query.answer("Неизвестный режим", show_alert=True)
        return

    # bash2mp4 needs to be installed before it can be «used» — otherwise
    # we'd lock the user into a dead-end where «скачай <URL>» tries
    # bash2mp4 and silently fails.
    if choice == "bash2mp4" and not _bash2mp4_installed():
        await query.answer(
            "bash2mp4 не установлен. Нажми «⬇️ Скачать (bash2mp4)» в "
            "главном меню чтобы поставить.",
            show_alert=True,
        )
        return

    _set_downloader_choice(choice)
    await query.answer(
        "Активный режим: /dl" if choice == "dl" else "Активный режим: bash2mp4"
    )
    # Refresh the screen the user clicked on.
    if choice == "dl":
        await _show_download_screen(query)
    else:
        await _show_bash2mp4_screen(query)


@wizard_router.callback_query(F.data == "dl:remove:bash2mp4")
async def cb_dl_remove_bash2mp4(query: CallbackQuery) -> None:
    """«🗑 Удалить bash2mp4» — wipe the cloned project dir.

    Also clears the active choice if it pointed at bash2mp4, so the
    user doesn't end up with «скачай <URL>» pointing at a deleted
    binary. The /dl screen survives — yt-dlp lives inside the bot's
    venv, not in data/projects/.
    """
    uid = query.from_user.id if query.from_user else None
    if not can_admin(uid):
        await query.answer("Только владелец.", show_alert=True)
        return

    from .config import DATA_DIR

    target = DATA_DIR / "projects" / "bash2mp4"
    if not target.is_dir():
        await query.answer("Уже удалён.")
        await _show_bash2mp4_screen(query)
        return

    import shutil

    try:
        shutil.rmtree(target)
    except OSError as exc:
        logger.exception("bash2mp4 uninstall failed")
        await query.answer(f"Ошибка удаления: {exc}", show_alert=True)
        return

    # If bash2mp4 was active — fall back to «unset», so the next
    # «скачай <URL>» prompts the user again instead of running a
    # non-existent binary.
    if _downloader_choice() == "bash2mp4":
        from .addons import state as _addon_state

        _addon_state.delete("downloader", "choice")

    await query.answer("🗑 bash2mp4 удалён.")
    await _show_bash2mp4_screen(query)


# ---- YouTube cookies upload ---------------------------------------------
#
# YouTube periodically tightens its anti-bot gate; when that happens
# yt-dlp throws «Sign in to confirm you're not a bot». The only
# reliable workaround is to feed it the cookies from an authenticated
# browser session. We expose this as a button on the /dl screen so
# the owner can paste a freshly-exported cookies.txt without touching
# the filesystem.

@wizard_router.callback_query(F.data == "dl:cookies:upload")
async def cb_dl_cookies_upload(query: CallbackQuery, state: FSMContext) -> None:
    """«🍪 Загрузить cookies» — switch to the «waiting for file» mode."""
    uid = query.from_user.id if query.from_user else None
    if not can_admin(uid):
        await query.answer("Только владелец.", show_alert=True)
        return
    await state.set_state(SetupStates.awaiting_yt_cookies_file)
    await query.answer()
    if query.message is not None:
        await query.message.answer(
            "🍪 <b>Жду cookies.txt</b>\n\n"
            "Экспортируй cookies из своего браузера в формате Netscape "
            "и пришли файл сюда документом.\n\n"
            "<b>Как экспортировать:</b>\n"
            "1. Поставь расширение <b>«Get cookies.txt LOCALLY»</b> "
            "(<a href=\"https://chromewebstore.google.com/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc\">Chrome</a> / Edge)\n"
            "2. Открой <a href=\"https://www.youtube.com/\">youtube.com</a>, "
            "залогинься\n"
            "3. Нажми на иконку расширения → <b>Export</b> → сохрани файл\n"
            "4. Пришли этот файл сюда\n\n"
            "Отмена — /cancel",
            disable_web_page_preview=True,
        )


@wizard_router.callback_query(F.data == "dl:cookies:delete")
async def cb_dl_cookies_delete(query: CallbackQuery) -> None:
    """«🗑 Удалить cookies» — wipe the saved cookies file."""
    uid = query.from_user.id if query.from_user else None
    if not can_admin(uid):
        await query.answer("Только владелец.", show_alert=True)
        return
    p = _yt_cookies_path()
    if p.exists():
        try:
            p.unlink()
        except OSError as exc:
            logger.exception("yt cookies delete failed")
            await query.answer(f"Ошибка: {exc}", show_alert=True)
            return
    await query.answer("🗑 cookies.txt удалён.")
    await _show_download_screen(query)


@wizard_router.message(
    StateFilter(SetupStates.awaiting_yt_cookies_file), F.document
)
async def capture_yt_cookies(message: Message, state: FSMContext) -> None:
    """Capture an uploaded cookies.txt and save it to ``data/yt_cookies.txt``."""
    if not _is_owner(message.from_user.id if message.from_user else 0):
        return
    if message.document is None:
        return

    name = (message.document.file_name or "").lower()
    # Be permissive on extensions — some exports name the file
    # ``cookies`` without ``.txt``. Accept anything with ``cookie`` in
    # the name or a plain ``.txt`` extension.
    if not (
        "cookie" in name
        or name.endswith(".txt")
    ):
        await message.answer(
            "Жду текстовый файл с cookies (имя содержит «cookie» или "
            "расширение .txt). Отмена — /cancel"
        )
        return

    out_path = _yt_cookies_path()
    try:
        await message.bot.download(
            message.document.file_id, destination=str(out_path)
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("yt cookies download failed")
        await message.answer(
            f"❌ Не получилось сохранить файл: <code>{_html_escape(str(exc))}</code>"
        )
        return

    # Sanity check: must look like Netscape cookies.txt (first non-comment
    # line should be a tab-separated record with at least 6 fields).
    try:
        sample = out_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        await message.answer(
            f"❌ Не смог прочитать сохранённый файл: <code>{_html_escape(str(exc))}</code>"
        )
        return

    looks_valid = False
    for line in sample.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        if len(line.split("\t")) >= 6:
            looks_valid = True
            break

    if not looks_valid:
        out_path.unlink(missing_ok=True)
        await message.answer(
            "❌ Файл не похож на Netscape cookies.txt — внутри нет ни одной "
            "строки с табуляциями (6+ полей). Экспортируй через «Get "
            "cookies.txt LOCALLY», и пришли заново."
        )
        return

    # Count cookie entries for the confirmation message.
    cookie_count = sum(
        1
        for line in sample.splitlines()
        if line.strip() and not line.startswith("#") and len(line.split("\t")) >= 6
    )

    await state.clear()
    await message.answer(
        f"✅ <b>cookies.txt сохранён.</b>\n\n"
        f"Записей: <code>{cookie_count}</code>\n"
        f"Путь: <code>{out_path}</code>\n\n"
        "Следующая команда <code>скачай &lt;youtube-url&gt;</code> "
        "пойдёт уже с авторизованной сессией."
    )


@wizard_router.message(
    StateFilter(SetupStates.awaiting_yt_cookies_file), _NOT_A_COMMAND
)
async def capture_yt_cookies_wrong_type(
    message: Message, state: FSMContext
) -> None:
    """Prompt again if the user sent text/photo/voice instead of a document."""
    if not _is_owner(message.from_user.id if message.from_user else 0):
        return
    await message.answer(
        "Жду <b>документ</b> с cookies (файл .txt). /cancel — отмена."
    )
