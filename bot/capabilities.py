"""Maintain ``DATA_DIR/capabilities.md`` — a free-form log of bot features.

The agent only remembers the last ``HISTORY_LIMIT`` chat messages
(currently 40), so any feature announcement falls off after enough
chatter. To keep the bot self-aware about what it can do, we maintain
a persistent capabilities log on disk and inject it into every agent
prompt (see ``bot.agent._capabilities_log``).

This module owns two things:

1. ``CAPABILITIES`` — a structured list of known capabilities with
   stable ``key`` ids. The runtime code (``main.py`` on boot) calls
   ``ensure_recorded()`` so any newly-added capability lands in the
   log on first run after a deploy.
2. ``ensure_recorded()`` — idempotently appends new entries to
   ``capabilities.md``, never re-writing existing ones, so user-edited
   notes between bullets survive across restarts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .config import DATA_DIR

logger = logging.getLogger(__name__)

CAPABILITIES_FILE = DATA_DIR / "capabilities.md"


@dataclass(frozen=True)
class Capability:
    """One feature/capability of the bot.

    Fields:
        key: stable id used to detect "already recorded" entries.
            Never change for an existing capability — change the
            ``description`` instead and bump ``revision`` if needed.
        title: short human-readable name (shown in markdown ``###``).
        description: 1-3 sentence summary aimed at the bot itself.
            Phrase as "I can do X" so it reads naturally inside the
            system prompt.
        example: optional usage example (Russian, since the user is
            Russian-speaking).
    """

    key: str
    title: str
    description: str
    example: str = ""


CAPABILITIES: list[Capability] = [
    Capability(
        key="github-auto-clone",
        title="Авто-клонирование GitHub репозиториев",
        description=(
            "I can clone a git repository (GitHub / GitLab / Bitbucket) "
            "directly from a URL the user pastes in chat, then read "
            "files, run commands, and answer questions about its code. "
            "Powered by the clone_repo / list_projects / switch_project "
            "tools."
        ),
        example=(
            "Пользователь: «посмотри https://github.com/aiogram/aiogram "
            "и расскажи как у них роутеры устроены» → я вызываю "
            "clone_repo(url=...), потом list_dir, потом read_file, "
            "потом отвечаю."
        ),
    ),
    Capability(
        key="github-settings-screen",
        title="Экран ⚙️ Настройки → 🐙 GitHub",
        description=(
            "The user can browse all cloned repositories, see each "
            "repo's description, and delete repos (which removes them "
            "from disk and from my memory). There is also a toggle to "
            "disable auto-clone if they prefer explicit instructions."
        ),
    ),
    Capability(
        key="menu-button-commands",
        title="Меню-кнопка ≡ рядом с «отправить»",
        description=(
            "Telegram shows a popup with 8 quick-action commands when "
            "the user taps the round menu button next to the input "
            "(/start, /chat, /helpzavr, /pretty, /mailbox, /media, "
            "/github, /work, /settings). ⚙️ Настройки sits at the bottom."
        ),
    ),
    Capability(
        key="main-menu-layout",
        title="Главное меню после /start",
        description=(
            "Inline keyboard shown after /start has feature buttons on "
            "top — 🤖 Helpzavr, ✨ Красивый текст, 📬 Проверка почты, "
            "🎬 Генерация фото/видео, 🐙 GitHub проекты, 🖥 Терминал — "
            "followed by admin-only install batches and at the very "
            "bottom ⚙️ Настройки so it never gets buried."
        ),
    ),
    Capability(
        key="terminal-shortcut",
        title="🖥 Терминал — пакетный /work из меню",
        description=(
            "Main-menu and settings now have a 🖥 Терминал shortcut "
            "that drops the user straight into /work's batch mode: "
            "send a newline-separated list of shell / git / clone / "
            "exec / cd / project commands and I run them sequentially "
            "with a 1-hour per-command timeout. Exit with /cancel."
        ),
    ),
    Capability(
        key="settings-learn-helper",
        title="📚 Узнать о функционале — LLM-помощник в настройках",
        description=(
            "⚙️ Настройки → 📚 Узнать о функционале opens a tutorial "
            "chat. Any text message in this mode is routed through "
            "the user's main LLM (Мозг 1 or custom endpoint) with the "
            "full capabilities log injected as system prompt, so the "
            "LLM can explain every feature of the bot. Exit with "
            "/cancel."
        ),
    ),
    Capability(
        key="tts-auto-enable-on-pick",
        title="🔊 Голосовой ответчик включается автоматически при выборе",
        description=(
            "Picking a built-in voice (M1-M5 / F1-F5) or importing a "
            "Voice-Builder JSON now auto-enables the narrator AND "
            "auto-activates the imported clone as the current voice. "
            "Previously users had to flip a separate toggle and click "
            "«Сохранить как активный» — most assumed the picker was "
            "broken when in fact it just needed extra clicks."
        ),
    ),
    Capability(
        key="tts-ram-guard",
        title="🔊 RAM-страховка: TTS не валит бот на 512 MB хостинге",
        description=(
            "Supertonic synthesis peaks ~470 MB RSS, which OOM-kills "
            "the bot on Render Free/Starter (both 0.5 GB). The TTS "
            "pipeline now reads cgroup memory accounting before each "
            "synthesis and silently skips with a logged warning when "
            "<450 MB is free (configurable via TTS_MIN_AVAILABLE_MB). "
            "On capped hosts the user still sees text replies; the "
            "⚙️ Настройки → 🔊 Голосовой ответчик screen surfaces the "
            "exact reason so people know voice replies need Standard "
            "(2 GB) or bigger plan to work."
        ),
    ),
    Capability(
        key="tts-baked-model",
        title="🔊 Голосовая модель запечена в Docker-образ",
        description=(
            "The 386 MB Supertonic ONNX model is downloaded during "
            "the Docker build (where Render gives the builder lots of "
            "RAM/bandwidth) instead of at runtime on the cramped "
            "container. Previously the first voice reply would "
            "trigger a HuggingFace download that OOM-killed the bot "
            "mid-transfer and restart-looped. Now the runtime "
            "container starts with the model already on disk."
        ),
    ),
    Capability(
        key="unified-thinking-style",
        title="Единая Соображалка во всех потоках",
        description=(
            "Whatever the user picks in ⚙️ Настройки → 🧠 Соображалка "
            "applies everywhere: main chat with LLM, Helpzavr photo "
            "pipeline, Pretty Text → Standard, Mailbox, and media "
            "generation. Three styles: ⚪ White, ⚪ White + model "
            "(suffix the model name), ⬛ Black/native typing-indicator."
        ),
    ),
    Capability(
        key="access-modes",
        title="3 режима доступа + со-владельцы",
        description=(
            "I support private / public / full_public access modes "
            "plus a co-owner list managed via ⚙️ Настройки → ⛔ Доступ. "
            "In public mode strangers can USE features but cannot see "
            "API keys / models / stats; in full_public they can do "
            "everything; in private only owner + co-owners get in."
        ),
    ),
    Capability(
        key="algorithm-slots",
        title="🧩 Алгоритм — 10 слотов с пошаговыми планами",
        description=(
            "⚙️ Настройки → 🧩 Алгоритм — 10 слотов на чат. В каждый "
            "слот можно сохранить пошаговый план: либо вручную, либо "
            "по тексту/новости — бот сам разложит на шаги через LLM. "
            "Приоритет планировщика: «Другое» (custom OpenAI-совместимый) → "
            "Brain 1 → Brain 2. Кнопка «▶ Запустить» исполняет шаги "
            "строго последовательно через агентский режим — каждый "
            "шаг видит ВСЕ тулзы бота: локальные (read_file, "
            "exec_bash, write_file) и внешние (tavily_search, "
            "brave_search, exa_search, firecrawl_scrape, "
            "apify_run_actor, github_search_code, github_get_file). "
            "Падение одного шага останавливает алгоритм с понятной "
            "причиной."
        ),
    ),
    Capability(
        key="algorithm-periodic",
        title="🧩 Алгоритм — периодика (мин / час / день / неделя / месяц)",
        description=(
            "У каждого слота есть «⏱ Периодичность: N мин» — дробные "
            "ОК (0.5 = 30 сек). Подсказки: 1440 = день, 10080 = "
            "неделя, 43200 = месяц. Фоновый scheduler-task просыпается "
            "каждые 30 сек и запускает все слоты у которых пришло "
            "время. Идеально для новостного мониторинга: «каждые 60 "
            "мин сходи tavily_search по теме X и пришли резюме»."
        ),
    ),
    Capability(
        key="external-research-tools",
        title="🛠 Внешние API-тулзы агента (Tavily / Brave / Exa / Firecrawl / Apify / GitHub)",
        description=(
            "Агент умеет ходить в интернет: tavily_search (AI-tuned "
            "поиск с готовым answer'ом), brave_search (Google-like), "
            "exa_search (семантический поиск похожих страниц), "
            "firecrawl_scrape (URL → markdown), apify_run_actor "
            "(скрейперы TikTok / Reddit / Twitter и др.), "
            "github_search_code + github_get_file (поиск и чтение "
            "кода с GitHub). Ключи задаются в /setup → 🛠 Внешние "
            "API; без ключа тулза возвращает дружелюбное «открой "
            "/setup и вставь ключ» вместо краша."
        ),
    ),
    Capability(
        key="elevenlabs-tts",
        title="☁️ ElevenLabs — облачный TTS-провайдер",
        description=(
            "⚙️ Настройки → 🔊 Голосовой ответчик → ☁️ ElevenLabs. "
            "Альтернатива локальному Supertonic — синтез делается на "
            "стороне ElevenLabs, не ест RAM. Поля: 🔑 API ключ + 🎤 "
            "Voice ID (копируется из <a href='https://elevenlabs.io/"
            "app/voice-library'>Voice Library</a>). URL хардкод. "
            "Кнопка «▶ Использовать ElevenLabs» делает его активным "
            "TTS-провайдером — встроенные голоса (M1..F5) и клон "
            "помечаются «▶ Использовать встроенный голос» / «▶ "
            "Использовать клон» в своих подменю. Free-тариф "
            "ElevenLabs: 10k символов/мес. Если ключ/Voice ID не "
            "заданы или API упал — мягкий фоллбэк на локальный "
            "Supertonic с warning'ом в логе."
        ),
    ),
    Capability(
        key="brain2-quick-pick-models",
        title="🧠 Мозг 2 — быстрый выбор моделей (Whisper / Llama-Scout / Llama-70b)",
        description=(
            "⚙️ Настройки → 🧠 Перенастроить мозг → выбери слот 2 → "
            "🤖 Модель. Открывается клавиатура с быстрыми пресетами: "
            "🎙 whisper-large-v3 (⭐ дефолт), 🎙 whisper-large-v3-turbo, "
            "🦙 llama-4-scout-17b-16e-instruct, 🦙 llama-3.3-70b-"
            "versatile, и ✍️ Свой вариант (ручной ввод для редких "
            "моделей). Все пресеты — Groq endpoint; для своего ввода "
            "помни задать base_url через 🌐 URL."
        ),
    ),
    Capability(
        key="batch-api-import",
        title="🔌 API — массовый импорт ключей одной строкой",
        description=(
            "⚙️ Настройки → 🔌 API (массовый импорт). Открывает поле "
            "ввода — пришли одной строкой все ключи которые хочешь "
            "загрузить. Формат: <code>target:key:value</code> внутри "
            "поля, <code>;</code> между полями, <code>$$$</code> "
            "между независимыми блоками. Поддерживаются: brain1/"
            "brain2 (api/model/url/provider), elevenlabs (api/voice), "
            "tavily, brave, exa, firecrawl, apify, github_pat. "
            "Сообщение с ключами автоматически удаляется из истории "
            "чата сразу после применения. Бот возвращает отчёт: "
            "«применено» + «ошибки» построчно."
        ),
    ),
]


def _existing_keys(text: str) -> set[str]:
    """Parse out which capability keys are already in the file."""
    keys: set[str] = set()
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("<!-- key:") and line.endswith("-->"):
            key = line[len("<!-- key:"):-len("-->")].strip()
            if key:
                keys.add(key)
    return keys


def _render_entry(cap: Capability) -> str:
    """Render one capability as a markdown block with a key marker."""
    parts = [
        f"<!-- key: {cap.key} -->",
        f"### {cap.title}",
        "",
        cap.description,
    ]
    if cap.example:
        parts.extend(["", f"*Пример:* {cap.example}"])
    return "\n".join(parts) + "\n"


def ensure_recorded() -> None:
    """Append any not-yet-recorded capabilities to ``capabilities.md``.

    Idempotent: existing entries (identified by their ``<!-- key:
    ... -->`` marker) are left alone, so users / operators can hand-
    edit the surrounding markdown without us clobbering their notes
    on the next boot.

    Best-effort: I/O errors are logged and swallowed — a corrupted
    capabilities log shouldn't stop the bot from booting.
    """
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:  # noqa: BLE001
        logger.exception("failed to mkdir DATA_DIR for capabilities log")
        return

    try:
        existing = (
            CAPABILITIES_FILE.read_text(errors="replace")
            if CAPABILITIES_FILE.exists()
            else ""
        )
    except Exception:  # noqa: BLE001
        logger.exception("failed to read capabilities log")
        existing = ""

    have = _existing_keys(existing)
    new_blocks = [
        _render_entry(cap) for cap in CAPABILITIES if cap.key not in have
    ]
    if not new_blocks:
        return

    header = (
        "# Bot capabilities\n\n"
        "_This file is auto-maintained by `bot/capabilities.py`._\n"
        "_Each `<!-- key: ... -->` block is appended once and never_\n"
        "_rewritten, so manual edits between blocks survive restarts._\n\n"
    )
    if not existing.strip():
        body = header + "\n".join(new_blocks)
    else:
        # Ensure separation from whatever the user has at the end.
        sep = "\n" if existing.endswith("\n") else "\n\n"
        body = existing + sep + "\n".join(new_blocks)
    try:
        CAPABILITIES_FILE.write_text(body)
        logger.info(
            "wrote %d new capability entries to %s",
            len(new_blocks),
            CAPABILITIES_FILE,
        )
    except Exception:  # noqa: BLE001
        logger.exception("failed to write capabilities log")
