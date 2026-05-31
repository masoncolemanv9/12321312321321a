"""Bot personas for the Lilush multi-bot farm.

Each Render service runs the SAME codebase but activates ONE persona via the
``BOT_PERSONA`` env var. The persona controls:

- Display name in /start / /help messages
- Short role description shown to the user
- The bot's slot in the farm (boss / lead / worker)
- Which department it belongs to (research / debate / coder / devops)

Coordination logic (boss → lead → workers) is handled separately by the
farm module and lives in ``bot/farm/`` (added in later phases). This
module is the source of truth for *who this process is*.

Default persona is ``boss`` so a single-bot deploy still works the same
as before — backwards compatibility with the existing @openaiopus_bot.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Persona:
    """One bot's identity within the farm."""

    key: str  # env var value, e.g. "boss"
    display_name: str  # shown to user, e.g. "Boss"
    title: str  # one-line role description
    department: str  # 'leadership' | 'research' | 'debate' | 'coder' | 'devops'
    rank: str  # 'boss' | 'lead' | 'worker'
    description: str  # multi-line role description for /start


# --- Roster ---------------------------------------------------------------

_PERSONAS: dict[str, Persona] = {
    # === Leadership ===
    "boss": Persona(
        key="boss",
        display_name="Lilush Boss",
        title="Orchestrator — главный интерфейс пользователя",
        department="leadership",
        rank="boss",
        description=(
            "Я главный в ферме. Принимаю команды от тебя, маршрутизирую "
            "задачи на 4 отдела (research / debate / coder / devops), "
            "отчитываюсь о результатах. Команды: /research, /debate, "
            "/implement, /review, /status, /budget, /farm."
        ),
    ),
    # === Leads (4) ===
    "research_lead": Persona(
        key="research_lead",
        display_name="Research Lead",
        title="Руководитель отдела исследований",
        department="research",
        rank="lead",
        description=(
            "Веду research-отдел: GitHub Scout, Reddit Scout, HN Scout, "
            "Apify Runner. Получаю тему от Boss → раздаю sub-задачи → "
            "агрегирую → выдаю dossier.json."
        ),
    ),
    "debate_lead": Persona(
        key="debate_lead",
        display_name="Debate Lead",
        title="Руководитель отдела дебатов",
        department="debate",
        rank="lead",
        description=(
            "Веду 4 debater-бота: D1 Skeptic, D2 Optimist, Judge, TZ "
            "Writer. Оркестрирую 8 раундов дебата → выпускаю валидный "
            "TZ.md для Coder Lead."
        ),
    ),
    "coder_lead": Persona(
        key="coder_lead",
        display_name="Coder Lead",
        title="Руководитель отдела разработки",
        department="coder",
        rank="lead",
        description=(
            "Веду Devin Spawner, PR Reviewer, Tester. Получаю TZ → "
            "запускаю Devin-сессии через api.devin.ai → дожимаю до "
            "merge-able PR."
        ),
    ),
    "devops_lead": Persona(
        key="devops_lead",
        display_name="DevOps Lead",
        title="Руководитель DevOps",
        department="devops",
        rank="lead",
        description=(
            "Веду Watchdog, Render Admin, GitHub Admin, Archivist. "
            "Мониторю кредиты API, ротирую секреты, управляю Render "
            "сервисами и GitHub репами фермы."
        ),
    ),
    # === Research workers (4) ===
    "github_scout": Persona(
        key="github_scout",
        display_name="GitHub Scout",
        title="Поиск кода и issues на GitHub",
        department="research",
        rank="worker",
        description=(
            "Ищу по GitHub Code Search и Issues. Использую "
            "GITHUB_RESEARCH_PAT (read-only). Возвращаю Research Lead'у "
            "ранжированные source'ы."
        ),
    ),
    "reddit_scout": Persona(
        key="reddit_scout",
        display_name="Reddit Scout",
        title="Поиск threads и комментариев на Reddit",
        department="research",
        rank="worker",
        description=(
            "Качаю Reddit JSON API по сабреддитам и search. Фильтрую "
            "через Kimi. Возвращаю топ комментариев + URL'ы."
        ),
    ),
    "hn_scout": Persona(
        key="hn_scout",
        display_name="HN Scout",
        title="Поиск историй на Hacker News",
        department="research",
        rank="worker",
        description=(
            "HN Algolia API. Беру топовые stories по теме + читаю "
            "комментарии. Дешёвый и точный источник expert opinion."
        ),
    ),
    "apify_runner": Persona(
        key="apify_runner",
        display_name="Apify Runner",
        title="Запуск Apify actors (TikTok, Google, и т.д.)",
        department="research",
        rank="worker",
        description=(
            "Платный fallback: если бесплатные API не нашли — запускаю "
            "Apify actor. С approval-flow DevOps Lead'а (cost cap $0.50)."
        ),
    ),
    # === Debate workers (4) ===
    "d1_skeptic": Persona(
        key="d1_skeptic",
        display_name="D1 Skeptic",
        title="Скептик в дебатах",
        department="debate",
        rank="worker",
        description=(
            "Задаю сложные вопросы, ищу дыры в предлагаемом решении. "
            "Не принимаю «и так сойдёт» — заставляю Debate Lead'а думать."
        ),
    ),
    "d2_optimist": Persona(
        key="d2_optimist",
        display_name="D2 Optimist",
        title="Оптимист в дебатах",
        department="debate",
        rank="worker",
        description=(
            "Защищаю решение, ищу компромиссы. После раунда даю "
            "контр-аргумент на критику D1."
        ),
    ),
    "judge": Persona(
        key="judge",
        display_name="Judge",
        title="Модератор раундов",
        department="debate",
        rank="worker",
        description=(
            "Слежу за конвергенцией дебата. Чекаю что D1 и D2 идут "
            "к выводу. После 8 раундов выношу финальный вердикт."
        ),
    ),
    "tz_writer": Persona(
        key="tz_writer",
        display_name="TZ Writer",
        title="Пишет финальное ТЗ после дебата",
        department="debate",
        rank="worker",
        description=(
            "После финального вердикта Judge собираю TZ.md: цели, "
            "ограничения, acceptance criteria, edge cases. Передаю "
            "Coder Lead'у."
        ),
    ),
    # === Coder workers (3) ===
    "devin_spawner": Persona(
        key="devin_spawner",
        display_name="Devin Spawner",
        title="Создаёт Devin-сессии через API",
        department="coder",
        rank="worker",
        description=(
            "POST api.devin.ai/v1/sessions с TZ. Слежу за сессией "
            "(GET sessions/{id}). Возвращаю PR URL когда готово."
        ),
    ),
    "pr_reviewer": Persona(
        key="pr_reviewer",
        display_name="PR Reviewer",
        title="Ревьюер PR'ов",
        department="coder",
        rank="worker",
        description=(
            "Читаю diff через GitHub API. Ставлю SCORE/10. Если < 9 — "
            "пишу конкретные комменты что переделать. Hybrid LLM: "
            "Kimi для простого, Opus для спорного."
        ),
    ),
    "tester": Persona(
        key="tester",
        display_name="Tester",
        title="Запускает тесты на PR",
        department="coder",
        rank="worker",
        description=(
            "После PR — клонирую ветку, прогоняю pytest. Парсю результаты. "
            "Если что-то падает — пишу Coder Lead'у."
        ),
    ),
    # === DevOps workers (4) ===
    "watchdog": Persona(
        key="watchdog",
        display_name="Watchdog",
        title="Мониторинг кредитов API всех ботов",
        department="devops",
        rank="worker",
        description=(
            "Каждые 5 минут пингую Anthropic/Kimi/Devin API на остаток "
            "кредитов. При < 10% — алерт в Leadership Room."
        ),
    ),
    "render_admin": Persona(
        key="render_admin",
        display_name="Render Admin",
        title="Управление Render-сервисами через API",
        department="devops",
        rank="worker",
        description=(
            "Использую Render Management API: redeploy, env vars, logs. "
            "По команде DevOps Lead'а — перезапускаю упавший сервис."
        ),
    ),
    "github_admin": Persona(
        key="github_admin",
        display_name="GitHub Admin",
        title="Управление GitHub репозиториями (WRITE)",
        department="devops",
        rank="worker",
        description=(
            "GITHUB_ADMIN_PAT с write-правами. Создаю репо, secrets, "
            "branch protection. Только по команде DevOps Lead'а."
        ),
    ),
    "archivist": Persona(
        key="archivist",
        display_name="Archivist",
        title="Архивирует dossier'ы и TZ в Knowledge Notes",
        department="devops",
        rank="worker",
        description=(
            "После каждого research/debate — сохраняю результаты в Devin "
            "Knowledge Notes для долговременной памяти фермы."
        ),
    ),
}


# --- API ------------------------------------------------------------------


def get_persona(key: str | None = None) -> Persona:
    """Resolve the active persona for this process.

    Resolution order when ``key=None``:
      1. ``storage.get_persona_override()`` — set via ``/role`` from TG.
      2. ``BOT_PERSONA`` env var (defaults to "boss").

    The override layer lets the owner reassign a deployed bot's role
    without re-deploying or editing Render env vars. Falls back to
    ``boss`` for an unknown value so a typo in env config doesn't crash
    the bot — just silently uses the boss persona.
    """
    if key is None:
        # Lazy import — storage imports config which imports persona for
        # other paths; deferring this keeps the module graph acyclic.
        try:
            from .storage import storage as _storage

            override = _storage.get_persona_override()
        except Exception:  # noqa: BLE001 — storage not ready (early boot)
            override = None
        key = override or os.environ.get("BOT_PERSONA", "boss").strip().lower()
    persona = _PERSONAS.get(key)
    if persona is None:
        # Last-resort fallback to keep the bot bootable.
        return _PERSONAS["boss"]
    return persona


def list_personas() -> list[Persona]:
    """All 20 personas in roster order."""
    return list(_PERSONAS.values())


def known_keys() -> list[str]:
    return list(_PERSONAS.keys())
