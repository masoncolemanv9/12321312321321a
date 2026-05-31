import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from aiogram import BaseMiddleware, F, Router
from aiogram.filters import Command, CommandObject, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import Message, TelegramObject

from .agent import NoApiKeyError, oneshot_summary, run_agent
from .config import ALLOWED_USER_IDS, DEFAULT_MODEL, PROJECTS_DIR, WORK_EXEC_TIMEOUT
from .inbox import log_inbox
from .jobs import get_default_queue
from .storage import KNOWN_PROVIDERS, storage
from .tools import ToolError, clone_repo, exec_bash, project_root_for

logger = logging.getLogger(__name__)
router = Router()


class _InboxLoggerMiddleware(BaseMiddleware):
    """Append every incoming Message to ``data/inbox.log`` before dispatch.

    Provides a chat-history backup AND the inbox a real Devin session reads
    when ``brain=devin``.
    """

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        if isinstance(event, Message) and event.text:
            kind = "cmd" if event.text.startswith("/") else "text"
            log_inbox(
                user_id=event.from_user.id if event.from_user else None,
                chat_id=event.chat.id,
                text=event.text,
                kind=kind,
            )
        return await handler(event, data)


router.message.middleware(_InboxLoggerMiddleware())


HELP_TEXT = (
    "<b>Lilush</b> — Telegram-бот + видео-конвейер + farm-роль для распределённой работы\n"
    "Brain: <code>{brain}</code> | model: <code>{model}</code> {state}\n\n"
    "<b>📹 Видео-конвейер</b>\n"
    "Кидаешь ссылку → скачиваю → транскрибирую → нарезаю на 3 шортса 1080×1920 → пишу SEO → собираю release-пакет.\n\n"
    "  /dl &lt;url&gt; — поставить видео в конвейер\n"
    "  /jobs — последние 20 джобов и их статус\n\n"
    "Этапы (5 параллельных воркеров):\n"
    "  1️⃣ <b>download</b> — yt-dlp, ≤1080p, ≤5 GB\n"
    "  2️⃣ <b>analyze</b> — faster-whisper транскрипт + pyscenedetect сцены + LLM/heuristic ранкер клипов\n"
    "  3️⃣ <b>edit</b> — ffmpeg vertical crop 1080×1920 + опциональный logo-overlay\n"
    "  4️⃣ <b>seo</b> — pytrends Google Trends + LLM/template title/description/tags\n"
    "  5️⃣ <b>publish</b> — DRY-RUN: clip.mp4 + youtube.json + tiktok.json + instagram.json в data/releases/&lt;id&gt;/\n\n"
    "После /dl можешь сразу кидать следующую ссылку — все 5 воркеров крутятся параллельно.\n\n"
    "<b>💬 LLM-чат</b>\n"
    "Любое сообщение без <code>/</code> уходит в текущий brain. <code>auto</code> = бот сам отвечает через OpenRouter (Kimi / Opus / GPT — что выбрал в /setmodel); <code>devin</code> = пишет в inbox.log и ждёт меня.\n\n"
    "<b>Старт / настройка</b>\n"
    "/start — главное меню + кнопки «Скачать видео», «Сменить роль», «Мозг», «Токены»\n"
    "/setup — заново выбрать мозг и ввести ключи через кнопки\n"
    "/role — сменить роль этого бота (20 кнопок персон, override BOT_PERSONA из env)\n\n"
    "<b>🛠 Внешние API (research/scraping)</b>\n"
    "В /setup → 🛠 Внешние API — Apify, Firecrawl, Tavily, Brave Search, Exa, GitHub PAT.\n"
    "Researcher-боты используют их для поиска данных вне TG.\n\n"
    "<b>📊 Расход и здоровье</b>\n"
    "/tokens — статистика token spend (сегодня / неделя / всего), оценка стоимости\n"
    "Heartbeat OpenRouter — в /setup кнопка <code>💓 ON / 💤 OFF</code>: бот периодически пингует OpenRouter (1 токен раз в 10 мин) чтобы держать сессию тёплой\n\n"
    "<b>Проекты (для /exec /git)</b>\n"
    "/projects — список загруженных проектов\n"
    "/clone &lt;git-url&gt; [имя] — клонировать репо\n"
    "/project &lt;имя&gt; — переключиться на проект\n"
    "/cd &lt;путь&gt;, /pwd — навигация по субпапкам\n\n"
    "<b>Выполнение</b>\n"
    "/exec &lt;команда&gt; — bash в текущем проекте\n"
    "/git &lt;args&gt; — то же что /exec git ...\n"
    "/work — пакетный терминал: пришлёшь список команд одним сообщением, "
    "я выполню их по порядку без жёсткого таймаута (до 1 часа на команду)\n"
    "/cancel — выйти из режима /work\n\n"
    "<b>Мозги и ключи</b>\n"
    "/brain — кто сейчас в седле (auto / devin)\n"
    "/setbrain auto|devin — переключить\n"
    "/keys, /setkey &lt;provider&gt; &lt;key&gt;, /delkey &lt;provider&gt; — ключи (openrouter, anthropic, openai)\n"
    "/models, /setmodel &lt;model&gt; — модель\n\n"
    "<b>Управление</b>\n"
    "/disable, /enable — выключить/включить бот\n"
    "/reset — сбросить контекст разговора\n"
    "/help — это сообщение"
)


# Краткая «визитка» для пустых/неизвестных запросов.
WHOAMI_TEXT = (
    "<b>Я Lilush</b> — превращаю длинные видео в шортсы для YouTube/TikTok/Instagram.\n\n"
    "Что умею прямо сейчас:\n"
    "• /dl &lt;ссылка-на-видео&gt; — запустить полный конвейер (5 этапов параллельно)\n"
    "• /jobs — посмотреть статус всех твоих джобов\n"
    "• общаться как ChatGPT — пиши любой текст без <code>/</code>\n"
    "• /clone, /exec, /git — работать с git-репо прямо из чата\n\n"
    "Полная справка: /help"
)

# Curated catalogue of models worth pinning. /setmodel accepts any string
# OpenRouter understands; this list is purely for /models output.
MODEL_CATALOGUE: list[tuple[str, str]] = [
    # Free tier (no credit required, rate-limited).
    ("nvidia/nemotron-3-super-120b-a12b:free", "Nemotron 120b — default free"),
    ("openai/gpt-oss-120b:free", "GPT-OSS 120b — free"),
    ("qwen/qwen3-coder:free", "Qwen3 Coder — free"),
    ("minimax/minimax-m2.5:free", "MiniMax 2.5 — free"),
    # Anthropic Claude via OpenRouter (needs OpenRouter credit, ~5% markup).
    ("anthropic/claude-opus-4.7", "Claude Opus 4.7 — newest (Apr 16 2026), strongest"),
    ("anthropic/claude-opus-4.6", "Claude Opus 4.6 — prev-gen Opus (Feb 2026)"),
    ("anthropic/claude-sonnet-4.5", "Claude Sonnet 4.5 — balanced"),
    ("anthropic/claude-haiku-4.5", "Claude Haiku 4.5 — fast, cheap"),
    # OpenAI via OpenRouter.
    ("openai/gpt-5", "GPT-5 — strongest OpenAI"),
    ("openai/gpt-4o", "GPT-4o — fast, capable"),
]


def _is_authorized(message: Message) -> bool:
    """A message is authorised when the sender is allowed to USE the bot.

    Authorisation depends on the configured access mode:
    * ``private`` — only the owner + explicit co-owners.
    * ``public`` / ``full_public`` — anyone is authorised to send
      messages. (Settings changes are gated separately via
      :func:`bot.access.can_admin`.)

    Backwards compatibility: if no owner has been claimed yet *and* the
    env var ``ALLOWED_USER_IDS`` is set, fall back to that list so
    existing self-hosters keep working until they re-onboard via /start.
    """
    from .access import can_use

    user = message.from_user
    if user is None:
        return False
    owner_id = storage.get_owner_id()
    if owner_id is None and ALLOWED_USER_IDS:
        return user.id in ALLOWED_USER_IDS
    return can_use(user.id)


async def _deny(message: Message) -> None:
    user = message.from_user
    uid = user.id if user else "unknown"
    owner_id = storage.get_owner_id()
    if owner_id is None:
        await message.answer(
            "Этот контейнер ещё не привязан к владельцу. Жми /start чтобы стать им."
        )
        return
    await message.answer(
        f"Этот бот принадлежит другому владельцу.\nТвой telegram id: <code>{uid}</code>"
    )


def _html_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# Telegram hard limit is 4096 characters per message. We reserve room for the
# `<pre></pre>` wrapper (13 chars) and a small margin for safety.
_TG_LIMIT = 4096
_PRE_OVERHEAD = len("<pre></pre>")
_CHUNK_LIMIT = _TG_LIMIT - _PRE_OVERHEAD - 16


def _safe_cut(escaped: str, limit: int) -> int:
    """Pick a cut position <= ``limit`` that does not split an HTML entity."""
    cut = escaped.rfind("\n", 0, limit)
    if cut < limit // 2:
        cut = escaped.rfind(" ", 0, limit)
    if cut < limit // 2:
        cut = limit
    # HTML entities introduced by `_html_escape` are at most 5 chars (`&amp;`)
    amp = escaped.rfind("&", max(0, cut - 6), cut)
    if amp != -1 and ";" not in escaped[amp:cut]:
        cut = amp
    return max(cut, 1)


async def _send_long(message: Message, text: str, *, code: bool = True) -> None:
    """Send a (possibly long) reply, splitting on Telegram's 4096-char limit.

    When ``code`` is True (default) each chunk is wrapped in ``<pre>...</pre>``
    so shell output / file contents render in monospace. Set ``code=False``
    for natural-language replies (e.g. LLM agent answers) so they show up as
    plain rich text instead of a copy-button code block.
    """
    if not text:
        text = "(пусто)"
    escaped = _html_escape(text)
    chunks: list[str] = []
    limit = _CHUNK_LIMIT if code else _TG_LIMIT - 16
    while len(escaped) > limit:
        cut = _safe_cut(escaped, limit)
        chunks.append(escaped[:cut])
        escaped = escaped[cut:]
    chunks.append(escaped)
    for chunk in chunks:
        if not chunk.strip():
            continue
        await message.answer(f"<pre>{chunk}</pre>" if code else chunk)


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    # Note: /start is handled by wizard.py — it triggers the owner-claim or
    # the brain re-picker. /help just dumps the command reference.
    if not _is_authorized(message):
        await _deny(message)
        return
    state_label = "" if storage.is_enabled() else "<i>(выключен — /enable чтобы поднять)</i>"
    await message.answer(
        HELP_TEXT.format(
            brain=storage.get_brain(),
            model=storage.get_model() or DEFAULT_MODEL,
            state=state_label,
        )
    )


def _role_chosen() -> bool:
    """True when project-side gating should let the user through.

    The historical contract: every project-touching command (``/exec``,
    ``/clone``, ``/work``, etc.) silently refuses until the user runs
    ``/role`` and picks a persona. That gate is now opt-in via Settings
    → 🎭 Роли → toggle. When the gate is disabled (default) we always
    return True so commands work without picking a role first.
    """
    if not storage.get_role_gate_enabled():
        return True
    return storage.get_persona_override() is not None


_ROLE_REQUIRED = (
    "Требование роли включено. Открой /role или Settings → «🎭 Роли» "
    "(там же можно его выключить)."
)


@router.message(Command("projects"))
async def cmd_projects(message: Message) -> None:
    if not _is_authorized(message):
        await _deny(message)
        return
    if not _role_chosen():
        await message.answer(_ROLE_REQUIRED)
        return
    projects = storage.list_projects()
    if not projects:
        await message.answer(
            "Проектов пока нет. Склонируй репо:\n<code>/clone https://github.com/owner/repo</code>"
        )
        return
    cwd = storage.get_cwd(message.from_user.id)
    cwd_name = cwd.relative_to(PROJECTS_DIR).parts[0] if cwd else None
    lines = []
    for name in projects:
        marker = "→ " if name == cwd_name else "  "
        lines.append(f"{marker}{name}")
    await message.answer("<b>Проекты</b>\n<code>" + "\n".join(lines) + "</code>")


@router.message(Command("clone"))
async def cmd_clone(message: Message, command: CommandObject) -> None:
    if not _is_authorized(message):
        await _deny(message)
        return
    if not _role_chosen():
        await message.answer(_ROLE_REQUIRED)
        return
    args = (command.args or "").split()
    if not args:
        await message.answer("Использование: <code>/clone &lt;git-url&gt; [имя]</code>")
        return
    url = args[0]
    name = args[1] if len(args) > 1 else None
    await message.answer(f"Клонирую <code>{_html_escape(url)}</code>...")
    try:
        dest = await clone_repo(url, name)
    except ToolError as exc:
        await message.answer(f"Ошибка: {_html_escape(str(exc))}")
        return
    storage.set_cwd(message.from_user.id, dest)
    await message.answer(f"Готово. Проект: <b>{dest.name}</b>\nТекущий cwd: <code>{dest}</code>")


@router.message(Command("project"))
async def cmd_project(message: Message, command: CommandObject) -> None:
    if not _is_authorized(message):
        await _deny(message)
        return
    if not _role_chosen():
        await message.answer(_ROLE_REQUIRED)
        return
    name = (command.args or "").strip()
    if not name:
        await message.answer("Использование: <code>/project &lt;имя&gt;</code>")
        return
    target = PROJECTS_DIR / name
    if not target.exists():
        await message.answer(
            f"Проект <code>{_html_escape(name)}</code> не найден. /projects — список."
        )
        return
    storage.set_cwd(message.from_user.id, target)
    await message.answer(f"Переключился на <b>{name}</b>\ncwd: <code>{target}</code>")


@router.message(Command("cd"))
async def cmd_cd(message: Message, command: CommandObject) -> None:
    if not _is_authorized(message):
        await _deny(message)
        return
    if not _role_chosen():
        await message.answer(_ROLE_REQUIRED)
        return
    cwd = storage.get_cwd(message.from_user.id)
    if cwd is None:
        await message.answer("Сначала выбери проект: /projects или /clone")
        return
    rel = (command.args or "").strip() or "."
    target = (cwd / rel).resolve()
    project_root = project_root_for(cwd).resolve()
    if project_root not in target.parents and target != project_root:
        await message.answer("Нельзя выйти за пределы проекта")
        return
    if not target.exists() or not target.is_dir():
        await message.answer(f"Не найдена директория: <code>{_html_escape(str(target))}</code>")
        return
    storage.set_cwd(message.from_user.id, target)
    await message.answer(f"cwd: <code>{target}</code>")


@router.message(Command("pwd"))
async def cmd_pwd(message: Message) -> None:
    if not _is_authorized(message):
        await _deny(message)
        return
    if not _role_chosen():
        await message.answer(_ROLE_REQUIRED)
        return
    cwd = storage.get_cwd(message.from_user.id)
    if cwd is None:
        await message.answer("Проект не выбран")
        return
    await message.answer(f"<code>{cwd}</code>")


@router.message(Command("exec"))
async def cmd_exec(message: Message, command: CommandObject) -> None:
    if not _is_authorized(message):
        await _deny(message)
        return
    if not _role_chosen():
        await message.answer(_ROLE_REQUIRED)
        return
    cmd = (command.args or "").strip()
    if not cmd:
        await message.answer("Использование: <code>/exec &lt;команда&gt;</code>")
        return
    cwd = storage.get_cwd(message.from_user.id)
    if cwd is None:
        await message.answer("Сначала выбери проект: /projects или /clone")
        return
    try:
        result = await exec_bash(cwd, cmd)
    except ToolError as exc:
        await message.answer(f"Ошибка: {_html_escape(str(exc))}")
        return
    await _send_long(message, result)


@router.message(Command("git"))
async def cmd_git(message: Message, command: CommandObject) -> None:
    if not _is_authorized(message):
        await _deny(message)
        return
    if not _role_chosen():
        await message.answer(_ROLE_REQUIRED)
        return
    args = (command.args or "").strip()
    if not args:
        await message.answer("Использование: <code>/git &lt;args&gt;</code>")
        return
    cwd = storage.get_cwd(message.from_user.id)
    if cwd is None:
        await message.answer("Сначала выбери проект: /projects или /clone")
        return
    try:
        result = await exec_bash(cwd, f"git {args}")
    except ToolError as exc:
        await message.answer(f"Ошибка: {_html_escape(str(exc))}")
        return
    await _send_long(message, result)


@router.message(Command("reset"))
async def cmd_reset(message: Message) -> None:
    if not _is_authorized(message):
        await _deny(message)
        return
    storage.clear_history(message.from_user.id)
    await message.answer("Контекст разговора очищен")


# ---- /work — sequential batch terminal ----------------------------------


class WorkStates(StatesGroup):
    awaiting_commands = State()


@router.message(Command("cancel"))
async def cmd_cancel(message: Message, state: FSMContext) -> None:
    """Exit any pending FSM state (primarily used to leave /work)."""
    current = await state.get_state()
    if current is None:
        await message.answer("Нечего отменять.")
        return
    await state.clear()
    await message.answer("Отменено.")


@router.message(Command("work"))
async def cmd_work(message: Message, state: FSMContext) -> None:
    """Switch the user into batch-terminal mode.

    After /work the next text message is treated as a newline-separated
    list of commands. Each line is executed sequentially via
    :func:`_work_run_line`; per-command timeout is bumped to
    ``WORK_EXEC_TIMEOUT`` (default 1 hour) so heavy installs don't get
    killed mid-way through the batch.
    """
    if not _is_authorized(message):
        await _deny(message)
        return
    if not _role_chosen():
        await message.answer(_ROLE_REQUIRED)
        return
    await state.set_state(WorkStates.awaiting_commands)
    await message.answer(
        "Жду команды списком (по одной на строку). Пришли одним сообщением — "
        "выполню по порядку, ждя завершения каждой. Таймаут на команду снят "
        "(до 1 часа).\n\n"
        "Поддерживаю: <code>/exec</code>, <code>/clone</code>, "
        "<code>/project</code>, <code>/cd</code>, <code>/git</code>. "
        "Строка без <code>/</code> — bash в текущем проекте. Пустые строки и "
        "<code>#</code>-комментарии пропускаю.\n\n"
        "Выйти из режима: /cancel"
    )


README_CANDIDATES: tuple[str, ...] = (
    "README.md",
    "Readme.md",
    "readme.md",
    "README.rst",
    "README.txt",
    "README",
)

_README_MAX_CHARS = 12000
_README_REPLY_TIMEOUT_S = 60.0


def _read_repo_readme(dest: Path) -> tuple[str, str] | None:
    """Return ``(filename, contents)`` for the first README we find in ``dest``.

    Trims to ``_README_MAX_CHARS`` so we don't blow the prompt budget on a
    25k-line `README.md`. Returns ``None`` when no README is present —
    caller silently skips the LLM summary in that case.
    """
    for name in README_CANDIDATES:
        path = dest / name
        if path.exists() and path.is_file():
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if len(text) > _README_MAX_CHARS:
                text = text[:_README_MAX_CHARS] + "\n…[обрезано]"
            return name, text
    return None


def _build_readme_prompt(
    url: str, dest_name: str, readme_name: str, readme_text: str
) -> str:
    """Prompt the active brain to extract Quick-Start from a README.

    Phrased exactly the way the user asked: "вытащи Quick Start, дай 3-5
    команд подряд с примерами, покажи что заменить под мой сервис".
    """
    return (
        f"Я только что склонировал репозиторий <{url}> в папку "
        f"{dest_name}/.\n"
        f"Ниже — содержимое {readme_name}.\n\n"
        "Задача:\n"
        "1) Найди раздел Quick Start / Installation / Usage.\n"
        "2) Дай мне 3-5 команд подряд с примерами — те, которые надо запустить, "
        "чтобы реально получить рабочий результат на моём сервере.\n"
        "3) Для каждой команды покажи, что в ней нужно заменить под мой "
        "сервис (плейсхолдеры типа <YOUR_TOKEN>, <repo>, <port> и т.п.).\n"
        "4) После команд — короткий пример: «если выполнить как есть → "
        "получишь X; если заменить Y на Z → получишь желаемый результат».\n"
        "5) Не выдумывай команды, которых нет в README. Если чего-то не "
        "хватает — напиши «в README этого нет, нужно искать в docs/».\n\n"
        "Отвечай по-русски, кратко, без воды.\n\n"
        f"--- {readme_name} ---\n{readme_text}"
    )


async def _maybe_send_readme_quickstart(
    message: Message, url: str, dest: Path
) -> None:
    """Try to summarise the cloned repo's README; silently skip on errors.

    Phase 6 of the dual-brain rework: after `/clone` in `/work` we feed
    the README to the active brain and dump the answer back to the chat
    so the user sees suggested commands BEFORE the rest of the batch
    runs.
    """
    found = _read_repo_readme(dest)
    if found is None:
        return
    readme_name, readme_text = found
    prompt = _build_readme_prompt(url, dest.name, readme_name, readme_text)
    await message.answer(
        f"📖 Нашёл <code>{_html_escape(readme_name)}</code> — спрашиваю активный мозг "
        "что попробовать первым делом…"
    )
    with contextlib.suppress(Exception):
        await message.bot.send_chat_action(message.chat.id, "typing")
    try:
        summary = await asyncio.wait_for(
            oneshot_summary(prompt, purpose="readme"),
            timeout=_README_REPLY_TIMEOUT_S,
        )
    except NoApiKeyError as exc:
        await message.answer(
            "README прочитать не смог — нет ключа активного мозга.\n"
            f"{_html_escape(str(exc))}"
        )
        return
    except TimeoutError:
        await message.answer(
            "README прочитать не успел (мозг отвечал >60 с). Пропускаю."
        )
        return
    except Exception as exc:  # noqa: BLE001 — never break the batch on README
        logger.exception("readme summary failed for %s", url)
        await message.answer(
            f"README прочитать не смог: {_html_escape(str(exc))}"
        )
        return
    if not summary.strip():
        return
    await message.answer(
        "📖 <b>Quick-Start из README:</b>"
    )
    await _send_long(message, summary, code=False)


async def _work_run_line(message: Message, line: str) -> None:
    """Run one line of a /work batch and send its output back.

    Recognises the same slash-commands as the bot's normal interface
    (``/exec``, ``/clone``, ``/project``, ``/cd``, ``/git``), plus bare
    bash lines without a leading slash. Each shell command is given
    ``WORK_EXEC_TIMEOUT`` instead of the global ``EXEC_TIMEOUT``.
    """
    user_id = message.from_user.id if message.from_user else 0
    header = f"<b>$ {_html_escape(line)}</b>"
    parts = line.split(maxsplit=1)
    cmd = parts[0]
    rest = parts[1].strip() if len(parts) > 1 else ""
    cwd = storage.get_cwd(user_id)

    if cmd in ("/exec", "/git"):
        if cwd is None:
            await message.answer(
                f"{header}\nСначала выбери проект: /projects или /clone"
            )
            return
        shell_cmd = rest if cmd == "/exec" else f"git {rest}"
        if not shell_cmd.strip() or shell_cmd.strip() == "git":
            await message.answer(f"{header}\nПустая команда")
            return
        await message.answer(header)
        with contextlib.suppress(Exception):
            await message.bot.send_chat_action(message.chat.id, "typing")
        try:
            result = await exec_bash(cwd, shell_cmd, timeout=WORK_EXEC_TIMEOUT)
        except ToolError as exc:
            await message.answer(f"Ошибка: {_html_escape(str(exc))}")
            return
        await _send_long(message, result)
        return

    if cmd == "/clone":
        args = rest.split()
        if not args:
            await message.answer(f"{header}\nИспользование: /clone &lt;url&gt; [имя]")
            return
        url = args[0]
        name = args[1] if len(args) > 1 else None
        await message.answer(f"{header}\nКлонирую…")
        try:
            dest = await clone_repo(url, name)
        except ToolError as exc:
            await message.answer(f"Ошибка: {_html_escape(str(exc))}")
            return
        storage.set_cwd(user_id, dest)
        await message.answer(
            f"Готово. Проект: <b>{dest.name}</b>\ncwd: <code>{dest}</code>"
        )
        # Phase 6: README-reader. Read repo's README and ask the active
        # brain for a Quick-Start. This runs synchronously so the
        # suggestions land in chat BEFORE the next batch line executes,
        # which is exactly what the user asked for.
        await _maybe_send_readme_quickstart(message, url, dest)
        return

    if cmd == "/project":
        if not rest:
            await message.answer(f"{header}\nИспользование: /project &lt;имя&gt;")
            return
        target = PROJECTS_DIR / rest
        if not target.exists():
            await message.answer(
                f"{header}\nПроект <code>{_html_escape(rest)}</code> не найден"
            )
            return
        storage.set_cwd(user_id, target)
        await message.answer(f"{header}\ncwd: <code>{target}</code>")
        return

    if cmd == "/cd":
        if cwd is None:
            await message.answer(f"{header}\nСначала выбери проект")
            return
        rel = rest or "."
        target = (cwd / rel).resolve()
        project_root = project_root_for(cwd).resolve()
        if project_root not in target.parents and target != project_root:
            await message.answer(f"{header}\nНельзя выйти за пределы проекта")
            return
        if not target.exists() or not target.is_dir():
            await message.answer(
                f"{header}\nНе найдена директория: <code>{_html_escape(str(target))}</code>"
            )
            return
        storage.set_cwd(user_id, target)
        await message.answer(f"{header}\ncwd: <code>{target}</code>")
        return

    if cmd.startswith("/"):
        await message.answer(
            f"{header}\nКоманда <code>{_html_escape(cmd)}</code> не поддерживается в /work."
        )
        return

    # Bare bash line — no leading slash.
    if cwd is None:
        await message.answer(
            f"{header}\nСначала выбери проект: /projects или /clone"
        )
        return
    await message.answer(header)
    with contextlib.suppress(Exception):
        await message.bot.send_chat_action(message.chat.id, "typing")
    try:
        result = await exec_bash(cwd, line, timeout=WORK_EXEC_TIMEOUT)
    except ToolError as exc:
        await message.answer(f"Ошибка: {_html_escape(str(exc))}")
        return
    await _send_long(message, result)


@router.message(StateFilter(WorkStates.awaiting_commands), F.text)
async def capture_work_batch(message: Message, state: FSMContext) -> None:
    """Consume the /work command-list message and run lines sequentially."""
    if not _is_authorized(message):
        await _deny(message)
        return
    raw = (message.text or "").strip()
    if not raw:
        await message.answer("Пустое сообщение. Пришли команды или /cancel.")
        return
    lines = [ln.strip() for ln in raw.splitlines()]
    lines = [ln for ln in lines if ln and not ln.startswith("#")]
    if not lines:
        await message.answer("Нечего выполнять. /cancel чтобы выйти.")
        return
    await message.answer(f"Принял <b>{len(lines)}</b> команд(ы). Выполняю по порядку…")
    for line in lines:
        try:
            await _work_run_line(message, line)
        except Exception as exc:  # noqa: BLE001 — report and continue with the batch
            logger.exception("/work line failed: %s", line)
            await message.answer(f"Ошибка строки: {_html_escape(str(exc))}")
        # Tiny pause so Telegram doesn't rate-limit us on bursty output.
        await asyncio.sleep(0.2)
    await state.clear()
    await message.answer("Готово. Все команды выполнены. /work — новая серия.")


async def run_work_batch(
    message: Message,
    lines: list[str],
    title: str | None = None,
    user_id: int | None = None,
) -> None:
    """Run a fixed list of `/work`-style commands sequentially.

    Public entry-point for callers outside the FSM flow (e.g. main-menu
    install buttons in ``wizard.py``). Behaves like
    :func:`capture_work_batch` but takes the list as an argument instead
    of parsing it from an incoming message, and does not touch FSM state.
    Same per-line executor (:func:`_work_run_line`) is used so output
    formatting and timeout behaviour stay consistent with `/work`.

    ``message`` is used purely as a destination for replies; ``user_id``
    should be passed explicitly when this is invoked from a callback
    query, because ``message.from_user`` of an inline-keyboard message
    is the *bot* itself, not the user who pressed the button.
    """
    from .access import can_use

    if user_id is not None:
        owner_id = storage.get_owner_id()
        authorised = (
            user_id in ALLOWED_USER_IDS
            if owner_id is None and ALLOWED_USER_IDS
            else can_use(user_id)
        )
        if not authorised:
            await message.answer(
                "Этот бот принадлежит другому владельцу.\n"
                f"Твой telegram id: <code>{user_id}</code>"
            )
            return
    else:
        if not _is_authorized(message):
            await _deny(message)
            return
    if not _role_chosen():
        await message.answer(_ROLE_REQUIRED)
        return
    cleaned = [ln.strip() for ln in lines]
    cleaned = [ln for ln in cleaned if ln and not ln.startswith("#")]
    if not cleaned:
        await message.answer("Пустой батч.")
        return
    header = (
        f"<b>{_html_escape(title)}</b> — {len(cleaned)} команд(ы), выполняю по порядку…"
        if title
        else f"Запускаю {len(cleaned)} команд(ы) по порядку…"
    )
    await message.answer(header)
    for line in cleaned:
        try:
            await _work_run_line(message, line)
        except Exception as exc:  # noqa: BLE001 — report and continue with the batch
            logger.exception("install batch line failed: %s", line)
            await message.answer(f"Ошибка строки: {_html_escape(str(exc))}")
        await asyncio.sleep(0.2)
    suffix = f" «{_html_escape(title)}»" if title else ""
    await message.answer(f"Готово.{suffix}")


# ---- /dl, /jobs (Lilush pipeline) ---------------------------------------


def _looks_like_url(s: str) -> bool:
    s = s.strip().lower()
    return s.startswith(("http://", "https://"))


@router.message(Command("dl"))
async def cmd_dl(message: Message, command: CommandObject) -> None:
    """Enqueue a download job for ``url`` and chain through the pipeline."""
    if not _is_authorized(message):
        await _deny(message)
        return
    url = (command.args or "").strip()
    if not url or not _looks_like_url(url):
        await message.answer(
            "Использование: <code>/dl &lt;url&gt;</code>\n"
            "Принимается любая ссылка, поддерживаемая yt-dlp "
            "(YouTube, Vimeo, прямые .mp4, и т.д.)."
        )
        return
    queue = get_default_queue()
    await queue.enqueue("download", {"url": url}, chat_id=message.chat.id)
    # Tail with project prompt only AFTER a role has been chosen — until
    # then we keep the response narrow so the user isn't told to pick a
    # project before they've picked a role.
    if _role_chosen():
        tail = (
            "\n\n<i>скачивание → анализ → нарезка → SEO → публикация</i>\n\n"
            "Прогресс — команда <code>/jobs</code>. "
            "Привязать к проекту — <code>/projects</code> или <code>/clone</code>."
        )
    else:
        tail = (
            "\n\n<i>скачивание → анализ → нарезка → SEO → публикация</i>\n\n"
            "Прогресс — команда <code>/jobs</code>."
        )
    await message.answer(
        f"🎬 <b>Принято в полный конвейер</b>\n\n"
        f"<a href=\"{url}\">ссылка</a> → анализ → нарезка → SEO → "
        f"публикация.{tail}"
    )


@router.message(Command("jobs"))
async def cmd_jobs(message: Message) -> None:
    """List the caller's recent jobs with status."""
    if not _is_authorized(message):
        await _deny(message)
        return
    queue = get_default_queue()
    jobs = await queue.list_by_chat(message.chat.id, limit=20)
    if not jobs:
        await message.answer("Активных и завершённых джобов пока нет. Запусти <code>/dl &lt;url&gt;</code>.")
        return
    lines = ["<b>Последние джобы</b>"]
    for job in jobs:
        marker = {
            "queued": "⏳",
            "running": "▶️",
            "done": "✓",
            "failed": "✗",
        }.get(job.status, "?")
        lines.append(f"{marker} <code>#{job.id:>4}</code>  {job.kind:<9}  {job.status}")
    await message.answer("\n".join(lines))


# ---- /tokens (LLM token spend stats) -------------------------------------


@router.message(Command("tokens"))
async def cmd_tokens(message: Message) -> None:
    """Show aggregated LLM-token usage from the local log."""
    if not _is_authorized(message):
        await _deny(message)
        return
    from .token_tracker import format_token_stats

    await message.answer(format_token_stats())


# ---- /keys, /setkey, /delkey ---------------------------------------------


@router.message(Command("keys"))
async def cmd_keys(message: Message) -> None:
    if not _is_authorized(message):
        await _deny(message)
        return
    info = storage.list_provider_keys()
    lines = ["<b>API-ключи</b>"]
    for provider, data in info.items():
        badge = (
            "не задан"
            if data["source"] == "none"
            else f"{data['masked']}  ({data['source']})"
        )
        lines.append(f"  <code>{provider:10}</code> {badge}")
    lines.append("")
    lines.append("<i>source=telegram — поставлен через /setkey, source=env — из переменной окружения</i>")
    await message.answer("\n".join(lines))


@router.message(Command("setkey"))
async def cmd_setkey(message: Message, command: CommandObject) -> None:
    if not _is_authorized(message):
        await _deny(message)
        return
    args = (command.args or "").split(maxsplit=1)
    if len(args) < 2:
        await message.answer(
            "Использование: <code>/setkey &lt;provider&gt; &lt;key&gt;</code>\n"
            f"Provider: {', '.join(KNOWN_PROVIDERS)}"
        )
        return
    provider, key = args[0].lower(), args[1].strip()
    # Try to delete the user's message right away — the key was sent in plaintext.
    # Older bots without delete permissions silently fall through.
    with contextlib.suppress(Exception):
        await message.delete()
    try:
        storage.set_provider_key(provider, key)
    except ValueError as exc:
        await message.answer(f"Ошибка: {_html_escape(str(exc))}")
        return
    masked = storage.list_provider_keys()[provider]["masked"]
    await message.answer(
        f"Ключ <code>{provider}</code> сохранён ({masked}).\n"
        "Сообщение с ключом удалено из чата."
    )


@router.message(Command("delkey"))
async def cmd_delkey(message: Message, command: CommandObject) -> None:
    if not _is_authorized(message):
        await _deny(message)
        return
    provider = (command.args or "").strip().lower()
    if not provider:
        await message.answer(
            f"Использование: <code>/delkey &lt;provider&gt;</code> ({', '.join(KNOWN_PROVIDERS)})"
        )
        return
    if storage.delete_provider_key(provider):
        await message.answer(f"Ключ <code>{provider}</code> удалён.")
    else:
        await message.answer(
            f"Ключ <code>{provider}</code> не был задан через /setkey "
            "(возможно ещё стоит env-переменная)."
        )


# ---- /models, /setmodel --------------------------------------------------


@router.message(Command("models"))
async def cmd_models(message: Message) -> None:
    if not _is_authorized(message):
        await _deny(message)
        return
    active = storage.get_model() or DEFAULT_MODEL
    lines = ["<b>Модели</b> (текущая помечена ▶)"]
    for model_id, label in MODEL_CATALOGUE:
        marker = "▶" if model_id == active else "  "
        lines.append(f"{marker} <code>{model_id}</code> — {label}")
    lines.append("")
    lines.append(
        "Переключить: <code>/setmodel &lt;model&gt;</code>\n"
        "Можно указать любую модель OpenRouter — список выше это просто рекомендации."
    )
    await message.answer("\n".join(lines))


@router.message(Command("setmodel"))
async def cmd_setmodel(message: Message, command: CommandObject) -> None:
    if not _is_authorized(message):
        await _deny(message)
        return
    model = (command.args or "").strip()
    if not model:
        await message.answer(
            "Использование: <code>/setmodel &lt;model&gt;</code>\n"
            "Список вариантов: /models"
        )
        return
    storage.set_model(model)
    await message.answer(
        f"Активная модель: <code>{_html_escape(model)}</code>\n"
        "Применится со следующего сообщения агенту."
    )


# ---- /enable, /disable ----------------------------------------------------


@router.message(Command("enable"))
async def cmd_enable(message: Message) -> None:
    if not _is_authorized(message):
        await _deny(message)
        return
    storage.set_enabled(True)
    await message.answer("Бот включён. Жду сообщений.")


@router.message(Command("disable"))
async def cmd_disable(message: Message) -> None:
    if not _is_authorized(message):
        await _deny(message)
        return
    storage.set_enabled(False)
    await message.answer(
        "Бот выключен. Текстовые сообщения и команды кроме /enable, /help, /keys будут игнорироваться."
    )


# ---- /brain, /setbrain ---------------------------------------------------


@router.message(Command("brain"))
async def cmd_brain(message: Message) -> None:
    if not _is_authorized(message):
        await _deny(message)
        return
    brain = storage.get_brain()
    if brain == "devin":
        await message.answer(
            "Brain: <b>devin</b>\n"
            "Бот не отвечает автоматически. Входящие пишутся в <code>data/inbox.log</code>; Devin (в своём чате с шелл-доступом к серверу) читает их и отвечает через <code>python -m bot.send</code>.\n"
            "Обратно в авто: <code>/setbrain auto</code>"
        )
    else:
        await message.answer(
            f"Brain: <b>auto</b>\n"
            f"Активная модель: <code>{_html_escape(storage.get_model() or DEFAULT_MODEL)}</code>"
        )


@router.message(Command("setbrain"))
async def cmd_setbrain(message: Message, command: CommandObject) -> None:
    if not _is_authorized(message):
        await _deny(message)
        return
    raw = (command.args or "").strip().lower()
    # Take only the first whitespace-separated token as the mode; ignore
    # anything trailing so users can copy-paste like '/setbrain devin посчитай 8+1'
    # without getting a usage error.
    mode = raw.split(maxsplit=1)[0] if raw else ""
    if mode not in ("auto", "devin"):
        await message.answer(
            "Использование: <code>/setbrain auto|devin</code>\n"
            "• <b>auto</b> — бот отвечает через LLM (текущая модель в /models).\n"
            "• <b>devin</b> — бот логирует в inbox.log и ждёт ручного ответа от Devin-сессии."
        )
        return
    storage.set_brain(mode)
    if mode == "devin":
        await message.answer(
            "Brain: <b>devin</b>. Бот будет писать входящие в <code>data/inbox.log</code> и отвечать краткой квитанцией. Ожидаю что Devin пришлёт ответ через <code>python -m bot.send</code>."
        )
    else:
        await message.answer(
            f"Brain: <b>auto</b>. Активная модель: <code>{_html_escape(storage.get_model() or DEFAULT_MODEL)}</code>."
        )


# ---- text handler ---------------------------------------------------------


def _make_status_updater(
    message: Message,
) -> tuple[Callable[[str], Awaitable[None]], Callable[[], Awaitable[None]]]:
    """Build a status-update closure honoring the chat's "Соображалка" style.

    Thin wrapper around :func:`addons.thinking_style.make_status_runner` so
    callers don't need to know the addon exists. The returned pair has
    identical semantics to the previous inline implementation — first
    ``update(text)`` call sends a status bubble, subsequent calls edit
    it in place (or do nothing in ``typing``-only mode), and
    ``finish()`` deletes the bubble (or cancels the typing-loop).
    """
    from .addons.thinking_style import make_status_runner

    def _model_hint() -> str:
        return storage.get_model() or DEFAULT_MODEL

    return make_status_runner(message, model_hint=_model_hint)


# Active in-flight agent runs keyed by a short hex run_id. The «🛑 Отмена»
# inline button posts ``cancel:<run_id>`` and the handler below sets the
# corresponding :class:`asyncio.Event`, which :func:`bot.agent.run_agent`
# polls between iterations.
_CANCEL_EVENTS: dict[str, asyncio.Event] = {}


async def _send_cancel_button(message: Message, run_id: str) -> Message | None:
    """Send a small message with a single «🛑 Отмена» inline button.

    Returns the sent Message (so the caller can delete it after the
    run finishes) or ``None`` if sending fails — cancellation is a
    nice-to-have, never a hard requirement.
    """
    from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

    kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🛑 Отмена", callback_data=f"cancel:{run_id}")]
        ]
    )
    try:
        return await message.answer("В работе…", reply_markup=kb)
    except Exception:  # noqa: BLE001
        logger.debug("cancel button send failed", exc_info=True)
        return None


@router.callback_query(F.data.startswith("cancel:"))
async def on_cancel_callback(query) -> None:
    """Owner / authorized user clicked the «🛑 Отмена» button on an
    in-flight agent run. Set the event so :func:`run_agent` aborts at
    its next checkpoint."""
    data = query.data or ""
    run_id = data.split(":", 1)[1] if ":" in data else ""
    event = _CANCEL_EVENTS.get(run_id)
    if event is None:
        await query.answer("Задача уже завершилась.", show_alert=False)
        return
    event.set()
    await query.answer("Отменяю…", show_alert=False)


@router.message(F.text & ~F.text.startswith("/"))
async def handle_text(message: Message) -> None:
    if not _is_authorized(message):
        await _deny(message)
        return
    user_id = message.from_user.id if message.from_user else None
    # Inbox logging happens in middleware above.
    if not storage.is_enabled():
        await message.answer(
            "Бот выключен. Включи через <code>/enable</code>."
        )
        return
    if storage.get_brain() == "devin":
        await message.answer(
            "Принято. Brain=devin — жди ответ от Devin.\n"
            "<i>Сообщение записано в inbox.log. Чтобы бот опять отвечал сам — /setbrain auto</i>"
        )
        return
    cwd = storage.get_cwd(user_id) if user_id is not None else None
    await message.bot.send_chat_action(message.chat.id, "typing")
    on_status, finish_status = _make_status_updater(message)

    import secrets

    run_id = secrets.token_hex(8)
    cancel_event = asyncio.Event()
    _CANCEL_EVENTS[run_id] = cancel_event
    cancel_msg = await _send_cancel_button(message, run_id)

    try:
        answer = await run_agent(
            user_id or 0,
            message.text or "",
            cwd,
            on_status=on_status,
            cancel_event=cancel_event,
        )
    except NoApiKeyError as exc:
        await finish_status()
        if cancel_msg is not None:
            with contextlib.suppress(Exception):
                await cancel_msg.delete()
        _CANCEL_EVENTS.pop(run_id, None)
        await message.answer(str(exc))
        return
    except Exception as exc:  # noqa: BLE001
        logger.exception("agent failed")
        await finish_status()
        if cancel_msg is not None:
            with contextlib.suppress(Exception):
                await cancel_msg.delete()
        _CANCEL_EVENTS.pop(run_id, None)
        await message.answer(f"Ошибка агента: {_html_escape(str(exc))}")
        return
    finally:
        _CANCEL_EVENTS.pop(run_id, None)
    await finish_status()
    if cancel_msg is not None:
        with contextlib.suppress(Exception):
            await cancel_msg.delete()
    await _send_long(message, answer, code=False)
    # «🔊 Голосовой ответчик» — when enabled in Settings, dub the agent's
    # reply via Supertonic. Best-effort: never raises, just skips voice
    # if TTS isn't installed / model isn't loaded / sanitiser produced
    # nothing readable.
    from .tts import maybe_send_voice_reply  # local import keeps cold-start cheap

    await maybe_send_voice_reply(message, answer)


# ---- unknown / empty fallback --------------------------------------------


@router.message(F.text.startswith("/"))
async def handle_unknown_command(message: Message) -> None:
    """Catch-all for unknown commands — show capability summary instead of silence.

    All Command(...) handlers above are matched first; this handler only
    fires when nothing else claimed the message.
    """
    if not _is_authorized(message):
        await _deny(message)
        return
    cmd = (message.text or "/").split()[0]
    await message.answer(
        f"Команда <code>{_html_escape(cmd)}</code> не известна.\n\n" + WHOAMI_TEXT
    )


@router.message()
async def handle_anything_else(message: Message) -> None:
    """Fallback for non-text messages (stickers, voice, etc.).

    Photo handling lives in ``bot/media_ui.py`` (registered before this
    router in main.py).
    """
    if not _is_authorized(message):
        await _deny(message)
        return
    await message.answer(WHOAMI_TEXT)
