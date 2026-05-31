import asyncio
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path

from openai import AsyncOpenAI
from openai._exceptions import APIError

from .config import (
    AGENT_MAX_STEPS,
    AGENT_MAX_TOKENS,
    APP_TITLE,
    DEFAULT_MODEL,
    FALLBACK_MODELS,
    HTTP_REFERER,
    OPENROUTER_BASE_URL,
)
from .storage import storage
from .token_tracker import record as record_token_usage
from .tools import TOOL_DEFINITIONS, ToolError, dispatch_tool

# Coroutine the agent calls to push a mini-status string to the user
# ("🔄 Думаю...", "🔄 Вызываю tool list_dir", "✅ Готово", ...).
# Empty no-op default is used in non-Telegram contexts (tests / CLI).
StatusUpdate = Callable[[str], Awaitable[None]]


async def _noop_status(_: str) -> None:
    return None


def _current_rss_mb() -> int:
    """Local import shim around :func:`bot.addons.ram_guard.current_rss_mb`.

    Kept here so :mod:`bot.agent` doesn't need to import the addon at
    module load time (avoids circular import in early bootstrap).
    """
    try:
        from .addons.ram_guard.handlers import current_rss_mb
        return current_rss_mb()
    except Exception:  # noqa: BLE001
        return 0


def _check_ram_state(limit_mb: int, behavior: str) -> tuple[int, str]:
    """Return ``(rss_mb, action)`` where action ∈ ``{ok, compress, refuse}``.

    Caller should re-read limit / behavior from storage once per turn —
    we take them as args so the per-step check is cheap.
    """
    if limit_mb <= 0:
        return 0, "ok"
    rss = _current_rss_mb()
    if rss == 0:
        return 0, "ok"
    pct = rss * 100 // limit_mb
    if behavior == "compress":
        if pct >= 95:
            return rss, "refuse"
        if pct >= 80:
            return rss, "compress"
        return rss, "ok"
    # behavior == "refuse" — only act when the actual limit is crossed.
    if pct >= 100:
        return rss, "refuse"
    return rss, "ok"


def _compress_tool_results(messages: list[dict]) -> int:
    """Truncate large ``role=tool`` payloads in-place. Returns chars removed.

    Called when RAM is at 80% in compress mode — we'd rather have the
    LLM lose tool-result detail than crash the whole process.
    """
    removed = 0
    for m in messages:
        if m.get("role") != "tool":
            continue
        content = m.get("content") or ""
        if len(content) > 1000:
            m["content"] = content[:800].rstrip() + "\n...[trimmed by RAM guard]"
            removed += len(content) - len(m["content"])
    return removed


# Map technical tool names → natural Russian gerund phrases, matching the
# tg_bot+(13) thinking-stream UX ("Смотрю на картинку…" / "Рисую…").
_TOOL_VERBS: dict[str, str] = {
    "list_dir": "Смотрю что в проекте…",
    "read_file": "Читаю файл…",
    "write_file": "Правлю файл…",
    "exec_bash": "Запускаю команду…",
    "search": "Ищу по проекту…",
    "git_status": "Смотрю git-статус…",
    "git_diff": "Смотрю что изменилось…",
    "git_commit": "Делаю коммит…",
    "clone_repo": "Клонирую репо…",
    "list_projects": "Смотрю какие проекты есть…",
    "switch_project": "Переключаюсь на проект…",
    "tavily_search": "Ищу в интернете…",
    "brave_search": "Ищу в интернете…",
    "exa_search": "Ищу в интернете…",
    "firecrawl_scrape": "Читаю страницу…",
    "apify_run_actor": "Запускаю скрейпер…",
    "github_search_code": "Ищу код на GitHub…",
    "github_get_file": "Читаю файл с GitHub…",
}


def _describe_tools(tool_calls) -> str:
    """Turn a list of OpenAI/OpenRouter tool calls into a single status line.

    Mirrors tg_bot+(13)'s style — plain Russian, no technical names, no
    emojis, no step numbers. Falls back to ``Работаю с инструментом X``
    for unknown tools so new ones still surface something readable.
    """
    seen: list[str] = []
    for tc in tool_calls:
        name = tc.function.name
        phrase = _TOOL_VERBS.get(name) or f"Работаю с инструментом {name}…"
        if phrase not in seen:
            seen.append(phrase)
    return " ".join(seen) if seen else "Работаю…"


logger = logging.getLogger(__name__)

# Base system rules shared by every persona. Persona-specific framing is
# appended in ``_build_system_prompt()`` below depending on whether the
# user has selected a role via /role. The "auto-clone" line is templated
# so we can flip it ON/OFF based on the user's setting.
_BASE_RULES_TEMPLATE = """You are a Telegram bot assistant.

General rules:
- Always answer in the user's language (Russian by default for this user).
- You have access to two groups of tools.
  • Project tools (work inside the active git repo): list_dir, read_file, write_file, exec_bash, clone_repo, list_projects, switch_project. Use them when the user asks you to look at, change, or run something in code. Do not invent file contents — read them.
  • External research tools (no project needed): tavily_search / brave_search / exa_search for the web, firecrawl_scrape to read a specific URL, apify_run_actor for platform scrapers (Reddit / TikTok / Twitter / etc.), github_search_code + github_get_file for GitHub spelunking. Use these whenever the answer isn't in the current project or your training data — fresh info, recent events, "найди мне", "посмотри на сайте", etc. Each requires its own API key in /setup → Внешние API; if a key is missing the tool will say so and you should pass that message to the user.
- Be concise: in chat replies aim for short paragraphs. Long file content goes via tools, not into the reply.
- After making changes, suggest a git commit message; do not commit unless the user confirms or explicitly asks.
- If the user asks for git operations, use exec_bash with git commands.

Formatting (the user explicitly asked for this — обязательно следуй):
- Render output as Telegram HTML, parse_mode=HTML. Plain text is fine; HTML tags are used only where they help readability.
- Put a BLANK LINE between paragraphs / between distinct ideas. Don't dump a wall of text. One thought per paragraph, separated by an empty line — same way a human would write a chat message that "breathes".
- All URLs MUST be clickable: wrap them in <a href="..."> tags rather than dropping raw URLs into text. Example: instead of «посмотри https://example.com», write «посмотри <a href="https://example.com">example.com</a>».
- Use <b> for emphasis on key terms, <i> for nuance, <code> for paths/commands/short snippets, <pre> for multi-line code blocks. Never invent unsupported HTML — Telegram only accepts: b, strong, i, em, u, s, strike, del, code, pre, a, blockquote.
- Lists: use a bullet like «• » at the start of each line. Don't try to render markdown lists (Telegram won't parse them).
- Never use markdown (# headings, **bold**, [text](url)). Always HTML.
- Mention commands as <code>/dl</code>, file paths as <code>data/projects/foo</code>.
- Escape literal <, >, & in user content with &lt; &gt; &amp; before sending — but you usually won't need to since your replies are prose.

Working with git repositories (use the tools, NOT slash-commands):
{auto_clone_rule}
- If they ask "что у меня в проектах" / "какие у меня репо" — call list_projects (or just answer from the project list already injected below if it's there).
- If they say "переключись на X" / "посмотри в проекте X" — call switch_project(name="X"). If it fails because the project isn't there, clone_repo it.
- After clone_repo or switch_project the active project is updated automatically for the rest of this turn — start exploring with list_dir(path=".") immediately.
- For ordinary chat that has nothing to do with code/files, just respond — don't gratuitously clone or list things.
"""

_AUTO_CLONE_RULE_ON = (
    "- If the user shares a GitHub / GitLab / Bitbucket URL and asks you to "
    "look at the code, summarize it, find something in it, or work with it — "
    "DO NOT tell them to run /clone. Instead call clone_repo(url=...) "
    "yourself, then list_dir / read_file / exec_bash to do the work. "
    "Auto-clone is currently ENABLED in this bot's settings."
)

_AUTO_CLONE_RULE_OFF = (
    "- Auto-clone is currently DISABLED in this bot's settings. Do NOT "
    "call clone_repo when the user merely shares a URL — wait for an "
    "explicit instruction like \"склонируй\" / \"clone\" / \"загрузи\". "
    "If they just paste a URL, briefly tell them you can clone it on "
    "request and ask if they want that."
)


def _capabilities_log() -> str:
    """Return the contents of ``DATA_DIR/capabilities.md`` if present.

    The file is a free-form markdown changelog of new bot capabilities
    written by the deploy/maintenance code (see ``bot/capabilities.py``
    and ``bot/main.py``). We inject it verbatim so the agent always
    "knows" what it can do, even after the chat history rotates past
    the message where a feature was introduced.

    Truncated to 4000 characters — anything longer should be summarised
    rather than dumped into every prompt.
    """
    from .config import DATA_DIR

    path = DATA_DIR / "capabilities.md"
    if not path.exists():
        return ""
    try:
        text = path.read_text(errors="replace").strip()
    except Exception:  # noqa: BLE001
        return ""
    if not text:
        return ""
    if len(text) > 4000:
        text = text[:4000].rstrip() + "\n... (truncated)"
    return text


def _project_inventory() -> str:
    """One-line-per-repo inventory injected into the system prompt.

    The agent only remembers the last ``HISTORY_LIMIT`` (40) messages,
    so without this it would forget which repos are available after
    enough chatter. Reading from storage on every turn keeps it
    accurate without spending tokens on a list_projects call.
    """
    repos = storage.list_github_repos()
    if not repos:
        return (
            "No git projects cloned yet. Call clone_repo(url=...) when "
            "the user asks you to look at one."
        )
    lines = []
    for repo in repos:
        desc = repo.get("description") or "(no description)"
        # Single line, truncated, no markdown headings — keep token cost
        # bounded for users with lots of repos.
        if len(desc) > 140:
            desc = desc[:140].rstrip() + "…"
        lines.append(f"- {repo['name']}: {desc}")
    return "Currently-cloned git projects on this server:\n" + "\n".join(lines)

# Free-chat framing used before the user picks a role via /role. The bot
# is encouraged to chat openly about anything so the user can probe its
# capabilities. Triggered when active persona is the default ``boss`` AND
# no override is set.
_DISCOVERY_FRAMING = """
Mode: DISCOVERY (no specific role selected yet).
The user is exploring your capabilities. Chat freely on any topic — casual conversation, technical help, file/code work, opinions, jokes — whatever they ask. If they ask "what can you do?" or "who are you?", answer openly.
Tell them they can run /role at any time to switch into a specialized persona (researcher, debater, coder, devops, etc.) once they want to focus your behavior.
"""

# Role-framing used after the user has picked a non-default persona. The
# bot is constrained to stay on-task within that role.
_ROLE_FRAMING_TEMPLATE = """
Your role: {display_name} — {title}.
Department: {department}. Rank: {rank}.

Role brief: {description}

STAY IN ROLE. Focus strictly on tasks that fit your role above. If the user asks for something far outside this role (e.g. asks a Research Lead to write code, or asks a Watchdog to run a debate), politely redirect: tell them which role would be appropriate and suggest they run /role to switch.
Don't break character. Don't help with arbitrary off-topic requests — keep the user on-task.
"""


def _build_system_prompt() -> str:
    """Compose the system prompt based on the active persona and override.

    Dynamic bits injected on every turn (so the agent never goes stale):

    * Auto-clone rule — ON / OFF depending on the user's setting in
      ⚙️ Настройки → 🐙 GitHub → 🔁 Авто-клонирование.
    * Project inventory — one line per cloned repo so the agent knows
      what's available without burning a list_projects call.
    * Capabilities log — DATA_DIR/capabilities.md if present.
    * Persona framing — DISCOVERY for the default boss, role-specific
      otherwise (so the bot stays on-task after /role).
    """
    # Lazy import to keep the module graph acyclic.
    from .persona import get_persona

    auto_rule = (
        _AUTO_CLONE_RULE_ON
        if storage.get_github_auto_clone()
        else _AUTO_CLONE_RULE_OFF
    )
    base = _BASE_RULES_TEMPLATE.format(auto_clone_rule=auto_rule)

    persona = get_persona()
    override = storage.get_persona_override()
    is_default_boss = persona.key == "boss" and not override
    if is_default_boss:
        framing = _DISCOVERY_FRAMING
    else:
        framing = _ROLE_FRAMING_TEMPLATE.format(
            display_name=persona.display_name,
            title=persona.title,
            department=persona.department,
            rank=persona.rank,
            description=persona.description,
        )

    # Inventory + capabilities go BEFORE the framing so the persona's
    # "STAY IN ROLE" / "discovery" framing is the last thing the model
    # reads — keeps emphasis right.
    inventory = _project_inventory()
    capabilities = _capabilities_log()
    extras: list[str] = []
    if inventory:
        extras.append("## Active GitHub projects on this server\n" + inventory)
    if capabilities:
        extras.append("## Bot capabilities log\n" + capabilities)
    extras_block = ("\n\n" + "\n\n".join(extras)) if extras else ""
    return base + extras_block + "\n" + framing


# Backwards-compat alias for callers/tests that imported the constant
# directly. ``_build_system_prompt()`` is the authoritative source —
# this static snapshot is only used by test fixtures that import the
# module without storage state.
SYSTEM_PROMPT = (
    _BASE_RULES_TEMPLATE.format(auto_clone_rule=_AUTO_CLONE_RULE_ON)
    + _DISCOVERY_FRAMING
)


class NoApiKeyError(RuntimeError):
    """Raised when no usable provider key is configured anywhere."""


def _slot_cfg(slot: str) -> tuple[str, str, str, str]:
    """Return ``(provider, api_key, base_url, model)`` for the given slot.

    Slot 1 reads the legacy top-level fields (so the existing user's
    config keeps working unchanged). Slot 2 reads the nested
    ``brain_slot2`` dict added with the dual-brain feature.
    """
    if slot == "2":
        s = storage.get_brain_slot2()
        return (
            (s.get("provider") or "custom"),
            (s.get("api_key") or ""),
            (s.get("base_url") or ""),
            (s.get("model") or ""),
        )
    return (
        storage.get_provider() or "openrouter",
        storage.get_provider_key("openrouter") or "",
        storage.get_base_url() or "",
        storage.get_model() or "",
    )


def _slot_is_configured(slot: str) -> bool:
    """A slot is *usable* when it has an api_key. Provider/url/model can
    fall back to sane defaults; the key is the only thing we cannot
    invent.
    """
    _, api_key, _, _ = _slot_cfg(slot)
    return bool(api_key)


def _slot_chain() -> list[str]:
    """Brain slots used for chat. Always returns ``["1"]``.

    Slots have fixed roles by design:
    * Slot 1 (or «Другое») answers chat.
    * Slot 2 is reserved for voice / photo overrides via Groq Whisper /
      Vision — it is NEVER used as a chat brain or chat-failover.

    Anyone changing slot routing should update this function,
    :func:`_build_client`, and the wizard «✅ Готово» handler
    (`cb_cfg_done` in bot/wizard.py) together to keep the three in sync.
    """
    return ["1"]


def _build_client_for_slot(slot: str) -> AsyncOpenAI:
    """Build an ``AsyncOpenAI`` configured against the given brain slot.

    Slot 1 with provider=openrouter → ``OPENROUTER_BASE_URL``. Anything
    else (custom / Groq / vLLM / ...) → ``storage.get_base_url()`` or the
    slot's stored ``base_url``. Raises :class:`NoApiKeyError` when the
    slot has no API key — we never silently swap a different slot in
    here, the failover wraps this at a higher level.
    """
    provider, api_key, base_url, _ = _slot_cfg(slot)
    if slot == "1" and provider != "custom":
        endpoint = OPENROUTER_BASE_URL
    else:
        endpoint = base_url or OPENROUTER_BASE_URL
    if not api_key:
        if slot == "2":
            raise NoApiKeyError(
                "У Мозга 2 не задан API-ключ. Зайди в /setup → 🧠 Авто мозг → "
                "Мозг 2 → 🔑 API ключ и вставь его."
            )
        if provider == "custom":
            raise NoApiKeyError(
                "Не задан API-ключ для кастомного endpoint-а. Открой /setup → 🔑 API ключ."
            )
        raise NoApiKeyError(
            "Не задан ключ OpenRouter. Поставь его командой:\n"
            "<code>/setkey openrouter sk-or-...</code>\n"
            "Получить ключ: https://openrouter.ai/keys"
        )
    headers = {"HTTP-Referer": HTTP_REFERER, "X-Title": APP_TITLE}
    return AsyncOpenAI(api_key=api_key, base_url=endpoint, default_headers=headers)


def _build_client() -> AsyncOpenAI:
    """Build the chat client. Always uses brain slot 1.

    The agent loop calls this once at the top of ``run_agent`` to surface
    :class:`NoApiKeyError` early. Slot 2 is reserved for voice/photo and
    is never used for chat — see :func:`_slot_chain`.
    """
    return _build_client_for_slot("1")


def _candidate_models_for_slot(slot: str) -> list[str]:
    """Active model first, then free fallbacks only if active itself is free.

    Premium models (e.g. ``anthropic/claude-opus-4.5``) intentionally do NOT
    fall back to free models — silent downgrade would be confusing. Slot 2
    uses its own configured model and does NOT inherit the OpenRouter
    free-model fallback list (Groq / custom endpoints don't speak those
    model names).
    """
    _, _, _, model = _slot_cfg(slot)
    active = model or (DEFAULT_MODEL if slot == "1" else "")
    if not active:
        return []
    candidates = [active]
    if slot == "1":
        is_free = ":free" in active or active in FALLBACK_MODELS
        if is_free:
            for m in FALLBACK_MODELS:
                if m != active and m not in candidates:
                    candidates.append(m)
    return candidates


def _candidate_models() -> list[str]:
    """Backwards-compat: model fallback chain for the chat brain (slot 1)."""
    return _candidate_models_for_slot("1")


def _build_messages(user_id: int, user_text: str) -> list[dict]:
    history = storage.get_history(user_id)
    return (
        [{"role": "system", "content": _build_system_prompt()}]
        + history
        + [{"role": "user", "content": user_text}]
    )


async def _call_model_for_slot(
    slot: str, messages: list[dict], purpose: str = "chat"
) -> tuple[object, str]:
    """Call the active model on the given brain slot, with model-fallback.

    Walks `_candidate_models_for_slot(slot)` and returns on the first
    successful response. Raises ``RuntimeError`` if every model in the
    slot fails — the caller then decides whether to fall over to the
    other slot.
    """
    client = _build_client_for_slot(slot)
    provider, _, _, _ = _slot_cfg(slot)
    last_error: Exception | None = None
    for model in _candidate_models_for_slot(slot):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
                temperature=0.2,
                max_tokens=AGENT_MAX_TOKENS,
            )
        except APIError as exc:
            logger.warning("slot %s model %s failed (APIError): %s", slot, model, exc)
            last_error = exc
            continue
        except Exception as exc:  # noqa: BLE001 — network / parse errors
            logger.warning("slot %s model %s failed (transport): %s", slot, model, exc)
            last_error = exc
            continue

        usage = getattr(resp, "usage", None)
        if usage is not None:
            pt = getattr(usage, "prompt_tokens", 0) or 0
            ct = getattr(usage, "completion_tokens", 0) or 0
            try:
                record_token_usage(
                    provider=provider,
                    model=model,
                    prompt_tokens=int(pt),
                    completion_tokens=int(ct),
                    purpose=purpose,
                )
            except Exception:  # noqa: BLE001 — never crash on stats
                logger.exception("token-tracker recording failed")

        choices = getattr(resp, "choices", None)
        if not choices:
            preview = repr(resp)[:300]
            logger.warning(
                "slot %s model %s returned response without .choices: %s",
                slot, model, preview,
            )
            last_error = RuntimeError(
                f"endpoint returned non-OpenAI response (no choices). "
                f"First 300 chars: {preview}"
            )
            continue
        return choices[0].message, model
    raise RuntimeError(f"slot {slot}: all models failed: {last_error}")


async def _call_model_with_failover(
    messages: list[dict], purpose: str = "chat"
) -> tuple[object, str, str]:
    """Run chat on slot 1, falling back through its free-model chain.

    Despite the name, this does NOT fail over to slot 2 — slot 2 is
    reserved for voice/photo (Groq override) by design. The «failover»
    here is the in-slot model chain (active model → free fallbacks)
    implemented in :func:`_call_model_for_slot`.

    Returns ``(message, used_model, used_slot)`` so the caller can show
    which model answered (handy for the ``[fallback model: …]`` tag in
    chat replies). ``used_slot`` is always ``"1"`` today; kept in the
    tuple so callers that destructure three values keep working.
    """
    last_error: Exception | None = None
    for slot in _slot_chain():
        try:
            msg, model = await _call_model_for_slot(slot, messages, purpose=purpose)
            return msg, model, slot
        except NoApiKeyError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("brain slot %s exhausted: %s", slot, exc)
            last_error = exc
            continue
    raise RuntimeError(f"all brain slots failed: {last_error}")


async def oneshot_summary(prompt: str, *, purpose: str = "readme") -> str:
    """One-shot, tool-less LLM call. Used by the README-reader in /work.

    Routes through the slot-aware failover path so an unavailable
    primary brain doesn't break this. Returns the assistant's text
    content (or an empty string when the model produced nothing).
    """
    messages: list[dict] = [
        {
            "role": "system",
            "content": (
                "Ты помогаешь юзеру быстро освоить только что склонированный "
                "GitHub-репозиторий. Отвечай по-русски, без лишней воды."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    # We don't want tool calls here — pass a hand-built request that
    # omits `tools=`, so the model can't decide to use them. Reusing
    # `_call_model_for_slot` would pull tools in.
    last_error: Exception | None = None
    for slot in _slot_chain():
        try:
            client = _build_client_for_slot(slot)
        except NoApiKeyError:
            raise
        provider, _, _, _ = _slot_cfg(slot)
        for model in _candidate_models_for_slot(slot):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=AGENT_MAX_TOKENS,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "oneshot_summary: slot %s model %s failed: %s",
                    slot, model, exc,
                )
                continue
            usage = getattr(resp, "usage", None)
            if usage is not None:
                pt = getattr(usage, "prompt_tokens", 0) or 0
                ct = getattr(usage, "completion_tokens", 0) or 0
                try:
                    record_token_usage(
                        provider=provider,
                        model=model,
                        prompt_tokens=int(pt),
                        completion_tokens=int(ct),
                        purpose=purpose,
                    )
                except Exception:  # noqa: BLE001
                    logger.exception("token-tracker recording failed")
            choices = getattr(resp, "choices", None)
            if not choices:
                continue
            msg = choices[0].message
            return (getattr(msg, "content", None) or "").strip()
    raise RuntimeError(f"oneshot_summary: all brains failed: {last_error}")


async def _call_model(
    client: AsyncOpenAI, messages: list[dict], purpose: str = "chat"
) -> tuple[object, str]:
    """Backwards-compat wrapper used by tests and any external callers.

    Ignores the ``client`` argument and routes through the slot-aware
    failover path so production behaviour matches what ``run_agent``
    actually does.
    """
    _ = client  # legacy signature; current implementation rebuilds clients
    msg, model, _slot = await _call_model_with_failover(messages, purpose=purpose)
    return msg, model


async def run_agent(
    user_id: int,
    user_text: str,
    cwd: Path | None,
    on_status: StatusUpdate | None = None,
    *,
    cancel_event: asyncio.Event | None = None,
) -> str:
    """Run the agent loop. ``on_status`` is called with progress strings
    for the mini-status indicator (caller edits a single TG message).

    If ``cancel_event`` is provided and gets set during the run (typically
    by the «🛑 Отмена» button in the status bubble), the loop aborts at
    the next safe checkpoint and returns a short «Задача отменена»
    string instead of an LLM answer.

    The bot's RSS is checked between iterations against the «💻 RAM»
    settings (``storage.get_ram_limit_mb`` / ``get_ram_behavior``):
    * ``compress`` mode → at 80% trim big tool results; at 95% abort.
    * ``refuse`` mode → just abort once the limit is crossed.
    """
    raw_status = on_status or _noop_status
    ram_show = storage.get_ram_show()
    ram_limit = storage.get_ram_limit_mb()
    ram_behavior = storage.get_ram_behavior()

    async def status(text: str) -> None:
        """Wrap ``on_status`` and (optionally) suffix RSS so the user
        sees how much RAM the bot is using right now."""
        if ram_show:
            rss = _current_rss_mb()
            if rss > 0:
                text = f"{text} (RSS {rss} MB)"
        await raw_status(text)

    # Validate the active slot up-front so we surface NoApiKeyError with a
    # helpful message before the user has to wait for the first model
    # round-trip.
    _build_client()
    messages = _build_messages(user_id, user_text)
    final_text = ""
    requested_model = storage.get_model() or DEFAULT_MODEL
    # Chat always uses slot 1 — slot 2 is reserved for voice/photo via
    # the Groq override. See :func:`_slot_chain`.
    requested_slot = "1"
    used_model = requested_model
    used_slot = requested_slot

    def _cancelled() -> bool:
        return cancel_event is not None and cancel_event.is_set()

    # Thinking-stream style copied from tg_bot+(13) — plain Russian,
    # no emojis, no step numbers, describes what's happening in human
    # terms instead of technical "step N: tool list_dir".
    await status("Думаю…")

    for step in range(AGENT_MAX_STEPS):
        if _cancelled():
            storage.append_history(user_id, {"role": "user", "content": user_text})
            return "Задача отменена."

        # RAM guard — check at the top of each step before the next
        # expensive LLM round-trip.
        rss, action = _check_ram_state(ram_limit, ram_behavior)
        if action == "refuse":
            storage.append_history(user_id, {"role": "user", "content": user_text})
            return (
                f"⚠️ Память сервера на пределе ({rss} MB / {ram_limit} MB). "
                "Разбей задачу на несколько шагов и нажми /reset, чтобы "
                "освободить контекст. Подробнее в /setup → 💻 RAM."
            )
        if action == "compress":
            removed = _compress_tool_results(messages)
            if removed > 0:
                logger.info(
                    "ram guard: compressed tool results, freed %d chars at RSS=%d MB",
                    removed, rss,
                )

        await status("Размышляю над ответом…" if step == 0 else "Думаю дальше…")
        msg, used_model, used_slot = await _call_model_with_failover(messages)
        tool_calls = getattr(msg, "tool_calls", None) or []

        assistant_entry: dict = {"role": "assistant", "content": msg.content or ""}
        if tool_calls:
            assistant_entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ]
        messages.append(assistant_entry)

        if not tool_calls:
            final_text = msg.content or ""
            break

        # Surface tool activity to the user so they see what the bot is doing.
        # Translate technical tool names into natural Russian — mirrors
        # tg_bot+(13)'s "Смотрю на картинку…" / "Рисую…" pattern.
        await status(_describe_tools(tool_calls))

        for tc in tool_calls:
            if _cancelled():
                storage.append_history(user_id, {"role": "user", "content": user_text})
                return "Задача отменена."
            try:
                result = await dispatch_tool(
                    tc.function.name,
                    tc.function.arguments,
                    cwd,
                    user_id=user_id,
                )
            except ToolError as exc:
                result = f"ERROR: {exc}"
            except Exception as exc:  # noqa: BLE001
                logger.exception("tool %s crashed", tc.function.name)
                result = f"ERROR: {exc}"
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.function.name,
                    "content": result[:8000],
                }
            )
            # ``clone_repo`` / ``switch_project`` mutate the user's active
            # project in storage. Re-read it so subsequent tool calls in
            # this same turn see the new cwd (otherwise the agent would
            # clone a repo and then try to read_file against the stale
            # previous project).
            if tc.function.name in {"clone_repo", "switch_project"}:
                cwd = storage.get_cwd(user_id) if user_id is not None else cwd
    else:
        final_text = (
            "(превышен лимит шагов агента — попробуй упростить запрос или /reset)"
        )

    storage.append_history(user_id, {"role": "user", "content": user_text})
    if final_text:
        storage.append_history(user_id, {"role": "assistant", "content": final_text})

    # Fallback / model-swap details intentionally NOT prefixed onto the
    # reply text — the user asked the bot to never say which brain or
    # model produced an answer (the "соображалка" / status bubble is
    # the only place that may surface that, and only when the user
    # explicitly picked the "белая + модель" thinking style). We do log
    # any deviation server-side so the operator can still tell what
    # happened.
    if used_slot != requested_slot:
        logger.info(
            "brain slot deviated: requested=%s used=%s",
            requested_slot, used_slot,
        )
    if used_model != requested_model:
        logger.info(
            "model deviated: requested=%s used=%s",
            requested_model, used_model,
        )
    await status("Готово")
    return final_text or "(пустой ответ)"
