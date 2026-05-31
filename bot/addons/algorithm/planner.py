"""AI plan-from-text for the algorithm addon.

Given a piece of free-form text (typically a news article or a
user-written "do these things" description), ask one of the bot's LLM
brains to translate it into:

* a short ``name`` (1-3 words, used as the slot button label), and
* an ordered ``steps`` list — strings phrased as imperative tasks the
  agent can execute one after another via ``bot.agent.run_agent``.

Brain priority (per user request):

1. **«Другое»** (custom) — any slot configured with
   ``provider == "custom"`` (= a user-supplied OpenAI-compatible
   endpoint). On request this *always* wins over OpenRouter / Groq.
2. **Slot 1** — whatever else is configured there.
3. **Slot 2** — whatever else is configured there.

The planner is purely a one-shot LLM call — it does NOT have access to
tools. That is intentional: the planner produces a plan, the executor
later asks the agent to perform each step (and the agent decides which
tools it needs).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from ...capabilities import CAPABILITIES

logger = logging.getLogger(__name__)


@dataclass
class Plan:
    name: str
    steps: list[str]
    used_brain: str  # "bomba" / "slot1" / "slot2" / "none"
    used_model: str  # e.g. "openrouter/anthropic/claude-3-haiku"
    raw: str = ""    # raw LLM response, kept for debugging / "перепродумать"


class NoBrainAvailable(RuntimeError):
    """Raised when none of the configured brains can answer."""


def _planner_chain() -> list[tuple[str, str]]:
    """Return ``[(label, slot_id), ...]`` in the order the planner
    should try them.

    ``label`` is a short tag used in logs and in the ``Plan.used_brain``
    field. ``slot_id`` is the same ``"1"`` / ``"2"`` string accepted by
    ``bot.agent._slot_cfg``.

    Both slots are considered, but slots whose ``provider`` is
    ``"custom"`` are surfaced *first* (= they are the «Другое»
    endpoint per the user's contract). Slots without an api_key
    are dropped entirely.
    """
    from ...agent import _slot_cfg as _agent_slot_cfg

    out: list[tuple[str, str]] = []
    # Pass 1: custom-provider slots → tag as "bomba".
    for slot in ("1", "2"):
        provider, api_key, _, _ = _agent_slot_cfg(slot)
        if provider == "custom" and api_key:
            out.append(("bomba", slot))
    # Pass 2: remaining configured slots in their natural numbering.
    for slot in ("1", "2"):
        provider, api_key, _, _ = _agent_slot_cfg(slot)
        if not api_key:
            continue
        if any(s == slot for _, s in out):
            continue
        out.append((f"slot{slot}", slot))
    return out


def _capabilities_summary() -> str:
    """Compact bullet list of what the bot can do, for the planner prompt."""
    lines = []
    for cap in CAPABILITIES:
        lines.append(f"- {cap.title}: {cap.description}")
    # Always advertise the agent tools as a generic catch-all so the
    # planner can use them in plans even if a specific feature isn't
    # in ``CAPABILITIES``.
    lines.append(
        "- Запуск shell-команды (`exec_bash`), чтение/запись файлов "
        "(`read_file`/`write_file`), клонирование git-репо "
        "(`clone_repo`) — через агентский режим."
    )
    return "\n".join(lines)


_PLANNER_SYSTEM_PROMPT = """Ты — планировщик действий для Telegram-бота.

Юзер пришлёт тебе ОДНО сообщение — это может быть новость, описание
задачи, или просто текст. Твоя задача: разбить это на пошаговый план,
который бот выполнит ВНУТРИ СЕБЯ (т.е. используя свои возможности
ниже). Каждый шаг — это команда боту, написанная в повелительном
наклонении («сделай X», «открой Y»), которую бот будет передавать
своему агенту.

Доступные возможности бота:
{capabilities}

ТРЕБОВАНИЯ К ОТВЕТУ:
1. Верни ТОЛЬКО валидный JSON, без markdown-форматирования, без
   преамбулы, без объяснений.
2. Структура: {{"name": "короткое имя слота, 1-3 слова, нижний регистр",
   "steps": ["шаг 1", "шаг 2", ...]}}
3. Имя слота — главное СЛОВО темы (если про гугл — то "гугл", если
   про скриншот — "скриншот", если про почту — "почта"). Не пиши
   полное предложение в name.
4. Шагов — от 1 до 12. Каждый шаг — одно конкретное действие.
5. Если задача требует возможности, которой у бота нет — всё равно
   попробуй разложить, но пометь такой шаг префиксом "[нет тулзы]".
6. Не повторяй сами шаги в name.
7. ОЧЕНЬ ВАЖНО: возвращай ИМЕННО валидный JSON, не оборачивай в
   ```json или другие фенсы.
"""


_USER_PROMPT_TEMPLATE = """Юзер прислал такой текст:

\"\"\"{news}\"\"\"

Сгенерируй JSON-план как описано в системном промпте."""


def _build_messages(news: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": _PLANNER_SYSTEM_PROMPT.format(
                capabilities=_capabilities_summary()
            ),
        },
        {"role": "user", "content": _USER_PROMPT_TEMPLATE.format(news=news[:6000])},
    ]


_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}", re.MULTILINE)


def _parse_plan_text(text: str, fallback_name: str) -> tuple[str, list[str]]:
    """Parse the LLM's response into ``(name, steps)``.

    Permissive: tries JSON first, then a bare ``{...}`` block fished
    out of the text, then a "split by lines and strip numbering"
    fallback. If everything fails, returns ``(fallback_name, [])``.
    """
    raw = (text or "").strip()
    if raw.startswith("```"):
        # strip optional ```json ... ``` fence
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    obj = None
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        m = _JSON_BLOCK_RE.search(raw)
        if m:
            try:
                obj = json.loads(m.group(0))
            except json.JSONDecodeError:
                obj = None

    if isinstance(obj, dict) and isinstance(obj.get("steps"), list):
        name = str(obj.get("name") or fallback_name).strip().lower()
        steps = [str(s).strip() for s in obj["steps"] if str(s).strip()]
        return name, steps

    # Plan-B: line-by-line, strip numbering ("1. ...", "1) ...", "- ...").
    steps: list[str] = []
    for line in raw.splitlines():
        clean = re.sub(r"^\s*[-*]?\s*\d+[.)]?\s*", "", line).strip()
        if clean:
            steps.append(clean)
    return fallback_name, steps


def _derive_fallback_name(news: str) -> str:
    """Short tag for the slot when JSON parsing didn't yield a name."""
    text = re.sub(r"https?://\S+", "", news)
    words = re.findall(r"[\wа-яА-ЯёЁ]+", text)
    if not words:
        return "без имени"
    return " ".join(words[:2]).lower()


async def plan_from_text(news: str) -> Plan:
    """Translate ``news`` into a :class:`Plan` via the brain chain.

    Raises:
        NoBrainAvailable: when every configured brain failed (or none
            are configured at all).
    """
    chain = _planner_chain()
    if not chain:
        raise NoBrainAvailable(
            "Нет настроенного мозга для планирования. Зайди в "
            "/setup → 🧠 Мозг и сконфигурируй хотя бы один слот."
        )

    from ...agent import (
        _build_client_for_slot,
        _candidate_models_for_slot,
        NoApiKeyError,
        AGENT_MAX_TOKENS,
    )

    messages = _build_messages(news)
    last_error: Exception | None = None
    for label, slot in chain:
        try:
            client = _build_client_for_slot(slot)
        except NoApiKeyError as exc:
            last_error = exc
            continue
        for model in _candidate_models_for_slot(slot):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=AGENT_MAX_TOKENS,
                )
            except Exception as exc:  # noqa: BLE001 — network / parse
                logger.warning(
                    "planner: brain=%s slot=%s model=%s failed: %s",
                    label, slot, model, exc,
                )
                last_error = exc
                continue
            choices = getattr(resp, "choices", None)
            if not choices:
                continue
            text = (choices[0].message.content or "").strip()
            name, steps = _parse_plan_text(text, _derive_fallback_name(news))
            return Plan(
                name=name,
                steps=steps,
                used_brain=label,
                used_model=model,
                raw=text,
            )
    raise NoBrainAvailable(
        f"Все настроенные мозги не ответили. Последняя ошибка: {last_error}"
    )
