"""Lightweight LLM-token usage tracker.

Every LLM call (chat completion or heartbeat ping) records prompt/
completion token counts into ``state.json`` via ``storage.log_llm_call``.
This module exposes:

- :func:`record` — call from inside the agent loop after each request.
- :func:`format_token_stats` — pretty stats for the ``/tokens`` command.
- :func:`pricing_for` — best-effort USD-per-million for known models.

It is intentionally simple — no Postgres, no Redis, no per-user
sharding. Goal is to give the owner a rough idea of «куда уходят
токены» without taking on infrastructure debt. CodeBurn / litellm /
similar tools can replace this later if the farm grows.
"""

from __future__ import annotations

import time

from .storage import storage

# Rough USD-per-million-tokens (input, output) for the most common
# OpenRouter routes. Free models are explicitly 0/0. Values are
# representative as of 2026-05 and may drift — they're displayed as
# «оценка», not as authoritative billing.
_PRICING_PER_MTOK_USD: dict[str, tuple[float, float]] = {
    "anthropic/claude-opus-4.7": (15.0, 75.0),
    "anthropic/claude-opus-4.6": (15.0, 75.0),
    "anthropic/claude-opus-4.5": (15.0, 75.0),
    "anthropic/claude-sonnet-4.5": (3.0, 15.0),
    "anthropic/claude-haiku-4.5": (1.0, 5.0),
    "openai/gpt-5": (10.0, 30.0),
    "openai/gpt-4o": (2.5, 10.0),
    # Kimi (Moonshot) — typical OpenRouter pricing as of Apr 2026.
    "moonshotai/kimi-k2": (0.5, 2.0),
    "moonshotai/kimi-k1.5": (0.5, 2.0),
    # Free-tier models — no cost.
    "nvidia/nemotron-3-super-120b-a12b:free": (0.0, 0.0),
    "openai/gpt-oss-120b:free": (0.0, 0.0),
    "openai/gpt-oss-20b:free": (0.0, 0.0),
    "qwen/qwen3-coder:free": (0.0, 0.0),
    "minimax/minimax-m2.5:free": (0.0, 0.0),
}


def pricing_for(model: str) -> tuple[float, float]:
    """Return (input_per_mtok, output_per_mtok) USD for ``model``.

    Falls back to (0, 0) for unknown models — we'd rather under-report
    than mislead. The actual OpenRouter bill is authoritative.
    """
    model = model.strip().lower()
    if model in _PRICING_PER_MTOK_USD:
        return _PRICING_PER_MTOK_USD[model]
    # ``:free`` suffix is always zero, regardless of base name.
    if model.endswith(":free"):
        return (0.0, 0.0)
    return (0.0, 0.0)


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate the USD cost of a single call. Returns 0 for unknown models."""
    in_rate, out_rate = pricing_for(model)
    return (prompt_tokens * in_rate + completion_tokens * out_rate) / 1_000_000


def record(
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    purpose: str = "chat",
) -> None:
    """Persist one LLM call's token usage into state.json."""
    storage.log_llm_call(provider, model, prompt_tokens, completion_tokens, purpose)


def _summarise(entries: list[dict]) -> dict:
    """Aggregate a list of token-log entries into rolled-up totals."""
    by_model: dict[str, dict] = {}
    by_purpose: dict[str, int] = {}
    total_prompt = 0
    total_completion = 0
    total_cost = 0.0
    for e in entries:
        model = e.get("model", "unknown")
        pt = int(e.get("prompt_tokens", 0))
        ct = int(e.get("completion_tokens", 0))
        purpose = e.get("purpose", "chat")
        cost = estimate_cost_usd(model, pt, ct)
        total_prompt += pt
        total_completion += ct
        total_cost += cost
        m = by_model.setdefault(
            model, {"prompt": 0, "completion": 0, "calls": 0, "cost": 0.0}
        )
        m["prompt"] += pt
        m["completion"] += ct
        m["calls"] += 1
        m["cost"] += cost
        by_purpose[purpose] = by_purpose.get(purpose, 0) + pt + ct
    return {
        "calls": len(entries),
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "total_tokens": total_prompt + total_completion,
        "cost_usd": total_cost,
        "by_model": by_model,
        "by_purpose": by_purpose,
    }


def _format_section(title: str, summary: dict) -> str:
    if summary["calls"] == 0:
        return f"<b>{title}</b>: пусто"
    lines = [
        f"<b>{title}</b> — {summary['calls']} вызовов, "
        f"{summary['total_tokens']} токенов "
        f"(~${summary['cost_usd']:.4f})"
    ]
    for model, m in sorted(
        summary["by_model"].items(), key=lambda kv: kv[1]["prompt"] + kv[1]["completion"], reverse=True
    )[:5]:
        lines.append(
            f"  • <code>{model}</code> — {m['calls']}× • "
            f"{m['prompt'] + m['completion']} t • ~${m['cost']:.4f}"
        )
    if summary["by_purpose"]:
        purposes = ", ".join(
            f"{k}={v}" for k, v in sorted(summary["by_purpose"].items(), key=lambda kv: kv[1], reverse=True)
        )
        lines.append(f"  <i>purpose: {purposes}</i>")
    return "\n".join(lines)


def format_token_stats() -> str:
    """Build the multi-section text shown by ``/tokens``."""
    now = int(time.time())
    day_ago = now - 86_400
    week_ago = now - 7 * 86_400

    all_log = storage.get_token_log()
    today = [e for e in all_log if e.get("ts", 0) >= day_ago]
    week = [e for e in all_log if e.get("ts", 0) >= week_ago]

    today_sum = _summarise(today)
    week_sum = _summarise(week)
    total_sum = _summarise(all_log)

    return (
        "<b>📊 Расход токенов</b> <i>(оценочная стоимость по таблице OpenRouter)</i>\n\n"
        + _format_section("Сегодня", today_sum)
        + "\n\n"
        + _format_section("За неделю", week_sum)
        + "\n\n"
        + _format_section("Всего", total_sum)
        + "\n\n<i>В журнале хранится последних 5000 вызовов. Стоимость "
        "приблизительная — фактические цифры в кабинете OpenRouter.</i>"
    )
