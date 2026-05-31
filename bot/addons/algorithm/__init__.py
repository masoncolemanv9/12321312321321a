"""Algorithm addon — saved multi-step sequences with optional periodic
auto-run.

Each chat has up to 10 named slots. A slot can be filled either:

* **Manually** — the user types out the step-by-step plan themselves.
* **AI-planned** — the user pastes a piece of free text (e.g. a news
  article describing some procedure) and the bot asks one of its LLM
  brains to translate that into an ordered list of bot-internal steps
  using the capabilities listed in :mod:`bot.capabilities`.

The brain used for AI planning falls back through:

1. **«Другое»** — the custom OpenAI-compatible endpoint the user
   wires up via Settings → 🧠 Мозг → «Другое». Top priority on user
   request.
2. **Brain slot 1** — the first slot of the Авто-мозг pair.
3. **Brain slot 2** — the second slot.

Slots store an ``interval_minutes`` (default ``0`` — disabled). When
non-zero, a background scheduler task triggers the slot's executor
every N minutes. The executor walks the saved steps strictly
sequentially, delegating each step to the existing ``run_agent`` so
no new tool surface is added.
"""

from __future__ import annotations

__all__ = [
    "build_algorithm_router",
    "scheduler_loop",
]


def build_algorithm_router():
    """Return the addon's aiogram router (lazy import to avoid pulling
    aiogram on test collection where it's already a hard dep)."""
    from .handlers import build_algorithm_router as _build

    return _build()


async def scheduler_loop(bot):
    """Background task that periodically runs slots whose interval has
    elapsed. Imported and started by :mod:`bot.main`."""
    from .scheduler import scheduler_loop as _loop

    await _loop(bot)
