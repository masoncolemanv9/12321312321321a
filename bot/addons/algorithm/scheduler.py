"""Background scheduler — periodically runs saved algorithms whose
``interval_minutes`` is non-zero.

Wakes up every ``_TICK_SECONDS``, walks every chat that has at least
one slot stored, and triggers ``run_plan`` for any slot where:

    now - slot.last_run_at >= slot.interval_minutes * 60

The scheduler avoids overlap by skipping slots whose ``is_running``
flag is already set (and the executor sets / clears that flag
atomically around its inner loop).

Started by ``bot.main`` alongside the mailbox poller and the worker
pool.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import Callable

from aiogram import Bot

from . import state as algo_state
from . import executor

logger = logging.getLogger(__name__)

# How often the loop wakes up. 30 s keeps the loop responsive enough
# for sub-minute intervals (0.5 min = 30 s) without flooding storage
# with re-reads.
_TICK_SECONDS = 30


async def _send_progress(bot: Bot, chat_id: int, text: str) -> None:
    """Best-effort progress message — never crashes the scheduler."""
    try:
        await bot.send_message(chat_id, text)
    except Exception:  # noqa: BLE001
        logger.exception(
            "scheduler: failed to send progress to chat %s", chat_id
        )


def _due_now(slot, now: float) -> bool:
    """True iff ``slot`` should run on this tick."""
    if slot.is_empty:
        return False
    if not slot.has_interval:
        return False
    if slot.is_running:
        return False
    interval_s = slot.interval_minutes * 60.0
    return (now - slot.last_run_at) >= interval_s


async def scheduler_loop(
    bot: Bot,
    *,
    sleep: Callable[[float], "asyncio.Future[None]"] | None = None,
) -> None:
    """Run forever, ticking every ``_TICK_SECONDS``.

    ``sleep`` is injectable for tests so they don't have to wait real
    wall-clock seconds between ticks.
    """
    _sleep = sleep or asyncio.sleep
    logger.info("algorithm scheduler started (tick=%ds)", _TICK_SECONDS)
    while True:
        try:
            await _tick(bot)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.exception("scheduler tick raised — continuing")
        await _sleep(_TICK_SECONDS)


async def _tick(bot: Bot) -> None:
    """One pass over every stored chat's slots."""
    now = time.time()
    for chat_id in algo_state.list_chat_ids():
        for slot in algo_state.list_slots(chat_id):
            if not _due_now(slot, now):
                continue
            asyncio.create_task(
                _run_one(bot, chat_id, slot.index),
                name=f"algo:chat{chat_id}:slot{slot.index}",
            )


async def _run_one(bot: Bot, chat_id: int, slot_index: int) -> None:
    """Kick off one slot. Uses chat_id as user_id — sufficient for
    self-hosted single-owner case; multi-owner setups will see this
    attribute correctly because algorithms are already per-chat."""
    async def _status(text: str) -> None:
        await _send_progress(bot, chat_id, text)

    with contextlib.suppress(Exception):
        await executor.run_plan(
            chat_id=chat_id,
            user_id=chat_id,
            slot_index=slot_index,
            status=_status,
        )
