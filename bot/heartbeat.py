"""Heartbeat ping to OpenRouter — keeps the connection warm while idle.

When :func:`storage.get_heartbeat_enabled` is True, the
:func:`heartbeat_loop` background task sends a tiny chat-completion
request to OpenRouter every ``HEARTBEAT_INTERVAL_S`` seconds. The
response is discarded; only the act of pinging matters.

This is **independent** from :mod:`bot.main`'s ``keep-alive`` (HTTP
self-ping that prevents Render from spinning the container down). The
two coexist:

- ``keep-alive``  → keeps Render Free awake.
- ``heartbeat``   → keeps OpenRouter aware that this key is active and
  burns a tiny fraction of free-tier quota so the user can see «токены
  тратятся» in the OpenRouter dashboard.

The toggle is per-bot (state.json), so the owner can turn it on for one
service and off for the other 19 without touching env vars.
"""

from __future__ import annotations

import asyncio
import logging
import os

from openai import AsyncOpenAI
from openai._exceptions import APIError

from .config import APP_TITLE, HTTP_REFERER, OPENROUTER_BASE_URL
from .storage import storage
from .token_tracker import record as record_token_usage

logger = logging.getLogger(__name__)

# Default cadence. 10 minutes is short enough that the OpenRouter
# session stays warm but long enough that a free key burns negligible
# quota (~144 pings/day × ~5 tokens ≈ 720 tokens/day).
HEARTBEAT_INTERVAL_S = int(os.environ.get("HEARTBEAT_INTERVAL_S", "600"))
HEARTBEAT_MODEL = os.environ.get(
    "HEARTBEAT_MODEL", "nvidia/nemotron-3-super-120b-a12b:free"
)
HEARTBEAT_PROMPT = os.environ.get("HEARTBEAT_PROMPT", "ping")


async def _ping_once() -> bool:
    """Send a single heartbeat ping. Returns ``True`` on success."""
    api_key = storage.get_provider_key("openrouter")
    if not api_key:
        logger.debug("heartbeat: no OpenRouter key configured — skip")
        return False
    headers = {"HTTP-Referer": HTTP_REFERER, "X-Title": APP_TITLE}
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        default_headers=headers,
    )
    try:
        resp = await client.chat.completions.create(
            model=HEARTBEAT_MODEL,
            messages=[{"role": "user", "content": HEARTBEAT_PROMPT}],
            max_tokens=1,
            temperature=0.0,
        )
    except APIError as exc:
        logger.warning("heartbeat ping failed: %s", exc)
        return False
    except Exception as exc:  # noqa: BLE001 — never crash the bot from heartbeat
        logger.warning("heartbeat ping unexpected error: %s", exc)
        return False

    usage = getattr(resp, "usage", None)
    if usage is not None:
        pt = int(getattr(usage, "prompt_tokens", 0) or 0)
        ct = int(getattr(usage, "completion_tokens", 0) or 0)
        try:
            record_token_usage(
                provider="openrouter",
                model=HEARTBEAT_MODEL,
                prompt_tokens=pt,
                completion_tokens=ct,
                purpose="heartbeat",
            )
        except Exception:  # noqa: BLE001
            logger.exception("heartbeat token recording failed")
    return True


async def heartbeat_loop() -> None:
    """Forever loop: sleep, then ping if the toggle is on.

    Re-reads the toggle each cycle so flipping it in TG takes effect
    on the next tick (no restart needed).
    """
    logger.info(
        "heartbeat: loop started, interval=%ds, model=%s",
        HEARTBEAT_INTERVAL_S,
        HEARTBEAT_MODEL,
    )
    # First-tick delay so we don't ping immediately at boot.
    await asyncio.sleep(min(HEARTBEAT_INTERVAL_S, 30))
    while True:
        try:
            if storage.get_heartbeat_enabled():
                ok = await _ping_once()
                logger.info("heartbeat ping → %s", "ok" if ok else "skipped/failed")
        except Exception:  # noqa: BLE001 — keep the loop alive no matter what
            logger.exception("heartbeat loop iteration crashed")
        await asyncio.sleep(HEARTBEAT_INTERVAL_S)
