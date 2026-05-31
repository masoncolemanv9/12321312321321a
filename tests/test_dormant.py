"""Dormant-mode tests: services with no BOT_TOKEN should boot, expose
``/healthz``, and not crash when imported.
"""

from __future__ import annotations

import importlib
import sys

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer


@pytest.mark.asyncio
async def test_config_loads_without_bot_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Removing BOT_TOKEN from env must not crash bot.config on import."""
    monkeypatch.delenv("BOT_TOKEN", raising=False)
    # Force re-import of bot.config to pick up the missing env var.
    if "bot.config" in sys.modules:
        del sys.modules["bot.config"]
    config = importlib.import_module("bot.config")
    assert config.BOT_TOKEN == ""
    assert config.WEBHOOK_PATH == "/tg/dormant"
    assert config.WEBHOOK_URL == ""


@pytest.mark.asyncio
async def test_dormant_health_endpoint_responds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dormant service must still answer /healthz so Render keeps it alive."""
    monkeypatch.delenv("BOT_TOKEN", raising=False)
    for module_name in ("bot.config", "bot.main"):
        sys.modules.pop(module_name, None)
    main = importlib.import_module("bot.main")

    # Build a minimal app the same way _run_dormant would, but without
    # blocking on an Event — we want to assert the routes, not the loop.
    app = web.Application()
    app.router.add_get("/", main._health)
    app.router.add_get("/healthz", main._health)

    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/healthz")
        assert resp.status == 200
        body = await resp.text()
        assert body == "ok"
        resp = await client.get("/")
        assert resp.status == 200


def test_main_runs_dormant_when_token_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """main() should hand off to _run_dormant when BOT_TOKEN is unset."""
    monkeypatch.delenv("BOT_TOKEN", raising=False)
    monkeypatch.setenv("PORT", "0")
    for module_name in ("bot.config", "bot.main"):
        sys.modules.pop(module_name, None)
    main = importlib.import_module("bot.main")

    called = {"dormant": False, "polling": False, "webhook": False}

    async def fake_dormant() -> None:
        called["dormant"] = True

    async def fake_polling() -> None:
        called["polling"] = True

    def fake_webhook() -> None:
        called["webhook"] = True

    monkeypatch.setattr(main, "_run_dormant", fake_dormant)
    monkeypatch.setattr(main, "_run_polling", fake_polling)
    monkeypatch.setattr(main, "_run_webhook", fake_webhook)

    main.main()

    assert called["dormant"] is True
    assert called["polling"] is False
    assert called["webhook"] is False
