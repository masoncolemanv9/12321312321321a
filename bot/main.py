import asyncio
import contextlib
import logging
import random

# Tag stdout the moment Python starts so a deploy log can never look
# blank. Render's free tier sometimes kills the container mid-import
# (OOM), and previously the user just saw "Cause of failure could not
# be determined" — no Python output at all. With this print we at
# least know we *got* to Python before something blew up.
print("[boot] bot.main entering", flush=True)

import aiohttp
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
from aiohttp import web

from .addons import build_addon_routers
from .capabilities import ensure_recorded as ensure_capabilities_recorded
from .config import (
    ALLOWED_USER_IDS,
    BOT_PERSONA,
    BOT_TOKEN,
    KEEP_ALIVE_BIAS,
    KEEP_ALIVE_INTERVAL,
    KEEP_ALIVE_MAX_SECONDS,
    KEEP_ALIVE_MIN_SECONDS,
    KEEP_ALIVE_URL,
    MODE,
    PORT,
    WEBHOOK_PATH,
    WEBHOOK_URL,
    WORKER_POLL_INTERVAL_S,
)
from .groq_handlers import groq_router
from .handlers import router
from .heartbeat import heartbeat_loop
from .jobs import get_default_queue
from .media_ui import media_router
from .persona import get_persona, known_keys
from .storage import storage
from .wizard import wizard_router
from .workers import (
    AnalyzerWorker,
    DownloaderWorker,
    PublisherWorker,
    SeoWorker,
    Worker,
    build_editor_worker,
)

print("[boot] all imports OK", flush=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _build_bot() -> Bot:
    return Bot(BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))


# Commands shown by Telegram when the user taps the menu button (≡)
# next to the input field. ``setMyCommands`` is set on startup so the
# popup matches whatever the user actually sees in the bot today.
# ``⚙️ Настройки`` lives at the bottom per user request.
_BOT_COMMANDS = [
    ("start", "🏠 Главное меню"),
    ("chat", "💬 Чат с LLM"),
    ("helpzavr", "🤖 Helpzavr"),
    ("pretty", "✨ Красивый текст"),
    ("mailbox", "📬 Почта"),
    ("media", "🎬 Генерация фото/видео"),
    ("github", "🐙 GitHub проекты"),
    ("work", "🖥 Терминал (пакет команд)"),
    ("settings", "⚙️ Настройки"),
]


async def _register_bot_commands(bot: Bot) -> None:
    """Publish the menu-button command list to Telegram.

    Best-effort: failures are logged but don't block startup — the bot
    still works without the menu (slash commands keep functioning;
    only the popup is missing). Defaults to clearing any previous
    list so renames / removals stick.
    """
    from aiogram.types import BotCommand, MenuButtonCommands

    try:
        commands = [BotCommand(command=name, description=desc) for name, desc in _BOT_COMMANDS]
        await bot.set_my_commands(commands)
        # Force the round (≡) menu button instead of the default web-app
        # button or no-button-at-all state. Without this the user has
        # to hold the "/" key to see commands — the round menu icon
        # next to the paperclip is what they actually want.
        await bot.set_chat_menu_button(menu_button=MenuButtonCommands())
        logger.info("registered %d bot commands with Telegram", len(commands))
    except Exception:  # noqa: BLE001
        logger.exception("failed to register bot commands — menu button will be empty")


def _build_dispatcher() -> Dispatcher:
    # MemoryStorage is fine — wizard FSM only holds short-lived "awaiting input"
    # states; if the bot restarts mid-onboarding the user just taps the button
    # again. No need for a Redis/SQLite FSM backend.
    dp = Dispatcher(storage=MemoryStorage())
    # Wizard router goes first so its /start, /setup and callback queries win
    # over the generic handlers in `router`. The media router sits
    # between them so its FSM-stated handlers (photo prompt capture,
    # slot-edit text input) match before the LLM chat catch-all.
    dp.include_router(wizard_router)
    # Groq overrides go right after the wizard so voice (no other
    # competitor) and photo-with-caption (would otherwise hit
    # photo_chooser first) get a chance to be transcribed / analysed
    # by Groq before the photo chooser asks the user what to do with
    # the image. Both handlers are state-less (`StateFilter(None)`) so
    # the wizard and addon FSMs are untouched.
    dp.include_router(groq_router)
    dp.include_router(media_router)
    # Addon routers (Helpzavr / Красивый текст / Проверка почты) sit
    # between the media and generic catch-all routers so their FSM-aware
    # message handlers run before the LLM chat fallback.
    for addon_router in build_addon_routers():
        dp.include_router(addon_router)
    dp.include_router(router)
    return dp


async def _health(_: web.Request) -> web.Response:
    return web.Response(text="ok")


def _next_keepalive_delay() -> float:
    """Compute the next ping delay in seconds.

    Three modes:
      * ``KEEP_ALIVE_INTERVAL`` is a positive int → fixed interval.
      * ``KEEP_ALIVE_INTERVAL`` is ``None`` (default, env var unset) →
        random delay in ``[MIN, MAX]`` with ``BIAS`` probability of
        landing in the upper half (closer to ``MAX``). This makes the
        ping pattern look human-ish rather than a fixed cron tick.
      * ``KEEP_ALIVE_INTERVAL == 0`` → disabled (caller short-circuits).
    """
    if KEEP_ALIVE_INTERVAL is not None and KEEP_ALIVE_INTERVAL > 0:
        return float(KEEP_ALIVE_INTERVAL)
    midpoint = (KEEP_ALIVE_MIN_SECONDS + KEEP_ALIVE_MAX_SECONDS) / 2
    if random.random() < KEEP_ALIVE_BIAS:
        return random.uniform(midpoint, KEEP_ALIVE_MAX_SECONDS)
    return random.uniform(KEEP_ALIVE_MIN_SECONDS, midpoint)


async def _keep_alive_loop() -> None:
    """Periodically GET our own ``/healthz`` to keep Render Free awake.

    Render counts only *incoming* HTTP traffic toward the 15-minute idle
    timer — the bot's outgoing Telegram polling doesn't qualify. Hitting
    our own public URL through the platform's load balancer DOES count.

    Disabled when no URL was resolved (local dev, VPS) or when
    ``KEEP_ALIVE_INTERVAL`` is explicitly set to ``0``.
    """
    if not KEEP_ALIVE_URL or KEEP_ALIVE_INTERVAL == 0:
        return
    target = f"{KEEP_ALIVE_URL}/healthz"
    if KEEP_ALIVE_INTERVAL is not None and KEEP_ALIVE_INTERVAL > 0:
        logger.info(
            "keep-alive: pinging %s every %ss (fixed)", target, KEEP_ALIVE_INTERVAL
        )
    else:
        logger.info(
            "keep-alive: pinging %s every %d-%ds (%.0f%% bias toward upper end)",
            target,
            KEEP_ALIVE_MIN_SECONDS,
            KEEP_ALIVE_MAX_SECONDS,
            KEEP_ALIVE_BIAS * 100,
        )
    timeout = aiohttp.ClientTimeout(total=10)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # First ping after a short grace period so the health server has
        # time to bind and the platform DNS to resolve our URL.
        await asyncio.sleep(min(_next_keepalive_delay(), 30))
        while True:
            try:
                async with session.get(target) as resp:
                    if resp.status >= 400:
                        logger.warning(
                            "keep-alive ping returned HTTP %s", resp.status
                        )
            except Exception as exc:  # noqa: BLE001 — log everything, retry forever
                logger.warning("keep-alive ping failed: %s", exc)
            await asyncio.sleep(_next_keepalive_delay())


def _build_workers() -> list[Worker]:
    """Instantiate one worker per pipeline stage, all bound to the default queue.

    Concrete worker types are kept as a flat list so each gets exactly one
    polling task. If we later want N concurrent editors (the CPU-heavy
    stage), we just add ``EditorWorker(queue)`` multiple times.
    """
    queue = get_default_queue()
    poll = WORKER_POLL_INTERVAL_S
    return [
        DownloaderWorker(queue, poll_interval_s=poll),
        AnalyzerWorker(queue, poll_interval_s=poll),
        # ``build_editor_worker`` picks v1 (default) or v2/v2.1/v6 based on
        # the ``EDITOR_VERSION`` env var (or the Telegram «🎞 Монтажёр»
        # screen at runtime). Default keeps v1 behaviour exactly as before.
        build_editor_worker(queue, poll_interval_s=poll),
        SeoWorker(queue, poll_interval_s=poll),
        PublisherWorker(queue, poll_interval_s=poll),
    ]


async def _run_polling() -> None:
    """Long-polling mode + a tiny aiohttp server for cloud health checks.

    Render free, Railway, Fly etc. usually require a process to bind a port
    so they know the deploy is healthy. A 200-OK on ``/`` and ``/healthz`` is
    enough; we keep the same paths the webhook mode exposes.

    The health server is bound FIRST (before any Telegram I/O or worker
    boot) so the platform marks the deploy as live as soon as possible.
    Earlier we set up the bot before binding the port — if any of the
    network calls (``delete_webhook``, ``set_my_commands``) hung or
    OOM-killed the process, Render saw no logs and aborted the deploy
    with the unhelpful "Cause of failure could not be determined". Now
    even a later crash leaves application logs behind.
    """
    health_app = web.Application()
    health_app.router.add_get("/", _health)
    health_app.router.add_get("/healthz", _health)
    runner = web.AppRunner(health_app)
    await runner.setup()
    site = web.TCPSite(runner, host="0.0.0.0", port=PORT)
    await site.start()
    logger.info("health server listening on 0.0.0.0:%s", PORT)

    bot = _build_bot()
    dp = _build_dispatcher()
    try:
        await bot.delete_webhook(drop_pending_updates=True)
    except Exception:  # noqa: BLE001 — log and keep going
        logger.exception("delete_webhook failed; continuing")
    try:
        await _register_bot_commands(bot)
    except Exception:  # noqa: BLE001
        logger.exception("set_my_commands failed; continuing")
    try:
        ensure_capabilities_recorded()
    except Exception:  # noqa: BLE001
        logger.exception("ensure_capabilities_recorded failed; continuing")

    queue = get_default_queue()
    await queue.init()
    workers = _build_workers()
    worker_tasks = [asyncio.create_task(w.run(), name=f"worker:{w.kind}") for w in workers]

    keepalive_task = asyncio.create_task(_keep_alive_loop())
    heartbeat_task = asyncio.create_task(heartbeat_loop(), name="heartbeat")

    # Mailbox addon background poller — pushes new emails to the chat
    # configured in /Проверка почты → Уведомления when any toggle is on.
    addon_tasks: list[asyncio.Task[None]] = []
    try:
        from .addons.mailbox.handlers import poller_loop as _mailbox_poller_loop

        def _resolve_chat_id():
            from .addons import state as _addon_state
            return _addon_state.get("mailbox", "notify_chat_id", None)

        addon_tasks.append(
            asyncio.create_task(
                _mailbox_poller_loop(bot, get_chat_id=_resolve_chat_id),
                name="mailbox-poller",
            )
        )
    except Exception:  # noqa: BLE001
        logger.exception("failed to start mailbox poller")

    # Algorithm addon scheduler — runs saved algorithms whose
    # ``interval_minutes > 0`` on their cadence. Lives next to the
    # mailbox poller so it shares the same task-lifecycle handling.
    try:
        from .addons.algorithm.scheduler import scheduler_loop as _algo_scheduler

        addon_tasks.append(
            asyncio.create_task(
                _algo_scheduler(bot), name="algorithm-scheduler",
            )
        )
    except Exception:  # noqa: BLE001
        logger.exception("failed to start algorithm scheduler")

    logger.info("starting polling with %d workers", len(workers))
    try:
        await dp.start_polling(bot)
    except Exception:  # noqa: BLE001
        # Polling crashed (most commonly: TelegramUnauthorizedError when
        # the BOT_TOKEN env var is wrong). Log the full trace and keep
        # the health server alive so Render keeps the deploy "live" and
        # the user can fix BOT_TOKEN in env vars without losing the
        # entire service. Without this the container exits, Render
        # restarts it in a loop, and runtime logs scroll past faster
        # than the user can copy them.
        logger.exception(
            "dp.start_polling crashed — check BOT_TOKEN in Render env "
            "vars. Health server stays up so you can patch the token "
            "and redeploy."
        )
        # Block forever so the container doesn't exit. Render will keep
        # /healthz green; the user reads the trace above and fixes the
        # token.
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.Event().wait()
    finally:
        for w in workers:
            w.stop()
        keepalive_task.cancel()
        heartbeat_task.cancel()
        for task in addon_tasks:
            task.cancel()
        for task in (*worker_tasks, keepalive_task, heartbeat_task, *addon_tasks):
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        await runner.cleanup()
        await bot.session.close()


def _run_webhook() -> None:
    if not WEBHOOK_URL:
        raise RuntimeError(
            "PUBLIC_URL is not set; cannot run in webhook mode. "
            "Set BOT_MODE=polling or provide PUBLIC_URL."
        )

    bot = _build_bot()
    dp = _build_dispatcher()
    workers: list[Worker] = []
    worker_tasks: list[asyncio.Task[None]] = []
    bg_tasks: list[asyncio.Task[None]] = []

    async def _on_startup(app: web.Application) -> None:
        logger.info("setting webhook to %s", WEBHOOK_URL)
        await bot.set_webhook(WEBHOOK_URL, drop_pending_updates=True)
        await _register_bot_commands(bot)
        ensure_capabilities_recorded()
        queue = get_default_queue()
        await queue.init()
        workers.extend(_build_workers())
        for w in workers:
            worker_tasks.append(asyncio.create_task(w.run(), name=f"worker:{w.kind}"))
        bg_tasks.append(asyncio.create_task(heartbeat_loop(), name="heartbeat"))
        # Render Free counts only INCOMING HTTP toward its 15-min idle
        # timer. Webhook traffic from Telegram qualifies, but a chatty
        # bot can still go quiet for long stretches. Run the same
        # self-ping loop the polling branch uses so Render doesn't spin
        # the service down between conversations.
        bg_tasks.append(asyncio.create_task(_keep_alive_loop(), name="keepalive"))
        # Algorithm scheduler also runs in webhook mode so periodic
        # algorithms fire regardless of which transport the deploy uses.
        try:
            from .addons.algorithm.scheduler import (
                scheduler_loop as _algo_scheduler,
            )

            bg_tasks.append(
                asyncio.create_task(
                    _algo_scheduler(bot), name="algorithm-scheduler",
                )
            )
        except Exception:  # noqa: BLE001
            logger.exception("failed to start algorithm scheduler")
        logger.info(
            "started %d workers + heartbeat + keep-alive + algorithm",
            len(workers),
        )

    async def _on_cleanup(app: web.Application) -> None:
        logger.info("removing webhook")
        for w in workers:
            w.stop()
        for task in bg_tasks:
            task.cancel()
        for task in (*worker_tasks, *bg_tasks):
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        try:
            await bot.delete_webhook()
        finally:
            await bot.session.close()

    app = web.Application()
    app.router.add_get("/", _health)
    app.router.add_get("/healthz", _health)
    SimpleRequestHandler(dp, bot).register(app, path=WEBHOOK_PATH)
    setup_application(app, dp, bot=bot)
    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)
    logger.info("starting webhook server on 0.0.0.0:%s, path %s", PORT, WEBHOOK_PATH)
    web.run_app(app, host="0.0.0.0", port=PORT)


async def _run_dormant() -> None:
    """Health-only server for services whose BOT_TOKEN hasn't been set yet.

    Render Free requires the process to bind a port (health check) within
    a few minutes or marks the deploy failed. So even without a token we
    spin up an aiohttp server on ``/`` and ``/healthz``, log a warning,
    and idle forever. The user just edits ``BOT_TOKEN`` in the Render env
    vars and the service auto-restarts into normal polling mode.

    Self-ping intentionally stays disabled in dormant mode — there's no
    bot to keep alive, the service may safely sleep.
    """
    logger.warning(
        "BOT_TOKEN is empty — entering DORMANT mode. Health server only, "
        "no Telegram polling. Set BOT_TOKEN in Render env vars to "
        "activate this bot."
    )
    health_app = web.Application()
    health_app.router.add_get("/", _health)
    health_app.router.add_get("/healthz", _health)
    runner = web.AppRunner(health_app)
    await runner.setup()
    site = web.TCPSite(runner, host="0.0.0.0", port=PORT)
    await site.start()
    logger.info("dormant health server listening on 0.0.0.0:%s", PORT)
    try:
        # Block forever. The process exits on signal (Render redeploy).
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()


def main() -> None:
    if BOT_PERSONA not in known_keys():
        logger.warning(
            "BOT_PERSONA=%r is not in the known roster %s — falling back to 'boss'. "
            "Set BOT_PERSONA correctly in Render env vars for this service.",
            BOT_PERSONA,
            known_keys(),
        )
    persona = get_persona()
    logger.info(
        "boot: persona=%s (%s, %s/%s)",
        persona.key,
        persona.display_name,
        persona.department,
        persona.rank,
    )

    if not BOT_TOKEN:
        # Service deployed without a token yet — run a stub health server
        # so Render keeps the deploy alive, and let the user add a token
        # in the dashboard later.
        asyncio.run(_run_dormant())
        return

    owner = storage.get_owner_id()
    if owner is None and not ALLOWED_USER_IDS:
        logger.info(
            "No owner claimed yet and ALLOWED_USER_IDS is empty. "
            "First user to send /start in Telegram becomes the owner."
        )
    elif owner is not None:
        logger.info("Owner already claimed: telegram id %s", owner)

    # Webhook mode requires PUBLIC_URL. If the user picked webhook but didn't
    # set the URL (typical right after a one-click cloud deploy), silently
    # fall back to polling so the bot still comes up.
    if MODE == "webhook" and not WEBHOOK_URL:
        logger.warning(
            "BOT_MODE=webhook but PUBLIC_URL is empty — falling back to polling. "
            "Set PUBLIC_URL=https://<your-host> to switch back."
        )
        asyncio.run(_run_polling())
    elif MODE == "polling":
        asyncio.run(_run_polling())
    else:
        _run_webhook()


if __name__ == "__main__":
    main()
