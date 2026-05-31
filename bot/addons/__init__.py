"""Bolt-on features that live alongside lilush without touching its core.

Each subpackage exposes ``build_router()`` returning an aiogram ``Router``
that lilush's dispatcher includes. State is persisted via the helpers in
:mod:`bot.addons.state`, which piggybacks on lilush's own
``storage._settings()`` dict — no extra JSON files, no schema changes.

The three addons:

* :mod:`bot.addons.pretty_text` — Markdown→Telegram-HTML formatter with a
  per-chat ON/OFF toggle.
* :mod:`bot.addons.helpzavr` — screenshot-helper from ``tg_bot+(13).zip``:
  vision + thinking + OCR refinement + arrow callout overlay.
* :mod:`bot.addons.mailbox` — IMAP (Yandex/Mail.ru/Gmail) poller with task
  card delivery, originally ``freelance-mailbox-mcp``.
"""

from __future__ import annotations

__all__ = ["build_addon_routers"]


def build_addon_routers():
    """Return the list of addon routers to include in lilush's dispatcher.

    Imports happen lazily so that an addon failing to import (e.g.
    missing system package) only breaks that addon, not the whole bot.

    Order matters: ``photo_chooser`` must be first because it owns the
    single ``F.photo`` entry that asks the user which pipeline to run.
    Helpzavr and media_ui no longer have their own photo handlers, so
    nothing else competes for that match.

    ``helpzavr``, ``mailbox`` and ``algorithm`` are included BEFORE
    ``pretty_text`` because the latter installs a catch-all
    ``F.text & ~F.text.startswith("/")`` message handler that, when
    enabled for the chat, would otherwise consume API keys / IMAP
    credentials / algorithm-input text the user is typing for those
    addons. (Note: algorithm's own message handlers are FSM-gated so
    they would survive even if pretty_text ran first — but keeping a
    consistent "addons with awaited input go before pretty_text" rule
    makes the next bug of this family physically impossible.)
    """
    routers = []
    from .photo_chooser.handlers import build_photo_chooser_router
    routers.append(build_photo_chooser_router())
    from .thinking_style.handlers import build_thinking_style_router
    routers.append(build_thinking_style_router())
    try:
        from .helpzavr.handlers import build_helpzavr_router
        routers.append(build_helpzavr_router())
    except Exception as e:  # noqa: BLE001
        import logging
        logging.getLogger(__name__).warning(
            "helpzavr addon disabled: %s", e
        )
    try:
        from .mailbox.handlers import build_mailbox_router
        routers.append(build_mailbox_router())
    except Exception as e:  # noqa: BLE001
        import logging
        logging.getLogger(__name__).warning(
            "mailbox addon disabled: %s", e
        )
    try:
        from .algorithm import build_algorithm_router
        routers.append(build_algorithm_router())
    except Exception as e:  # noqa: BLE001
        import logging
        logging.getLogger(__name__).warning(
            "algorithm addon disabled: %s", e
        )
    from .pretty_text.handlers import build_pretty_text_router
    routers.append(build_pretty_text_router())
    from .media_toggle.handlers import build_media_toggle_router
    routers.append(build_media_toggle_router())
    from .memory.handlers import build_memory_router
    routers.append(build_memory_router())
    from .ram_guard.handlers import build_ram_router
    routers.append(build_ram_router())
    try:
        from .editor_agent.handlers import build_editor_agent_router
        routers.append(build_editor_agent_router())
    except Exception as e:  # noqa: BLE001
        import logging
        logging.getLogger(__name__).warning(
            "editor_agent addon disabled: %s", e
        )
    return routers
