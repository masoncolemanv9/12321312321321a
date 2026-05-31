"""Regression tests for the two Helpzavr/Pretty-text routing bugs that
Devin Review flagged on PR #1:

1. ``bot/addons/__init__.py:39`` — ``pretty_text`` router was registered
   before ``helpzavr`` and ``mailbox``. Its catch-all message handler
   (``F.text & ~F.text.startswith("/")``, ``StateFilter(None)``) would
   then consume the API keys / IMAP credentials a user was typing for
   the later addons. The fix moves ``helpzavr`` and ``mailbox`` ahead of
   ``pretty_text`` in ``build_addon_routers()``.

2. ``bot/addons/helpzavr/handlers.py:481`` — ``awaiting_key`` was stored
   as a *global* flag via ``addon_state.set_``. The next plain-text
   message from ANY chat would then be saved as the API key. The fix
   stores the flag per-chat (``addon_state.chat_set``), mirroring how
   ``bot/addons/mailbox/handlers.py`` already scopes its
   ``awaiting_field``.
"""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock


class AddonRouterOrderTest(unittest.TestCase):
    """Bug 1: build_addon_routers() must place ``helpzavr`` and
    ``mailbox`` before ``pretty_text`` so the latter's text catch-all
    does not steal API-key / credential input destined for the former.
    """

    def test_helpzavr_and_mailbox_are_registered_before_pretty_text(self):
        from bot.addons import build_addon_routers

        routers = build_addon_routers()
        names = [r.name for r in routers]

        self.assertIn("pretty_text_addon", names)
        self.assertIn("helpzavr_addon", names)
        self.assertIn("mailbox_addon", names)

        pretty_idx = names.index("pretty_text_addon")
        helpzavr_idx = names.index("helpzavr_addon")
        mailbox_idx = names.index("mailbox_addon")

        self.assertLess(
            helpzavr_idx,
            pretty_idx,
            f"helpzavr must come before pretty_text; got order {names}",
        )
        self.assertLess(
            mailbox_idx,
            pretty_idx,
            f"mailbox must come before pretty_text; got order {names}",
        )


class HelpzavrAwaitingKeyPerChatTest(unittest.TestCase):
    """Bug 2: helpzavr's ``awaiting_key`` flag must be per-chat, not
    global. ``chat_set`` writes under ``addons.helpzavr.by_chat.<id>``
    in lilush's settings dict; the previous ``set_`` wrote it at the
    addon root and was therefore shared across all chats.
    """

    def setUp(self):
        from bot.storage import storage as _storage

        self.storage = _storage
        # Snapshot the live settings dict and wipe just the helpzavr
        # sub-tree + owner_id so we have a clean slate per test, while
        # leaving the rest of lilush's settings untouched.
        self._settings = self.storage._settings()  # noqa: SLF001
        self._helpzavr_snapshot = (
            self._settings.get("addons", {}).get("helpzavr")
        )
        self._owner_snapshot = self._settings.get("owner_id")
        self._settings.setdefault("addons", {}).pop("helpzavr", None)
        self._settings.pop("owner_id", None)

    def tearDown(self):
        addons = self._settings.setdefault("addons", {})
        if self._helpzavr_snapshot is None:
            addons.pop("helpzavr", None)
        else:
            addons["helpzavr"] = self._helpzavr_snapshot
        if self._owner_snapshot is None:
            self._settings.pop("owner_id", None)
        else:
            self._settings["owner_id"] = self._owner_snapshot

    def test_awaiting_key_is_isolated_between_chats(self):
        """Two chats can be in ``awaiting_key`` for different providers
        simultaneously without leaking into each other's state.
        """
        from bot.addons import state as addon_state

        addon_state.chat_set("helpzavr", 111, "awaiting_key", "groq")
        addon_state.chat_set(
            "helpzavr", 222, "awaiting_key", "openrouter"
        )

        self.assertEqual(
            addon_state.chat_get("helpzavr", 111, "awaiting_key", ""),
            "groq",
        )
        self.assertEqual(
            addon_state.chat_get("helpzavr", 222, "awaiting_key", ""),
            "openrouter",
        )
        # Clearing one chat must not affect the other.
        addon_state.chat_set("helpzavr", 111, "awaiting_key", "")
        self.assertEqual(
            addon_state.chat_get("helpzavr", 111, "awaiting_key", ""),
            "",
        )
        self.assertEqual(
            addon_state.chat_get("helpzavr", 222, "awaiting_key", ""),
            "openrouter",
        )
        # Old global key must not be set as a side-effect of chat_set.
        self.assertFalse(
            addon_state.get("helpzavr", "awaiting_key", "")
        )

    def test_on_setkey_writes_per_chat_flag(self):
        """The ``hz:setkey:<provider>`` callback must scope the
        ``awaiting_key`` flag to the chat that initiated the request.
        """
        from bot.addons import state as addon_state
        from bot.addons.helpzavr.handlers import build_helpzavr_router

        # Grant admin so we don't get blocked by the can_admin check.
        # The first user to send /start in a fresh state becomes owner;
        # we shortcut by claiming the slot directly via storage.
        self.storage.set_owner_id(42)

        router = build_helpzavr_router()
        on_setkey = next(
            h.callback
            for h in router.observers["callback_query"].handlers
            if h.callback.__name__ == "on_setkey"
        )

        query = MagicMock()
        query.from_user.id = 42
        query.data = "hz:setkey:groq"
        query.message = MagicMock()
        query.message.chat.id = 999
        query.message.answer = AsyncMock()
        query.answer = AsyncMock()

        asyncio.run(on_setkey(query))

        # Per-chat flag set …
        self.assertEqual(
            addon_state.chat_get("helpzavr", 999, "awaiting_key", ""),
            "groq",
        )
        # … and the *global* flag is NOT set (this is the regression
        # we are guarding against).
        self.assertFalse(
            addon_state.get("helpzavr", "awaiting_key", "")
        )
        query.message.answer.assert_awaited()
        query.answer.assert_awaited()


if __name__ == "__main__":
    unittest.main()
