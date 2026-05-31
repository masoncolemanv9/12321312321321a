"""Regression tests for the Helpzavr key-validation + Brain 2 fallback
fix.

Background bug
==============

A user pasted the **model name** ``meta-llama/llama-4-scout-17b-16e-instruct``
into the «🔑 Groq API key» prompt by accident. The handler accepted it
without any validation and wrote it into ``addons.helpzavr.groq_api_key``,
which is the field whose value is sent verbatim as the Bearer token to
``api.groq.com``. Every subsequent call therefore failed with::

    HTTP 401 Unauthorized — Invalid API Key

The state.json field was indistinguishable from a real key from the
addon's point of view, so the bot stayed broken until manually fixed.

Fix
===

Three guard rails were added to :mod:`bot.addons.helpzavr.handlers`:

* :func:`_looks_like_groq_key` — Groq keys are ``gsk_<base62>``; model
  names contain ``/`` and never start with ``gsk_``.
* :func:`_looks_like_openrouter_key` — same idea, ``sk-or-…``.
* :func:`get_groq_key` now ignores a stored value that doesn't look
  like a key, falls back to ``Brain 2`` if it's pointed at Groq, then
  the env var. So a previously-confused state.json no longer strands
  the user with permanent 401s — the addon transparently picks up the
  ``gsk_…`` the user already configured in Brain 2.
* :func:`build_helpzavr_router.on_key_input` rejects non-key text and
  asks the user to paste the real ``gsk_…`` / ``sk-or-…`` instead.
"""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch


# ---- pure helpers --------------------------------------------------------


class LooksLikeGroqKeyTest(unittest.TestCase):
    def test_real_groq_key_is_accepted(self):
        from bot.addons.helpzavr.handlers import _looks_like_groq_key

        self.assertTrue(
            _looks_like_groq_key(
                "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"
            )
        )

    def test_model_name_is_rejected(self):
        from bot.addons.helpzavr.handlers import _looks_like_groq_key

        for value in [
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "openai/gpt-4o-mini",
            "google/gemma-2-9b-it",
            "nvidia/nemotron-3-super-120b-a12b:free",
        ]:
            self.assertFalse(
                _looks_like_groq_key(value),
                f"model name {value!r} must not look like a Groq key",
            )

    def test_empty_and_whitespace_rejected(self):
        from bot.addons.helpzavr.handlers import _looks_like_groq_key

        self.assertFalse(_looks_like_groq_key(""))
        self.assertFalse(_looks_like_groq_key("   "))
        self.assertFalse(_looks_like_groq_key(None))  # type: ignore[arg-type]

    def test_unrelated_string_rejected(self):
        from bot.addons.helpzavr.handlers import _looks_like_groq_key

        # Wrong prefix even if "long enough".
        self.assertFalse(_looks_like_groq_key("sk-or-v1-aaaaaaaaaaaaaaaa"))
        # Right prefix but too short.
        self.assertFalse(_looks_like_groq_key("gsk_short"))


class LooksLikeOpenRouterKeyTest(unittest.TestCase):
    def test_real_openrouter_key_is_accepted(self):
        from bot.addons.helpzavr.handlers import _looks_like_openrouter_key

        self.assertTrue(
            _looks_like_openrouter_key(
                "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"
            )
        )

    def test_groq_key_does_not_look_like_openrouter(self):
        from bot.addons.helpzavr.handlers import _looks_like_openrouter_key

        self.assertFalse(
            _looks_like_openrouter_key(
                "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"
            )
        )

    def test_model_name_rejected(self):
        from bot.addons.helpzavr.handlers import _looks_like_openrouter_key

        self.assertFalse(
            _looks_like_openrouter_key(
                "nvidia/nemotron-3-super-120b-a12b:free"
            )
        )


# ---- get_groq_key fallback chain -----------------------------------------


class _StorageSnapshot:
    """Helper that snapshots / restores the live lilush state dict so a
    test can mutate ``brain_slot2`` and ``addons.helpzavr`` without
    leaking into sibling tests.

    The snapshot uses :func:`copy.deepcopy` because ``addon_state.set_``
    mutates the same sub-dict the snapshot held a reference to — without
    a copy, restoring would put the test's own modifications back into
    place instead of the pristine starting state.
    """

    def __init__(self):
        import copy

        # Pull storage off the long-lived helpzavr handlers module —
        # same singleton ``addon_state.storage`` was closed over when
        # ``bot.addons.helpzavr.handlers`` first loaded. Other test
        # fixtures (``test_external_tools``, ``test_editor_agent_addon``)
        # pop ``bot.storage`` / ``bot.addons.state`` out of
        # ``sys.modules`` and re-import them against a tmp ``DATA_DIR``,
        # which means ``from bot.storage import storage`` (or even
        # ``from bot.addons import state``) returns a *fresh* instance
        # that the production handlers will never see. Grabbing storage
        # off the handlers module keeps every read/write in this test
        # file pointed at the same dict the handlers themselves touch,
        # so the snapshot/restore actually undoes the test's mutations.
        from bot.addons.helpzavr import handlers

        self.storage = handlers.addon_state.storage
        self._settings = self.storage._settings()  # noqa: SLF001
        self._helpzavr_before = copy.deepcopy(
            self._settings.get("addons", {}).get("helpzavr")
        )
        self._slot2_before = copy.deepcopy(
            self._settings.get("brain_slot2")
        )

    def restore(self):
        addons = self._settings.setdefault("addons", {})
        if self._helpzavr_before is None:
            addons.pop("helpzavr", None)
        else:
            addons["helpzavr"] = self._helpzavr_before
        if self._slot2_before is None:
            self._settings.pop("brain_slot2", None)
        else:
            self._settings["brain_slot2"] = self._slot2_before


class GetGroqKeyFallbackTest(unittest.TestCase):
    """Test ``get_groq_key()``'s fallback chain in isolation.

    These tests bypass the live :mod:`bot.storage` singleton and patch
    the two functions ``get_groq_key`` calls — ``addon_state.get`` and
    ``_brain_slot2_groq_key``. Going through storage instead would
    couple us to whatever ``Storage`` instance is in ``sys.modules``
    at the moment, which several other test fixtures swap out for tmp
    directories without cleaning up — see ``test_external_tools`` /
    ``test_editor_agent_addon``. Patching at the handler module
    boundary makes the test deterministic regardless of session order.
    """

    REAL_KEY = "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"

    def _run_with(self, *, stored: str, brain2: str) -> str:
        """Invoke ``get_groq_key`` with the two inputs forced to known
        values. Also clears ``GROQ_API_KEY`` for the duration so the
        env-fallback layer doesn't sneak in a real key on CI hosts.
        """
        import os

        from bot.addons.helpzavr import handlers

        prev_env = os.environ.pop("GROQ_API_KEY", None)
        with patch.object(
            handlers.addon_state,
            "get",
            side_effect=lambda addon, key, default="": (
                stored if (addon == "helpzavr" and key == "groq_api_key")
                else default
            ),
        ), patch.object(
            handlers, "_brain_slot2_groq_key", return_value=brain2
        ):
            try:
                return handlers.get_groq_key()
            finally:
                if prev_env is not None:
                    os.environ["GROQ_API_KEY"] = prev_env

    def test_stored_real_key_wins(self):
        self.assertEqual(
            self._run_with(stored=self.REAL_KEY, brain2=""), self.REAL_KEY
        )

    def test_model_name_in_field_is_ignored_and_brain2_used(self):
        """Historical regression: state.json had a model name in
        ``groq_api_key``. The stored value is rejected by the validator
        and Brain 2's real key is returned instead.
        """
        self.assertEqual(
            self._run_with(
                stored="meta-llama/llama-4-scout-17b-16e-instruct",
                brain2=self.REAL_KEY,
            ),
            self.REAL_KEY,
        )

    def test_brain2_used_when_helpzavr_field_empty(self):
        self.assertEqual(
            self._run_with(stored="", brain2=self.REAL_KEY), self.REAL_KEY
        )

    def test_brain2_ignored_when_base_url_is_not_groq(self):
        """A Brain 2 slot pointed at e.g. OpenRouter returns "" from
        ``_brain_slot2_groq_key`` (that helper checks the base_url).
        With no stored key and no env var, ``get_groq_key`` must
        return "" rather than blindly handing back any base64-shaped
        string the user happens to have.
        """
        self.assertEqual(self._run_with(stored="", brain2=""), "")


# ---- on_key_input validation --------------------------------------------


class OnKeyInputValidationTest(unittest.TestCase):
    """The router-level handler must refuse non-key text and keep the
    ``awaiting_key`` flag set so the next message gets another shot.
    """

    def setUp(self):
        # Reach for ``addon_state`` *via the handlers module* so that
        # reads/writes in the test and reads/writes inside the handler
        # both go through the same module object — even after other
        # test fixtures have rebuilt ``sys.modules['bot.addons.state']``
        # against a tmp ``DATA_DIR``. (``from bot.addons import state``
        # would resolve to whatever sys.modules has now, which is
        # *different* from the reference the long-loaded handlers
        # closed over at import time.) See conftest.py for the
        # session-level snapshot that keeps state.json clean.
        from bot.addons.helpzavr import handlers

        self.handlers = handlers
        self.addon_state = handlers.addon_state
        self.storage = self.addon_state.storage

        self.snap = _StorageSnapshot()
        # Same storage instance the handlers use → safe to clear here.
        self.snap._settings.setdefault("addons", {}).pop("helpzavr", None)
        self.snap._settings.pop("brain_slot2", None)
        self.snap._settings.pop("owner_id", None)
        self.storage.set_owner_id(7)

    def tearDown(self):
        self.snap.restore()

    def _on_key_input(self):
        router = self.handlers.build_helpzavr_router()
        for h in router.observers["message"].handlers:
            if h.callback.__name__ == "on_key_input":
                return h.callback
        raise AssertionError("on_key_input not registered")

    def _fake_message(self, text: str, chat_id: int = 700):
        msg = MagicMock()
        msg.chat.id = chat_id
        msg.text = text
        msg.answer = AsyncMock()
        msg.delete = AsyncMock()
        return msg

    def test_groq_rejects_model_name(self):
        self.addon_state.chat_set("helpzavr", 700, "awaiting_key", "groq")
        callback = self._on_key_input()
        msg = self._fake_message(
            "meta-llama/llama-4-scout-17b-16e-instruct", chat_id=700
        )
        asyncio.run(callback(msg, bot=MagicMock()))

        # No key written to storage.
        self.assertEqual(
            self.addon_state.get("helpzavr", "groq_api_key", ""), ""
        )
        # Awaiting flag still on — user gets another shot without
        # re-opening the screen.
        self.assertEqual(
            self.addon_state.chat_get("helpzavr", 700, "awaiting_key", ""),
            "groq",
        )
        # User got an explanatory reply.
        msg.answer.assert_awaited()
        body = msg.answer.await_args.args[0]
        self.assertIn("gsk_", body)

    def test_openrouter_rejects_model_name(self):
        self.addon_state.chat_set(
            "helpzavr", 701, "awaiting_key", "openrouter"
        )
        callback = self._on_key_input()
        msg = self._fake_message(
            "nvidia/nemotron-3-super-120b-a12b:free", chat_id=701
        )
        asyncio.run(callback(msg, bot=MagicMock()))

        self.assertEqual(
            self.addon_state.get("helpzavr", "openrouter_api_key", ""), ""
        )
        self.assertEqual(
            self.addon_state.chat_get(
                "helpzavr", 701, "awaiting_key", ""
            ),
            "openrouter",
        )
        msg.answer.assert_awaited()
        body = msg.answer.await_args.args[0]
        self.assertIn("sk-or-", body)

    def test_groq_accepts_real_key(self):
        self.addon_state.chat_set("helpzavr", 702, "awaiting_key", "groq")
        callback = self._on_key_input()
        real = "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"
        msg = self._fake_message(real, chat_id=702)
        asyncio.run(callback(msg, bot=MagicMock()))

        self.assertEqual(
            self.addon_state.get("helpzavr", "groq_api_key", ""), real
        )
        # Awaiting flag cleared on success.
        self.assertEqual(
            self.addon_state.chat_get("helpzavr", 702, "awaiting_key", ""),
            "",
        )

    def test_openrouter_accepts_real_key(self):
        self.addon_state.chat_set(
            "helpzavr", 703, "awaiting_key", "openrouter"
        )
        callback = self._on_key_input()
        real = (
            "sk-or-v1-"
            "817a266334c34fe7b3ebdcfdd80a6b6ec388dcf74e5dc9ce23c1bb786773cefe"
        )
        msg = self._fake_message(real, chat_id=703)
        asyncio.run(callback(msg, bot=MagicMock()))

        self.assertEqual(
            self.addon_state.get("helpzavr", "openrouter_api_key", ""),
            real,
        )
        self.assertEqual(
            self.addon_state.chat_get(
                "helpzavr", 703, "awaiting_key", ""
            ),
            "",
        )


if __name__ == "__main__":
    unittest.main()
