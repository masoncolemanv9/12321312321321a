"""Tests for the new pretty_text style infrastructure.

Coverage:

* ``rule-based presets`` — each preset is deterministic, decoration-only,
  and the words are unchanged after :func:`render_preset`.
* ``standard verifier`` — :func:`verify_words_unchanged` accepts decoration-
  only edits and rejects paraphrases.
* ``random cycling`` — clicking "Случайная стилистика" cycles through the
  five presets and the saved-preset slot persists independently.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch


def _isolate_state(monkey) -> None:
    """Point addon_state at a fresh tmp file for this test."""
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".json", prefix="addon-state-"
    ) as tmp:
        tmp_name = tmp.name
    monkey.setenv("DATA_DIR", os.path.dirname(tmp_name))


class PresetsTest(unittest.TestCase):
    """Each rule-based preset must preserve every word of the input.

    We strip Markdown markers + emojis from the output and compare it
    (case-folded, whitespace-normalised) to the original. If any preset
    adds, removes, or reorders words this test catches it.
    """

    SAMPLE = (
        "Привет, это тест.\n\n"
        "Прислал тебе текст из 50 символов. Скоро посмотрим как "
        "отформатируется.\n"
        "Главное чтобы все слова сохранились и стало красивее."
    )

    def _strip(self, text: str) -> str:
        from bot.addons.pretty_text.standard import _strip_decoration

        return _strip_decoration(text)

    def test_all_presets_preserve_words(self):
        from bot.addons.pretty_text.presets import PRESETS, render_preset

        orig = self._strip(self.SAMPLE)
        for preset_id, _ in PRESETS:
            with self.subTest(preset_id=preset_id):
                out = render_preset(preset_id, self.SAMPLE)
                stripped = self._strip(out)
                self.assertEqual(
                    stripped,
                    orig,
                    f"Preset {preset_id!r} altered words: "
                    f"{stripped!r} vs orig {orig!r}",
                )

    def test_next_preset_cycles(self):
        from bot.addons.pretty_text.presets import PRESETS, next_preset

        ids = [pid for pid, _ in PRESETS]
        cur = None
        seen: list[str] = []
        for _ in range(len(ids) + 2):  # past end → wrap
            cur = next_preset(cur)
            seen.append(cur)
        # First five entries should be exactly all preset ids in order.
        self.assertEqual(seen[: len(ids)], ids)
        # And it must wrap around without crashing.
        self.assertEqual(seen[len(ids)], ids[0])

    def test_render_preset_unknown_falls_back(self):
        from bot.addons.pretty_text.presets import render_preset

        # Unknown id → falls back to the first preset, no crash.
        out = render_preset("does-not-exist", "Привет")
        self.assertIn("Привет", out)


class StandardVerifierTest(unittest.TestCase):
    """``verify_words_unchanged`` is the safety net for the LLM."""

    def test_pure_decoration_passes(self):
        from bot.addons.pretty_text.standard import verify_words_unchanged

        orig = "Привет, друг! Сегодня в 18:00 разбор."
        # Decoration-only — wrap bold, prepend emoji, line breaks.
        decorated = "📌 **Привет**, друг!\n\nСегодня в *18:00* разбор."
        self.assertTrue(verify_words_unchanged(orig, decorated))

    def test_extra_words_fail(self):
        from bot.addons.pretty_text.standard import verify_words_unchanged

        orig = "Сегодня разбор в 18:00."
        # Model added a CTA word — must be rejected.
        decorated = "Сегодня разбор в **18:00**. Жми, чтобы записаться!"
        self.assertFalse(verify_words_unchanged(orig, decorated))

    def test_word_substitution_fails(self):
        from bot.addons.pretty_text.standard import verify_words_unchanged

        orig = "Это тест функции."
        # Synonym substitution — must be rejected even if emoji/bold added.
        decorated = "🔥 Это **проверка** функции."
        self.assertFalse(verify_words_unchanged(orig, decorated))

    def test_word_order_change_fails(self):
        from bot.addons.pretty_text.standard import verify_words_unchanged

        orig = "Утром в магазин."
        decorated = "В магазин **утром**."
        self.assertFalse(verify_words_unchanged(orig, decorated))


class HandlerFallbackTest(unittest.IsolatedAsyncioTestCase):
    """When ``standard`` LLM verification fails, ``_apply_style`` must
    fall back to mechanical formatting without raising.
    """

    async def test_standard_fallback_to_mechanical_on_word_change(self):
        from bot.addons.pretty_text import handlers as h

        async def _bad_llm(text):
            # Pretend the model paraphrased.
            from bot.addons.pretty_text.standard import RuntimeError as _RE  # noqa: F401

            raise RuntimeError("Модель попыталась переписать текст.")

        with patch.object(h, "rewrite_standard_decor_only", _bad_llm):
            formatted, preset = await h._apply_style(
                "Привет, мир!", h.STYLE_STANDARD, chat_id=10001
            )
        # Mechanical formatting preserves the words verbatim.
        self.assertIn("Привет", formatted)
        self.assertIn("мир", formatted)
        self.assertIsNone(preset)

    async def test_random_style_uses_current_preset(self):
        from bot.addons.pretty_text import handlers as h

        h.set_random_preset(10002, "minimal_bold")
        formatted, preset = await h._apply_style(
            "Это тест.", h.STYLE_RANDOM, chat_id=10002
        )
        self.assertEqual(preset, "minimal_bold")
        # The minimal_bold preset bolds the first words.
        self.assertIn("<b>", formatted)

    async def test_saved_style_falls_back_when_unset(self):
        from bot.addons.pretty_text import handlers as h

        # Make sure nothing is saved for this chat id.
        h.set_saved_preset(10003, None)
        formatted, preset = await h._apply_style(
            "Привет.", h.STYLE_SAVED, chat_id=10003
        )
        self.assertIsNone(preset)
        self.assertIn("Привет", formatted)


if __name__ == "__main__":
    unittest.main()
