"""Algorithm addon — planner parsing + brain chain priority."""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch


class PlannerParseTest(unittest.TestCase):
    def test_parses_clean_json(self):
        from bot.addons.algorithm.planner import _parse_plan_text

        name, steps = _parse_plan_text(
            '{"name": "гугл", "steps": ["Шаг 1", "Шаг 2"]}', "fallback"
        )
        self.assertEqual(name, "гугл")
        self.assertEqual(steps, ["Шаг 1", "Шаг 2"])

    def test_parses_json_in_markdown_fence(self):
        from bot.addons.algorithm.planner import _parse_plan_text

        raw = '```json\n{"name": "гугл", "steps": ["a", "b"]}\n```'
        name, steps = _parse_plan_text(raw, "fallback")
        self.assertEqual(name, "гугл")
        self.assertEqual(steps, ["a", "b"])

    def test_extracts_json_block_from_chatty_response(self):
        from bot.addons.algorithm.planner import _parse_plan_text

        raw = (
            "Конечно! Вот план:\n\n"
            '{"name": "почта", "steps": ["проверить", "ответить"]}\n\n'
            "Удачи!"
        )
        name, steps = _parse_plan_text(raw, "fallback")
        self.assertEqual(name, "почта")
        self.assertEqual(steps, ["проверить", "ответить"])

    def test_falls_back_to_line_split_on_non_json(self):
        from bot.addons.algorithm.planner import _parse_plan_text

        raw = "1. Открыть сайт\n2. Сделать скриншот\n3. Прислать в чат"
        name, steps = _parse_plan_text(raw, "fallback")
        self.assertEqual(name, "fallback")
        self.assertEqual(
            steps,
            ["Открыть сайт", "Сделать скриншот", "Прислать в чат"],
        )

    def test_empty_input_returns_no_steps(self):
        from bot.addons.algorithm.planner import _parse_plan_text

        name, steps = _parse_plan_text("", "fallback")
        self.assertEqual(name, "fallback")
        self.assertEqual(steps, [])

    def test_derive_fallback_name_short_first_words(self):
        from bot.addons.algorithm.planner import _derive_fallback_name

        self.assertEqual(
            _derive_fallback_name("Откройте Google Cloud и сделайте"),
            "откройте google",
        )

    def test_derive_fallback_name_strips_urls(self):
        from bot.addons.algorithm.planner import _derive_fallback_name

        self.assertEqual(
            _derive_fallback_name("https://example.com тест проверка"),
            "тест проверка",
        )


class PlannerChainTest(unittest.TestCase):
    """The planner must prefer any slot configured with
    ``provider == "custom"`` (= «Другое»), then slot 1, then
    slot 2. Slots without an API key are skipped entirely."""

    def test_only_slot1_openrouter_configured(self):
        from bot.addons.algorithm import planner

        with patch(
            "bot.agent._slot_cfg",
            side_effect=lambda s: {
                "1": ("openrouter", "key-1", "", "model-1"),
                "2": ("", "", "", ""),
            }[s],
        ):
            chain = planner._planner_chain()  # noqa: SLF001

        self.assertEqual(chain, [("slot1", "1")])

    def test_bomba_in_slot1_wins(self):
        from bot.addons.algorithm import planner

        with patch(
            "bot.agent._slot_cfg",
            side_effect=lambda s: {
                "1": ("custom", "bomba-key", "https://b/v1", "x"),
                "2": ("openrouter", "or-key", "", "y"),
            }[s],
        ):
            chain = planner._planner_chain()  # noqa: SLF001

        # «Другое» (slot1, custom) first, then slot2.
        self.assertEqual(chain, [("bomba", "1"), ("slot2", "2")])

    def test_bomba_in_slot2_jumps_ahead(self):
        from bot.addons.algorithm import planner

        with patch(
            "bot.agent._slot_cfg",
            side_effect=lambda s: {
                "1": ("openrouter", "or-key", "", "x"),
                "2": ("custom", "bomba-key", "https://b/v1", "y"),
            }[s],
        ):
            chain = planner._planner_chain()  # noqa: SLF001

        self.assertEqual(chain, [("bomba", "2"), ("slot1", "1")])

    def test_slot_without_key_is_dropped(self):
        from bot.addons.algorithm import planner

        with patch(
            "bot.agent._slot_cfg",
            side_effect=lambda s: {
                "1": ("custom", "", "", ""),  # missing key — drop
                "2": ("openrouter", "or-key", "", "y"),
            }[s],
        ):
            chain = planner._planner_chain()  # noqa: SLF001

        self.assertEqual(chain, [("slot2", "2")])

    def test_no_brains_configured(self):
        from bot.addons.algorithm import planner

        with patch(
            "bot.agent._slot_cfg",
            side_effect=lambda s: ("", "", "", ""),
        ):
            chain = planner._planner_chain()  # noqa: SLF001
        self.assertEqual(chain, [])

    def test_no_brains_raises_no_brain_available(self):
        from bot.addons.algorithm import planner

        async def _run():
            with patch(
                "bot.agent._slot_cfg",
                side_effect=lambda s: ("", "", "", ""),
            ):
                await planner.plan_from_text("hello")

        with self.assertRaises(planner.NoBrainAvailable):
            asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
