"""Algorithm addon — scheduler ``_due_now`` logic + tick routing."""

from __future__ import annotations

import asyncio
import time
import unittest
from unittest.mock import patch


class DueNowTest(unittest.TestCase):
    def test_empty_slot_never_due(self):
        from bot.addons.algorithm.scheduler import _due_now
        from bot.addons.algorithm.state import Slot

        slot = Slot(index=1)
        self.assertFalse(_due_now(slot, now=time.time()))

    def test_no_interval_never_due(self):
        from bot.addons.algorithm.scheduler import _due_now
        from bot.addons.algorithm.state import Slot

        slot = Slot(index=1, plan="step", interval_minutes=0.0)
        self.assertFalse(_due_now(slot, now=time.time()))

    def test_already_running_skipped(self):
        from bot.addons.algorithm.scheduler import _due_now
        from bot.addons.algorithm.state import Slot

        now = time.time()
        slot = Slot(
            index=1,
            plan="step",
            interval_minutes=1.0,
            last_run_at=0.0,  # very old
            is_running=True,
        )
        self.assertFalse(_due_now(slot, now=now))

    def test_due_when_interval_has_passed(self):
        from bot.addons.algorithm.scheduler import _due_now
        from bot.addons.algorithm.state import Slot

        now = time.time()
        slot = Slot(
            index=1,
            plan="step",
            interval_minutes=1.0,
            last_run_at=now - 120,  # 2 min ago, interval = 1 min
        )
        self.assertTrue(_due_now(slot, now=now))

    def test_not_due_when_interval_not_passed(self):
        from bot.addons.algorithm.scheduler import _due_now
        from bot.addons.algorithm.state import Slot

        now = time.time()
        slot = Slot(
            index=1,
            plan="step",
            interval_minutes=5.0,
            last_run_at=now - 60,  # only 1 min ago, interval = 5 min
        )
        self.assertFalse(_due_now(slot, now=now))

    def test_sub_minute_interval_supported(self):
        """0.5 min = 30 s — the user's verbatim ‘0.5 = pol minuty’."""
        from bot.addons.algorithm.scheduler import _due_now
        from bot.addons.algorithm.state import Slot

        now = time.time()
        slot = Slot(
            index=1,
            plan="step",
            interval_minutes=0.5,
            last_run_at=now - 45,  # 45 s ago, interval = 30 s
        )
        self.assertTrue(_due_now(slot, now=now))


class SchedulerTickTest(unittest.TestCase):
    def setUp(self):
        from bot.addons import state as addon_state

        self._addons_root = addon_state._root()  # noqa: SLF001
        self._algo_snapshot = self._addons_root.get("algorithm")
        self._addons_root.pop("algorithm", None)

    def tearDown(self):
        if self._algo_snapshot is None:
            self._addons_root.pop("algorithm", None)
        else:
            self._addons_root["algorithm"] = self._algo_snapshot

    def test_tick_fires_run_one_for_due_slot(self):
        from bot.addons.algorithm import scheduler, state as algo_state

        slot = algo_state.get_slot(42, 1)
        slot.name = "test"
        slot.plan = "step"
        slot.interval_minutes = 1.0
        slot.last_run_at = time.time() - 600  # 10 min ago
        algo_state.save_slot(42, slot)

        run_calls: list[tuple[int, int]] = []

        async def fake_run_one(bot, chat_id, slot_index):
            run_calls.append((chat_id, slot_index))

        async def _run():
            with patch.object(scheduler, "_run_one", fake_run_one):
                await scheduler._tick(bot=object())  # noqa: SLF001
                # Let the create_task'd coroutine run.
                await asyncio.sleep(0.05)

        asyncio.run(_run())
        self.assertIn((42, 1), run_calls)

    def test_tick_skips_non_due_slot(self):
        from bot.addons.algorithm import scheduler, state as algo_state

        slot = algo_state.get_slot(42, 1)
        slot.name = "test"
        slot.plan = "step"
        slot.interval_minutes = 1440.0  # 1 day
        slot.last_run_at = time.time() - 60  # 1 min ago
        algo_state.save_slot(42, slot)

        run_calls: list[tuple[int, int]] = []

        async def fake_run_one(bot, chat_id, slot_index):
            run_calls.append((chat_id, slot_index))

        async def _run():
            with patch.object(scheduler, "_run_one", fake_run_one):
                await scheduler._tick(bot=object())  # noqa: SLF001
                await asyncio.sleep(0.05)

        asyncio.run(_run())
        self.assertEqual(run_calls, [])


if __name__ == "__main__":
    unittest.main()
