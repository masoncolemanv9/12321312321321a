"""Algorithm addon — executor sequencing + failure short-circuit."""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, patch


class ExecutorRunPlanTest(unittest.TestCase):
    def setUp(self):
        # See test_algorithm_state.py: must read/write state through the
        # same storage reference that the algorithm addon uses.
        from bot.addons import state as addon_state

        self._addons_root = addon_state._root()  # noqa: SLF001
        self._algo_snapshot = self._addons_root.get("algorithm")
        self._addons_root.pop("algorithm", None)

    def tearDown(self):
        if self._algo_snapshot is None:
            self._addons_root.pop("algorithm", None)
        else:
            self._addons_root["algorithm"] = self._algo_snapshot

    def test_runs_all_steps_in_order_when_each_succeeds(self):
        from bot.addons.algorithm import executor, state as algo_state

        slot = algo_state.get_slot(42, 1)
        slot.name = "test"
        slot.plan = "do A\ndo B\ndo C"
        algo_state.save_slot(42, slot)

        statuses: list[str] = []

        async def _status(text: str) -> None:
            statuses.append(text)

        calls: list[str] = []

        async def fake_run_agent(*, user_id, user_text, cwd, on_status):
            calls.append(user_text)
            return f"OK: {user_text}"

        async def _run():
            with patch("bot.agent.run_agent", side_effect=fake_run_agent):
                return await executor.run_plan(
                    chat_id=42, user_id=7, slot_index=1, status=_status
                )

        ran, total, last = asyncio.run(_run())
        self.assertEqual(calls, ["do A", "do B", "do C"])
        self.assertEqual(ran, 3)
        self.assertEqual(total, 3)
        self.assertTrue(last.startswith("OK"))
        # At least one progress line per step + a final summary.
        self.assertTrue(any("1/3" in s for s in statuses))
        self.assertTrue(any("3/3" in s for s in statuses))
        # After completion the is_running flag should be cleared.
        slot_after = algo_state.get_slot(42, 1)
        self.assertFalse(slot_after.is_running)
        self.assertGreater(slot_after.last_run_at, 0)

    def test_stops_on_first_failed_step(self):
        from bot.addons.algorithm import executor, state as algo_state

        slot = algo_state.get_slot(42, 2)
        slot.name = "test"
        slot.plan = "A\nB\nC"
        algo_state.save_slot(42, slot)

        statuses: list[str] = []

        async def _status(text: str) -> None:
            statuses.append(text)

        calls: list[str] = []

        async def fake_run_agent(*, user_id, user_text, cwd, on_status):
            calls.append(user_text)
            if user_text == "B":
                return "ERROR: cannot do B"
            return "OK"

        async def _run():
            with patch("bot.agent.run_agent", side_effect=fake_run_agent):
                return await executor.run_plan(
                    chat_id=42, user_id=7, slot_index=2, status=_status
                )

        ran, total, last = asyncio.run(_run())
        # Step C must NOT have been called.
        self.assertEqual(calls, ["A", "B"])
        self.assertEqual(ran, 2)
        self.assertEqual(total, 3)
        self.assertIn("ERROR", last)

    def test_empty_plan_short_circuits(self):
        from bot.addons.algorithm import executor, state as algo_state

        # Slot is empty.
        async def _run():
            statuses: list[str] = []

            async def _status(text: str) -> None:
                statuses.append(text)

            ran, total, last = await executor.run_plan(
                chat_id=42, user_id=7, slot_index=1, status=_status
            )
            return ran, total, statuses

        ran, total, statuses = asyncio.run(_run())
        self.assertEqual(ran, 0)
        self.assertEqual(total, 0)
        self.assertTrue(any("пустой" in s.lower() for s in statuses))

    def test_concurrent_runs_blocked_by_is_running(self):
        from bot.addons.algorithm import executor, state as algo_state

        slot = algo_state.get_slot(42, 4)
        slot.name = "t"
        slot.plan = "A"
        slot.is_running = True  # pretend a run is already in progress
        algo_state.save_slot(42, slot)

        statuses: list[str] = []

        async def _status(text: str) -> None:
            statuses.append(text)

        called = []

        async def fake_run_agent(*, user_id, user_text, cwd, on_status):
            called.append(user_text)
            return "OK"

        async def _run():
            with patch("bot.agent.run_agent", side_effect=fake_run_agent):
                return await executor.run_plan(
                    chat_id=42, user_id=7, slot_index=4, status=_status
                )

        ran, total, last = asyncio.run(_run())
        self.assertEqual(called, [])
        self.assertEqual(ran, 0)
        self.assertTrue(
            any("уже выполняется" in s for s in statuses),
            f"expected single-flight warning, got {statuses}",
        )

    def test_explicit_plan_steps_bypass_slot(self):
        """`plan_steps=[...]` lets the AI-draft preview run before the
        plan is saved into a slot."""
        from bot.addons.algorithm import executor

        statuses: list[str] = []

        async def _status(text: str) -> None:
            statuses.append(text)

        calls: list[str] = []

        async def fake_run_agent(*, user_id, user_text, cwd, on_status):
            calls.append(user_text)
            return "OK"

        async def _run():
            with patch("bot.agent.run_agent", side_effect=fake_run_agent):
                return await executor.run_plan(
                    chat_id=42,
                    user_id=7,
                    slot_index=1,
                    status=_status,
                    plan_steps=["draft1", "draft2"],
                )

        ran, total, _ = asyncio.run(_run())
        self.assertEqual(calls, ["draft1", "draft2"])
        self.assertEqual((ran, total), (2, 2))


if __name__ == "__main__":
    unittest.main()
