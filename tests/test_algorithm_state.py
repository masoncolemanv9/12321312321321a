"""Algorithm addon — slot state CRUD + name deduplication."""

from __future__ import annotations

import unittest


class AlgorithmStateTest(unittest.TestCase):
    def setUp(self):
        # NOTE: must clear state through the SAME storage reference
        # that ``bot.addons.algorithm.state`` uses internally. Some
        # earlier tests (test_access etc.) reassign
        # ``bot.storage.storage`` to a fresh instance, but
        # ``bot.addons.state`` caches the original at import time —
        # the two diverge for the rest of the process.
        from bot.addons import state as addon_state

        self._addons_root = addon_state._root()  # noqa: SLF001
        self._algo_snapshot = self._addons_root.get("algorithm")
        self._addons_root.pop("algorithm", None)

    def tearDown(self):
        if self._algo_snapshot is None:
            self._addons_root.pop("algorithm", None)
        else:
            self._addons_root["algorithm"] = self._algo_snapshot

    def test_empty_slots_default(self):
        from bot.addons.algorithm import state as algo_state

        slots = algo_state.list_slots(chat_id=42)
        self.assertEqual(len(slots), 10)
        for i, slot in enumerate(slots, start=1):
            self.assertEqual(slot.index, i)
            self.assertTrue(slot.is_empty, f"slot {i} not empty: {slot}")
            self.assertEqual(slot.label(), f"Слот {i}")
            self.assertFalse(slot.has_interval)

    def test_save_and_load_slot(self):
        from bot.addons.algorithm import state as algo_state

        s = algo_state.get_slot(42, 3)
        s.name = "гугл"
        s.plan = "step 1\nstep 2"
        s.interval_minutes = 1.5
        algo_state.save_slot(42, s)

        loaded = algo_state.get_slot(42, 3)
        self.assertEqual(loaded.name, "гугл")
        self.assertEqual(loaded.plan, "step 1\nstep 2")
        self.assertEqual(loaded.interval_minutes, 1.5)
        self.assertFalse(loaded.is_empty)
        self.assertTrue(loaded.has_interval)
        self.assertEqual(loaded.label(), "3. гугл")

    def test_per_chat_isolation(self):
        from bot.addons.algorithm import state as algo_state

        s1 = algo_state.get_slot(111, 1)
        s1.name = "одно"
        s1.plan = "do A"
        algo_state.save_slot(111, s1)

        s2 = algo_state.get_slot(222, 1)
        self.assertTrue(s2.is_empty)

        s2.name = "другое"
        s2.plan = "do B"
        algo_state.save_slot(222, s2)

        self.assertEqual(algo_state.get_slot(111, 1).name, "одно")
        self.assertEqual(algo_state.get_slot(222, 1).name, "другое")
        self.assertCountEqual(algo_state.list_chat_ids(), [111, 222])

    def test_clear_slot(self):
        from bot.addons.algorithm import state as algo_state

        s = algo_state.get_slot(42, 5)
        s.name = "x"
        s.plan = "step"
        algo_state.save_slot(42, s)
        self.assertFalse(algo_state.get_slot(42, 5).is_empty)

        algo_state.clear_slot(42, 5)
        self.assertTrue(algo_state.get_slot(42, 5).is_empty)

    def test_slot_index_validated(self):
        from bot.addons.algorithm import state as algo_state

        with self.assertRaises(ValueError):
            algo_state.get_slot(42, 0)
        with self.assertRaises(ValueError):
            algo_state.get_slot(42, 11)


class AlgorithmNameDedupTest(unittest.TestCase):
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

    def test_unique_name_returned_as_is(self):
        from bot.addons.algorithm import state as algo_state

        self.assertEqual(
            algo_state.derive_unique_name(42, "гугл"), "гугл"
        )

    def test_duplicate_gets_suffix_2(self):
        from bot.addons.algorithm import state as algo_state

        s = algo_state.get_slot(42, 1)
        s.name = "гугл"
        s.plan = "step"
        algo_state.save_slot(42, s)

        self.assertEqual(
            algo_state.derive_unique_name(42, "гугл"), "гугл 2"
        )

    def test_third_duplicate_gets_suffix_3(self):
        from bot.addons.algorithm import state as algo_state

        for i, name in enumerate(("гугл", "гугл 2"), start=1):
            s = algo_state.get_slot(42, i)
            s.name = name
            s.plan = "step"
            algo_state.save_slot(42, s)
        self.assertEqual(
            algo_state.derive_unique_name(42, "гугл"), "гугл 3"
        )

    def test_exclude_index_lets_slot_keep_its_own_name(self):
        """Re-saving the same slot with the same name must NOT bump it
        to ``"гугл 2"`` — the slot's existing name is excluded from the
        collision set.
        """
        from bot.addons.algorithm import state as algo_state

        s = algo_state.get_slot(42, 1)
        s.name = "гугл"
        s.plan = "step"
        algo_state.save_slot(42, s)

        self.assertEqual(
            algo_state.derive_unique_name(42, "гугл", exclude_index=1),
            "гугл",
        )

    def test_empty_name_falls_back_to_placeholder(self):
        from bot.addons.algorithm import state as algo_state

        self.assertEqual(algo_state.derive_unique_name(42, ""), "без имени")
        self.assertEqual(algo_state.derive_unique_name(42, "   "), "без имени")

    def test_per_chat_dedup_does_not_cross_chats(self):
        from bot.addons.algorithm import state as algo_state

        s = algo_state.get_slot(111, 1)
        s.name = "гугл"
        s.plan = "step"
        algo_state.save_slot(111, s)

        # Other chat is unaffected.
        self.assertEqual(
            algo_state.derive_unique_name(222, "гугл"), "гугл"
        )


if __name__ == "__main__":
    unittest.main()
