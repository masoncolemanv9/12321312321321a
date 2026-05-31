"""Algorithm addon — router registration + handler wiring smoke tests."""

from __future__ import annotations

import inspect
import unittest


class AlgorithmRouterRegistrationTest(unittest.TestCase):
    """Verify the algorithm addon is wired into ``build_addon_routers``.

    We intentionally do NOT *call* ``build_addon_routers()`` from these
    tests because that eagerly imports every addon's handler module —
    including ``bot.addons.memory.handlers``, which caches a
    ``bot.storage.storage`` reference at import time. Some unrelated
    test suites (notably ``test_github_settings``) reassign
    ``bot.storage.storage`` mid-process via a pytest fixture, and if
    memory's handlers were imported BEFORE that reassignment they end up
    operating on a now-stale storage instance — which silently breaks
    ``test_memory_addon`` whenever the algorithm router tests run first.
    Static source inspection sidesteps that landmine.
    """

    def test_algorithm_router_is_in_build_list(self):
        from bot.addons import build_addon_routers

        src = inspect.getsource(build_addon_routers)
        self.assertIn("from .algorithm import build_algorithm_router", src)
        self.assertIn("build_algorithm_router()", src)

    def test_algorithm_registered_before_pretty_text(self):
        """Same rule as helpzavr/mailbox: input-awaiting addons before
        the catch-all formatter. Even though algorithm's text handlers
        are FSM-gated and would survive either order, keeping the
        invariant prevents the next bug of this family.
        """
        from bot.addons import build_addon_routers

        src = inspect.getsource(build_addon_routers)
        algo_pos = src.find("build_algorithm_router")
        pretty_pos = src.find("build_pretty_text_router")
        self.assertGreater(algo_pos, 0)
        self.assertGreater(pretty_pos, 0)
        self.assertLess(
            algo_pos,
            pretty_pos,
            "build_algorithm_router must be registered before "
            "build_pretty_text_router so pretty_text's catch-all "
            "message handler does not steal algorithm input.",
        )

    def test_callback_handlers_exist(self):
        """The router exposes the callback handlers the wizard wires
        from Settings."""
        from bot.addons.algorithm import build_algorithm_router

        router = build_algorithm_router()
        callbacks = router.observers["callback_query"].handlers
        names = {h.callback.__name__ for h in callbacks}
        # Spot-check the key entrypoints.
        for needed in (
            "cb_menu",
            "cb_slot_detail",
            "cb_ai",
            "cb_manual",
            "cb_run",
            "cb_delete",
            "cb_interval",
            "cb_interval_clear",
            "cb_rethink",
            "cb_save_draft",
            "cb_run_draft",
            "cb_back_to_list",
            "cb_back_to_slot",
            "cb_close",
        ):
            self.assertIn(needed, names, f"missing handler: {needed}")


if __name__ == "__main__":
    unittest.main()
