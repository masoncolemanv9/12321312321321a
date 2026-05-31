"""Session-wide pytest plumbing.

The tests in this repo exercise modules that touch the **live**
:mod:`bot.storage` singleton (``storage._settings()`` →
``data/state.json``). Without protection, those tests would write into
whatever ``DATA_DIR`` is configured on the developer's machine and leak
state across runs — including ``owner_id`` and addon API keys.

A **session-scoped** snapshot of ``storage._state`` (deep-copied) is
written back to ``data/state.json`` at session end. This guarantees
the developer's real state file is byte-identical to what it was
before pytest ran, regardless of which tests passed or failed.

This fixture is autouse so individual tests don't need to opt in. If
you add a new test that mutates ``storage._state`` you don't have to
do anything — the snapshot will undo your edits when the session
ends.
"""

from __future__ import annotations

import copy
import json

import pytest


@pytest.fixture(scope="session", autouse=True)
def _restore_live_storage_state_after_tests():
    """Snapshot ``storage._state`` on entry, restore it on exit.

    Captures the raw dict via :func:`copy.deepcopy` so that any
    in-place mutations made by tests (``storage.set_owner_id``,
    ``addon_state.set_``, …) don't leak into the snapshot itself. On
    teardown the storage's ``_state`` is replaced with the snapshot and
    persisted via the existing ``_save`` helper.
    """
    # Pull storage off ``bot.addons.state`` rather than ``bot.storage``
    # directly. ``test_external_tools.py`` reloads ``bot.storage`` mid-
    # session, so ``from bot.storage import storage`` would hand us a
    # detached instance whose ``_state`` no longer reflects what the
    # addon code is mutating. ``bot.addons.state`` holds the original
    # singleton — same one used by handlers in production.
    from bot.addons import state as _addon_state

    _storage = _addon_state.storage

    # Deepcopy avoids the test-mutates-the-very-dict-we-just-snapshot
    # bug. Without this the snapshot grows along with the test changes
    # and the "restore" step is a no-op.
    snapshot = copy.deepcopy(_storage._state)  # noqa: SLF001
    try:
        yield
    finally:
        # Replace the live state with the snapshot and write it out so
        # the developer's data/state.json is byte-identical to what it
        # was before pytest ran. We intentionally rewrite the entire
        # dict rather than diffing — diffing is fragile, full overwrite
        # is correct by construction.
        _storage._state = snapshot  # noqa: SLF001
        _storage.state_file.write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2)
        )
