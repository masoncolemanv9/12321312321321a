"""Tests for the /work batch-terminal command and related plumbing.

The /work flow is FSM-driven (``WorkStates.awaiting_commands``) and runs
multi-line command batches sequentially with a much higher per-command
timeout than /exec. These tests cover the user-visible contracts without
spinning up the full aiogram dispatcher:

* ``EXEC_TIMEOUT`` / ``WORK_EXEC_TIMEOUT`` defaults are reasonable.
* ``exec_bash`` accepts a ``timeout`` override and forwards it.
* ``cmd_work`` flips the FSM into the ``awaiting_commands`` state.
* ``capture_work_batch`` parses the batch, ignores blanks/comments, and
  invokes ``_work_run_line`` once per real command.
"""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.fixture()
def fresh_handlers(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BOT_PERSONA", "boss")
    for mod in (
        "bot.config",
        "bot.storage",
        "bot.tools",
        "bot.persona",
        "bot.access",
        "bot.inbox",
        "bot.agent",
        "bot.handlers",
    ):
        sys.modules.pop(mod, None)
    config = importlib.import_module("bot.config")
    storage_mod = importlib.import_module("bot.storage")
    storage_mod.storage = storage_mod.Storage(data_dir=config.DATA_DIR)
    handlers = importlib.import_module("bot.handlers")
    return handlers, storage_mod.storage, config


def _msg(text: str, *, user_id: int = 1, chat_id: int = 1):
    """Build a minimal Message-shaped stub with an async ``answer``."""
    return SimpleNamespace(
        text=text,
        from_user=SimpleNamespace(id=user_id),
        chat=SimpleNamespace(id=chat_id),
        answer=AsyncMock(),
        bot=SimpleNamespace(send_chat_action=AsyncMock()),
    )


def test_exec_timeout_defaults_are_sane(fresh_handlers) -> None:
    """EXEC_TIMEOUT should be roomy enough for installs; /work even more so."""
    _, _, config = fresh_handlers
    assert config.EXEC_TIMEOUT >= 300, (
        f"EXEC_TIMEOUT={config.EXEC_TIMEOUT} is too tight for real installs"
    )
    assert config.WORK_EXEC_TIMEOUT >= config.EXEC_TIMEOUT, (
        "WORK_EXEC_TIMEOUT must be at least as large as EXEC_TIMEOUT"
    )


def test_exec_bash_accepts_timeout_override(fresh_handlers, monkeypatch) -> None:
    """``exec_bash`` should forward an explicit ``timeout`` to ``_run_shell``."""
    handlers, _, _ = fresh_handlers
    import asyncio

    from bot import tools

    seen: dict[str, object] = {}

    async def fake_run_shell(command, cwd=None, timeout=0):
        seen["command"] = command
        seen["timeout"] = timeout
        return 0, "hi\n", ""

    monkeypatch.setattr(tools, "_run_shell", fake_run_shell)

    out = asyncio.run(tools.exec_bash(tools.Path("/tmp"), "echo hi", timeout=1234))
    assert "hi" in out
    assert seen["timeout"] == 1234


def test_cmd_work_sets_fsm_state(fresh_handlers) -> None:
    """/work should flip the user into the awaiting_commands state."""
    handlers, _, _ = fresh_handlers
    import asyncio

    # Make the auth gate transparent — we're testing the state transition.
    handlers._is_authorized = lambda _msg: True  # type: ignore[assignment]
    handlers._role_chosen = lambda: True  # type: ignore[assignment]

    state = SimpleNamespace(
        set_state=AsyncMock(),
        get_state=AsyncMock(return_value=None),
        clear=AsyncMock(),
    )
    msg = _msg("/work")
    asyncio.run(handlers.cmd_work(msg, state))

    state.set_state.assert_awaited_once_with(handlers.WorkStates.awaiting_commands)
    msg.answer.assert_awaited()


def test_capture_work_batch_filters_blanks_and_comments(fresh_handlers) -> None:
    """Blank lines and ``#``-comments must NOT reach ``_work_run_line``."""
    handlers, _, _ = fresh_handlers
    import asyncio

    handlers._is_authorized = lambda _msg: True  # type: ignore[assignment]

    runs: list[str] = []

    async def fake_run_line(_msg, line: str) -> None:
        runs.append(line)

    handlers._work_run_line = fake_run_line  # type: ignore[assignment]

    state = SimpleNamespace(clear=AsyncMock())
    batch = "\n".join(
        [
            "# install deps",
            "",
            "/exec pip install -e .",
            "   ",
            "/exec playwright install chromium",
            "# done",
        ]
    )
    msg = _msg(batch)
    asyncio.run(handlers.capture_work_batch(msg, state))

    assert runs == [
        "/exec pip install -e .",
        "/exec playwright install chromium",
    ]
    state.clear.assert_awaited_once()


def test_capture_work_batch_reports_empty(fresh_handlers) -> None:
    """A comments-only message should be rejected without clearing state."""
    handlers, _, _ = fresh_handlers
    import asyncio

    handlers._is_authorized = lambda _msg: True  # type: ignore[assignment]

    runs: list[str] = []

    async def fake_run_line(_msg, line: str) -> None:
        runs.append(line)

    handlers._work_run_line = fake_run_line  # type: ignore[assignment]

    state = SimpleNamespace(clear=AsyncMock())
    msg = _msg("# only a comment\n# nothing else\n")
    asyncio.run(handlers.capture_work_batch(msg, state))

    assert runs == []
    state.clear.assert_not_awaited()
