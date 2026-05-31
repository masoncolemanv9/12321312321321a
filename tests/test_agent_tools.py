"""Tests for the agent-facing tool dispatcher.

The agent exposes a set of OpenAI-compatible tools (``list_dir`` /
``read_file`` / ``write_file`` / ``exec_bash`` / ``clone_repo`` /
``list_projects`` / ``switch_project``) via ``bot.tools.dispatch_tool``.
These tests focus on the GitHub-tooling additions: that ``clone_repo``,
``list_projects`` and ``switch_project`` work without a pre-existing
``cwd`` and that ``clone_repo`` mutates the per-user active project so
subsequent tool calls land inside the cloned repo automatically.

We never hit the network — we stub ``git clone`` by monkey-patching
``bot.tools._run`` to return a fake success and pre-creating the
destination directory ourselves.
"""

from __future__ import annotations

import importlib
import sys

import pytest


@pytest.fixture()
def fresh_tools(tmp_path, monkeypatch):
    """Fresh ``bot.tools`` + ``bot.storage`` rooted at ``tmp_path``."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BOT_PERSONA", "boss")
    # ``bot.external_tools`` captures ``storage`` at module load (``from
    # .storage import storage``) — without popping it we'd keep the
    # binding to the production singleton, and the test would happily
    # call the real Tavily API if the user had a key saved.
    for mod in ("bot.config", "bot.storage", "bot.tools", "bot.persona",
                "bot.access", "bot.external_tools"):
        sys.modules.pop(mod, None)
    config = importlib.import_module("bot.config")
    storage_mod = importlib.import_module("bot.storage")
    storage_mod.storage = storage_mod.Storage(data_dir=config.DATA_DIR)
    tools = importlib.import_module("bot.tools")
    # Pin the fresh storage onto ``external_tools`` too. When a previous
    # test imported ``bot.external_tools as ext`` at module level, pytest
    # still holds that *module object* alive — even after we pop the
    # module from ``sys.modules``. Subsequent re-imports go through
    # ``sys.modules``, but lazy ``from . import external_tools`` calls
    # inside ``tools.dispatch_tool`` can in some test orderings resolve
    # to the *original* module object whose ``storage`` attribute still
    # points at the production singleton (which now holds whatever real
    # keys the user batch-imported into ``/tmp/bot-data/state.json``).
    external_tools = importlib.import_module("bot.external_tools")
    external_tools.storage = storage_mod.storage
    monkeypatch.setattr(external_tools, "storage", storage_mod.storage)
    return tools, storage_mod.storage, config.PROJECTS_DIR


def test_tool_definitions_include_git_tools(fresh_tools) -> None:
    tools, _, _ = fresh_tools
    names = {t["function"]["name"] for t in tools.TOOL_DEFINITIONS}
    assert {"clone_repo", "list_projects", "switch_project"} <= names


def test_tool_definitions_include_external_research_tools(fresh_tools) -> None:
    tools, _, _ = fresh_tools
    names = {t["function"]["name"] for t in tools.TOOL_DEFINITIONS}
    assert {
        "tavily_search",
        "firecrawl_scrape",
        "brave_search",
        "exa_search",
        "apify_run_actor",
        "github_search_code",
        "github_get_file",
    } <= names


def test_dispatch_external_tool_without_key_returns_friendly_message(
    fresh_tools,
) -> None:
    """Sanity check: dispatch routes the new name to ``external_tools``,
    and a missing key produces the user-friendly hint instead of a
    crash. We pick ``tavily_search`` arbitrarily — all six share the
    same code path."""
    tools, _, _ = fresh_tools
    import asyncio

    out = asyncio.run(
        tools.dispatch_tool(
            "tavily_search",
            '{"query": "hello"}',
            None,
            user_id=1,
        )
    )
    assert "не задан" in out.lower()
    assert "tavily" in out.lower()


def test_list_projects_empty_returns_hint(fresh_tools) -> None:
    tools, _, _ = fresh_tools
    import asyncio

    out = asyncio.run(tools.dispatch_tool("list_projects", "{}", None, user_id=1))
    assert "clone_repo" in out
    assert "No projects yet" in out


def test_clone_repo_sets_active_project(fresh_tools, monkeypatch) -> None:
    tools, storage, projects_dir = fresh_tools
    import asyncio

    # Stub `git clone` — create the dest dir like a real clone would.
    async def fake_run(cmd, cwd=None, timeout=60):
        dest = cmd[-1]
        from pathlib import Path

        Path(dest).mkdir(parents=True, exist_ok=True)
        (Path(dest) / "README.md").write_text("hello world\n")
        return 0, "", ""

    monkeypatch.setattr(tools, "_run", fake_run)

    user_id = 42
    out = asyncio.run(
        tools.dispatch_tool(
            "clone_repo",
            '{"url": "https://github.com/owner/repo"}',
            None,
            user_id=user_id,
        )
    )
    assert "Cloned" in out and "repo" in out

    # Side effects: project dir exists AND user's active cwd is set.
    assert (projects_dir / "repo").exists()
    assert storage.get_cwd(user_id) == projects_dir / "repo"


def test_clone_repo_rejects_traversal_in_name(fresh_tools) -> None:
    tools, _, _ = fresh_tools
    import asyncio

    with pytest.raises(tools.ToolError):
        asyncio.run(
            tools.dispatch_tool(
                "clone_repo",
                '{"url": "https://x.example/repo", "name": "../escape"}',
                None,
                user_id=1,
            )
        )


def test_clone_repo_blocks_existing_project(fresh_tools, monkeypatch) -> None:
    tools, _, projects_dir = fresh_tools
    import asyncio

    (projects_dir / "already").mkdir()

    with pytest.raises(tools.ToolError) as exc:
        asyncio.run(
            tools.dispatch_tool(
                "clone_repo",
                '{"url": "https://x/repo", "name": "already"}',
                None,
                user_id=1,
            )
        )
    assert "switch_project" in str(exc.value)


def test_switch_project_updates_active_cwd(fresh_tools) -> None:
    tools, storage, projects_dir = fresh_tools
    import asyncio

    (projects_dir / "alpha").mkdir()
    (projects_dir / "beta").mkdir()

    out = asyncio.run(
        tools.dispatch_tool(
            "switch_project",
            '{"name": "beta"}',
            None,
            user_id=7,
        )
    )
    assert "Switched" in out and "beta" in out
    assert storage.get_cwd(7) == projects_dir / "beta"


def test_switch_project_unknown_returns_error(fresh_tools) -> None:
    tools, _, _ = fresh_tools
    import asyncio

    with pytest.raises(tools.ToolError) as exc:
        asyncio.run(
            tools.dispatch_tool(
                "switch_project",
                '{"name": "ghost"}',
                None,
                user_id=1,
            )
        )
    assert "not found" in str(exc.value).lower()


def test_list_projects_after_clone_lists_repo(fresh_tools, monkeypatch) -> None:
    tools, _, projects_dir = fresh_tools
    import asyncio

    async def fake_run(cmd, cwd=None, timeout=60):
        from pathlib import Path

        Path(cmd[-1]).mkdir(parents=True, exist_ok=True)
        return 0, "", ""

    monkeypatch.setattr(tools, "_run", fake_run)

    asyncio.run(
        tools.dispatch_tool(
            "clone_repo",
            '{"url": "https://x.example/myrepo"}',
            None,
            user_id=99,
        )
    )
    out = asyncio.run(
        tools.dispatch_tool("list_projects", "{}", None, user_id=99)
    )
    assert "myrepo" in out


def test_file_tools_still_require_cwd(fresh_tools) -> None:
    """Backwards-compat: list_dir / read_file etc. still need a project."""
    tools, _, _ = fresh_tools
    import asyncio

    with pytest.raises(tools.ToolError) as exc:
        asyncio.run(tools.dispatch_tool("list_dir", '{"path": "."}', None, user_id=1))
    assert "No project selected" in str(exc.value)
