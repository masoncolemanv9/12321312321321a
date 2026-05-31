"""Tests for the GitHub settings UI bits: storage primitives, the
auto-clone toggle, repo metadata, the agent's system-prompt inventory,
and the capabilities log writer.

The wizard's inline-keyboard callbacks aren't easy to test without a
full aiogram Dispatcher stack — we cover them indirectly by verifying
the helper functions they call (`_kb_github_menu`, `_github_menu_body`)
do the right thing.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


@pytest.fixture()
def fresh(tmp_path, monkeypatch):
    """Fresh storage / tools / agent / capabilities rooted in tmp_path."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BOT_PERSONA", "boss")
    for mod in (
        "bot.config",
        "bot.storage",
        "bot.tools",
        "bot.persona",
        "bot.access",
        "bot.agent",
        "bot.capabilities",
    ):
        sys.modules.pop(mod, None)
    config = importlib.import_module("bot.config")
    storage_mod = importlib.import_module("bot.storage")
    storage_mod.storage = storage_mod.Storage(data_dir=config.DATA_DIR)
    tools = importlib.import_module("bot.tools")
    agent = importlib.import_module("bot.agent")
    capabilities = importlib.import_module("bot.capabilities")
    return {
        "storage": storage_mod.storage,
        "tools": tools,
        "agent": agent,
        "capabilities": capabilities,
        "config": config,
    }


def test_github_auto_clone_defaults_to_true(fresh) -> None:
    assert fresh["storage"].get_github_auto_clone() is True


def test_github_auto_clone_toggle(fresh) -> None:
    storage = fresh["storage"]
    storage.set_github_auto_clone(False)
    assert storage.get_github_auto_clone() is False
    storage.set_github_auto_clone(True)
    assert storage.get_github_auto_clone() is True


def test_list_github_repos_empty_by_default(fresh) -> None:
    assert fresh["storage"].list_github_repos() == []


def test_clone_writes_repo_meta_with_description(fresh, monkeypatch) -> None:
    tools = fresh["tools"]
    storage = fresh["storage"]
    import asyncio

    async def fake_run(cmd, cwd=None, timeout=60):
        dest = Path(cmd[-1])
        dest.mkdir(parents=True, exist_ok=True)
        # Realistic README with a heading and a description paragraph.
        (dest / "README.md").write_text(
            "# myrepo\n\n"
            "Async Telegram Bot API framework for Python 3.7+ that "
            "supports both polling and webhook modes. Comes batteries-"
            "included with FSM, routers, and middleware.\n"
        )
        return 0, "", ""

    monkeypatch.setattr(tools, "_run", fake_run)

    asyncio.run(
        tools.dispatch_tool(
            "clone_repo",
            '{"url": "https://github.com/aiogram/aiogram"}',
            None,
            user_id=1,
        )
    )

    repos = storage.list_github_repos()
    assert len(repos) == 1
    repo = repos[0]
    assert repo["name"] == "aiogram"
    assert repo["url"] == "https://github.com/aiogram/aiogram"
    assert "Async Telegram Bot API framework" in repo["description"]
    assert repo["cloned_at"] > 0


def test_clone_without_readme_leaves_description_empty(fresh, monkeypatch) -> None:
    tools = fresh["tools"]
    storage = fresh["storage"]
    import asyncio

    async def fake_run(cmd, cwd=None, timeout=60):
        Path(cmd[-1]).mkdir(parents=True, exist_ok=True)
        return 0, "", ""

    monkeypatch.setattr(tools, "_run", fake_run)
    asyncio.run(
        tools.dispatch_tool(
            "clone_repo",
            '{"url": "https://x.example/empty"}',
            None,
            user_id=1,
        )
    )
    repo = storage.get_github_repo("empty")
    assert repo is not None
    assert repo["description"] == ""


def test_clone_uses_package_json_when_no_readme(fresh, monkeypatch) -> None:
    tools = fresh["tools"]
    storage = fresh["storage"]
    import asyncio

    async def fake_run(cmd, cwd=None, timeout=60):
        dest = Path(cmd[-1])
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "package.json").write_text(
            '{"name": "x", "description": "A neat Node thing."}'
        )
        return 0, "", ""

    monkeypatch.setattr(tools, "_run", fake_run)
    asyncio.run(
        tools.dispatch_tool(
            "clone_repo",
            '{"url": "https://x.example/node-thing"}',
            None,
            user_id=2,
        )
    )
    repo = storage.get_github_repo("node-thing")
    assert repo is not None
    assert "neat Node thing" in repo["description"]


def test_forget_github_repo_removes_metadata(fresh, monkeypatch) -> None:
    tools = fresh["tools"]
    storage = fresh["storage"]
    import asyncio

    async def fake_run(cmd, cwd=None, timeout=60):
        Path(cmd[-1]).mkdir(parents=True, exist_ok=True)
        return 0, "", ""

    monkeypatch.setattr(tools, "_run", fake_run)
    asyncio.run(
        tools.dispatch_tool(
            "clone_repo",
            '{"url": "https://x/foo"}',
            None,
            user_id=1,
        )
    )
    assert storage.get_github_repo("foo") is not None

    assert storage.forget_github_repo("foo") is True
    # Stale entry is also auto-hidden if folder is deleted, but explicit
    # forget removes the metadata immediately.
    raw = storage._github_repos()
    assert "foo" not in raw


def test_delete_project_removes_disk_and_storage(fresh, monkeypatch) -> None:
    tools = fresh["tools"]
    storage = fresh["storage"]
    projects_dir = fresh["config"].PROJECTS_DIR
    import asyncio

    async def fake_run(cmd, cwd=None, timeout=60):
        Path(cmd[-1]).mkdir(parents=True, exist_ok=True)
        return 0, "", ""

    monkeypatch.setattr(tools, "_run", fake_run)
    asyncio.run(
        tools.dispatch_tool(
            "clone_repo",
            '{"url": "https://x/bye"}',
            None,
            user_id=1,
        )
    )
    repo_path = projects_dir / "bye"
    assert repo_path.exists()
    assert storage.get_github_repo("bye") is not None

    existed = tools.delete_project("bye")
    assert existed is True
    assert not repo_path.exists()
    assert storage.get_github_repo("bye") is None


def test_delete_project_rejects_traversal(fresh) -> None:
    tools = fresh["tools"]
    with pytest.raises(tools.ToolError):
        tools.delete_project("../escape")


def test_agent_system_prompt_includes_project_inventory(fresh, monkeypatch) -> None:
    tools = fresh["tools"]
    agent = fresh["agent"]
    import asyncio

    async def fake_run(cmd, cwd=None, timeout=60):
        dest = Path(cmd[-1])
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "README.md").write_text(
            "# Foo\n\nDescription of Foo project — a tool that does something cool.\n"
        )
        return 0, "", ""

    monkeypatch.setattr(tools, "_run", fake_run)
    asyncio.run(
        tools.dispatch_tool(
            "clone_repo",
            '{"url": "https://x/Foo"}',
            None,
            user_id=1,
        )
    )

    prompt = agent._build_system_prompt()
    assert "Foo" in prompt
    assert "Description of Foo project" in prompt
    assert "Active GitHub projects on this server" in prompt


def test_agent_prompt_reflects_auto_clone_toggle(fresh) -> None:
    storage = fresh["storage"]
    agent = fresh["agent"]

    storage.set_github_auto_clone(True)
    on_prompt = agent._build_system_prompt()
    assert "ENABLED" in on_prompt or "DO NOT tell them to run /clone" in on_prompt

    storage.set_github_auto_clone(False)
    off_prompt = agent._build_system_prompt()
    assert "DISABLED" in off_prompt
    assert "wait for an explicit instruction" in off_prompt


def test_capabilities_ensure_recorded_writes_file(fresh) -> None:
    capabilities = fresh["capabilities"]
    config = fresh["config"]

    target = config.DATA_DIR / "capabilities.md"
    assert not target.exists()
    capabilities.ensure_recorded()
    assert target.exists()
    text = target.read_text()
    # All 5 capability keys must be present after a fresh init.
    for key in (
        "github-auto-clone",
        "github-settings-screen",
        "menu-button-commands",
        "unified-thinking-style",
        "access-modes",
    ):
        assert f"<!-- key: {key} -->" in text


def test_capabilities_ensure_recorded_is_idempotent(fresh) -> None:
    capabilities = fresh["capabilities"]
    config = fresh["config"]

    capabilities.ensure_recorded()
    first = (config.DATA_DIR / "capabilities.md").read_text()
    capabilities.ensure_recorded()
    capabilities.ensure_recorded()
    second = (config.DATA_DIR / "capabilities.md").read_text()
    assert first == second


def test_agent_prompt_includes_capabilities_log(fresh) -> None:
    capabilities = fresh["capabilities"]
    agent = fresh["agent"]

    capabilities.ensure_recorded()
    prompt = agent._build_system_prompt()
    assert "Bot capabilities log" in prompt
    assert "Авто-клонирование GitHub" in prompt


def test_wizard_github_menu_renders_with_no_repos(fresh) -> None:
    """The wizard's _kb_github_menu / _github_menu_body must not crash
    on an empty state and must include the auto-toggle row."""
    sys.modules.pop("bot.wizard", None)
    wizard = importlib.import_module("bot.wizard")
    kb = wizard._kb_github_menu()
    body = wizard._github_menu_body()
    # 1 row: placeholder "📭 Пусто…" or repo list, 1 row: auto toggle,
    # 1 row: back. With no repos that's 3 rows.
    assert len(kb.inline_keyboard) == 3
    assert "ВКЛ" in kb.inline_keyboard[1][0].text or "ВЫКЛ" in kb.inline_keyboard[1][0].text
    assert "GitHub" in body
    assert "Авто-клонирование" in body


def test_wizard_github_menu_lists_cloned_repos(fresh, monkeypatch) -> None:
    tools = fresh["tools"]
    import asyncio

    async def fake_run(cmd, cwd=None, timeout=60):
        Path(cmd[-1]).mkdir(parents=True, exist_ok=True)
        return 0, "", ""

    monkeypatch.setattr(tools, "_run", fake_run)
    asyncio.run(
        tools.dispatch_tool(
            "clone_repo",
            '{"url": "https://x/alpha"}',
            None,
            user_id=1,
        )
    )
    asyncio.run(
        tools.dispatch_tool(
            "clone_repo",
            '{"url": "https://x/beta"}',
            None,
            user_id=1,
        )
    )

    sys.modules.pop("bot.wizard", None)
    wizard = importlib.import_module("bot.wizard")
    kb = wizard._kb_github_menu()
    # 2 repos + auto toggle + back = 4 rows.
    assert len(kb.inline_keyboard) == 4
    labels = [row[0].text for row in kb.inline_keyboard]
    assert any("alpha" in label for label in labels)
    assert any("beta" in label for label in labels)
