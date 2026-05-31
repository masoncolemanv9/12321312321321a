import asyncio
import json
from pathlib import Path

from .config import EXEC_TIMEOUT, MAX_FILE_BYTES, PROJECTS_DIR

# How many commits deep ``clone_repo`` pulls. Shallower = faster boot for
# large repos. The agent / user can still do ``/git fetch --unshallow``
# afterwards if they need full history.
_CLONE_DEPTH = 20

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files and directories at the given relative path inside the current project.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path inside the project. Use '.' for the project root.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file from the current project. Returns up to ~200KB of content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to the file."}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write or overwrite a text file inside the current project.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to the file."},
                    "content": {"type": "string", "description": "Full new content of the file."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "exec_bash",
            "description": (
                "Run a shell command in the current project directory. "
                f"Timeout {EXEC_TIMEOUT}s. Returns stdout+stderr+exit code."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run."}
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "clone_repo",
            "description": (
                "Clone a git repository (e.g. a GitHub URL) into the bot's "
                "projects folder and make it the active project. After "
                "cloning, follow-up tool calls (list_dir, read_file, "
                f"exec_bash) target the new repo. Uses --depth {_CLONE_DEPTH}. "
                "Call this whenever the user shares a GitHub / GitLab / "
                "Bitbucket URL and asks you to look at the code, instead "
                "of telling them to run /clone."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": (
                            "HTTPS or SSH git URL "
                            "(e.g. https://github.com/owner/repo[.git])."
                        ),
                    },
                    "name": {
                        "type": "string",
                        "description": (
                            "Optional local folder name. Defaults to the "
                            "last path segment of the URL with .git stripped."
                        ),
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_projects",
            "description": (
                "List git projects that have already been cloned into the "
                "bot's projects folder. Use this when the user asks "
                '"что у меня есть в проектах" or wants to know which '
                "repos are available before switching."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "switch_project",
            "description": (
                "Switch the active project to an already-cloned repo. "
                "Subsequent file / shell tool calls target the new project. "
                "If the project does not exist, returns an error and you "
                "should clone it with clone_repo instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": (
                            "Folder name of the project (as shown by "
                            "list_projects)."
                        ),
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tavily_search",
            "description": (
                "Search the web with Tavily (AI-tuned search API). Returns "
                "ranked results with snippets plus a one-sentence answer "
                "when Tavily produces one. Use this when the user asks "
                "for recent info, news, or any general lookup that isn't "
                "in the current project. Requires a Tavily API key set "
                "in /setup → Внешние API."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query in natural language.",
                    },
                    "depth": {
                        "type": "string",
                        "enum": ["basic", "advanced"],
                        "description": (
                            "'basic' (default, fast) or 'advanced' (slower, "
                            "deeper crawl)."
                        ),
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "1–10. Defaults to 5.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "firecrawl_scrape",
            "description": (
                "Fetch a webpage and return its main content as clean "
                "markdown via Firecrawl. Use this when the user gives a "
                "URL and asks you to read / summarise / quote from it, "
                "or when a search result looks promising and you want "
                "the full page text. Requires a Firecrawl API key in "
                "/setup → Внешние API."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Absolute https:// URL of the page to scrape.",
                    }
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "brave_search",
            "description": (
                "Web search via Brave Search API. Use as an alternative "
                "to tavily_search when you want raw Google-like results "
                "without AI re-ranking, or as a fallback if Tavily is "
                "down. Requires a Brave Search API key in /setup → "
                "Внешние API."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "count": {
                        "type": "integer",
                        "description": "1–20. Defaults to 10.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "exa_search",
            "description": (
                "Semantic web search via Exa (formerly Metaphor). Returns "
                "pages that are *similar in meaning* to the query, not "
                "just keyword matches. Use for 'find articles like X' or "
                "'find startups that do Y' kinds of questions. Requires "
                "an Exa API key in /setup → Внешние API."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural-language query."},
                    "num_results": {
                        "type": "integer",
                        "description": "1–20. Defaults to 10.",
                    },
                    "search_type": {
                        "type": "string",
                        "enum": ["neural", "keyword", "auto"],
                        "description": (
                            "'neural' (semantic, default), 'keyword' "
                            "(traditional), 'auto' (let Exa pick)."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apify_run_actor",
            "description": (
                "Run an Apify actor synchronously and return the dataset "
                "items it produced. Use this for scraping platforms that "
                "Tavily/Brave can't reach well (Reddit, TikTok, YouTube, "
                "Twitter/X, Instagram, ...). The user must already know "
                "the actor's slug (e.g. 'apify/instagram-scraper') and "
                "what input it expects. Requires an Apify API token in "
                "/setup → Внешние API."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "actor_id": {
                        "type": "string",
                        "description": (
                            "Apify actor identifier, e.g. "
                            "'apify/instagram-scraper' or "
                            "'clockworks/free-tiktok-scraper'."
                        ),
                    },
                    "run_input": {
                        "type": "object",
                        "description": (
                            "Actor-specific input JSON. See the actor's "
                            "page on apify.com for the expected schema."
                        ),
                    },
                },
                "required": ["actor_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_search_code",
            "description": (
                "Search GitHub code via the /search/code API. Useful for "
                "finding examples of a function call, a config option, or "
                "any code snippet across public repos. Query syntax: "
                "github.com/search docs. Requires a GitHub PAT in /setup "
                "→ Внешние API (also lifts the anonymous 60-req/hour limit "
                "to 5000)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Code search query, e.g. "
                            "'asyncio.create_subprocess_exec language:python'."
                        ),
                    },
                    "per_page": {
                        "type": "integer",
                        "description": "1–20. Defaults to 10.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_get_file",
            "description": (
                "Fetch a single file's text content from a public GitHub "
                "repo via the contents API. Use this for 'show me the "
                "README of X' / 'what's in setup.py of Y' kinds of "
                "questions when you don't want to clone the whole repo. "
                "Requires a GitHub PAT in /setup → Внешние API."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {
                        "type": "string",
                        "description": "GitHub repo as 'owner/name'.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Path inside the repo, e.g. 'README.md'.",
                    },
                    "ref": {
                        "type": "string",
                        "description": (
                            "Optional branch / tag / commit SHA. Defaults "
                            "to the repo's default branch."
                        ),
                    },
                },
                "required": ["repo", "path"],
            },
        },
    },
]


class ToolError(Exception):
    pass


def project_root_for(cwd: Path) -> Path:
    """Return the project root for ``cwd`` (the direct child of PROJECTS_DIR).

    If ``cwd`` already is a project root, returns it. If ``cwd`` is somewhere
    deeper (after ``/cd subdir``) walks up to the immediate child of
    ``PROJECTS_DIR``. Falls back to ``cwd`` if no PROJECTS_DIR ancestor is
    found.
    """
    for candidate in [cwd, *cwd.parents]:
        if candidate.parent == PROJECTS_DIR:
            return candidate
    return cwd


def _resolve(cwd: Path, rel: str) -> Path:
    if not cwd:
        raise ToolError("No project selected. Use /clone or /project first.")
    target = (cwd / rel).resolve()
    root = project_root_for(cwd).resolve()
    if root not in target.parents and target != root:
        raise ToolError(f"path '{rel}' escapes project '{root.name}'")
    return target


async def _run(cmd: list[str], cwd: Path | None = None, timeout: int = EXEC_TIMEOUT) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd) if cwd else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return 124, "", f"timeout after {timeout}s"
    return proc.returncode or 0, stdout.decode(errors="replace"), stderr.decode(errors="replace")


async def _run_shell(command: str, cwd: Path | None, timeout: int = EXEC_TIMEOUT) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_shell(
        command,
        cwd=str(cwd) if cwd else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return 124, "", f"timeout after {timeout}s"
    return proc.returncode or 0, stdout.decode(errors="replace"), stderr.decode(errors="replace")


async def list_dir(cwd: Path, path: str) -> str:
    target = _resolve(cwd, path or ".")
    if not target.exists():
        raise ToolError(f"path '{path}' does not exist")
    if target.is_file():
        return f"{path} is a file ({target.stat().st_size} bytes)"
    items = []
    for child in sorted(target.iterdir()):
        kind = "d" if child.is_dir() else "f"
        size = child.stat().st_size if child.is_file() else 0
        items.append(f"{kind} {child.name}{' (' + str(size) + ')' if kind == 'f' else '/'}")
    return "\n".join(items) if items else "(empty)"


async def read_file(cwd: Path, path: str) -> str:
    target = _resolve(cwd, path)
    if not target.exists() or not target.is_file():
        raise ToolError(f"file '{path}' not found")
    data = target.read_bytes()
    if len(data) > MAX_FILE_BYTES:
        return data[:MAX_FILE_BYTES].decode(errors="replace") + f"\n\n[...truncated, total {len(data)} bytes]"
    return data.decode(errors="replace")


async def write_file(cwd: Path, path: str, content: str) -> str:
    target = _resolve(cwd, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return f"wrote {len(content)} chars to {path}"


async def exec_bash(cwd: Path, command: str, timeout: int | None = None) -> str:
    """Run ``command`` in ``cwd`` and return formatted stdout/stderr + exit code.

    ``timeout`` overrides the global ``EXEC_TIMEOUT`` — pass a larger value
    for batched terminal flows like ``/work`` where the 10-minute default
    isn't enough (e.g. ``pip install``, ``playwright install chromium``).
    """
    if not cwd:
        raise ToolError("No project selected. Use /clone or /project first.")
    effective_timeout = timeout if timeout is not None else EXEC_TIMEOUT
    code, out, err = await _run_shell(command, cwd=cwd, timeout=effective_timeout)
    parts = []
    if out:
        parts.append(f"--- stdout ---\n{out}")
    if err:
        parts.append(f"--- stderr ---\n{err}")
    parts.append(f"exit code: {code}")
    return "\n".join(parts)


def _extract_repo_description(repo_root: Path) -> str:
    """Pull a one-line description out of a freshly-cloned repo.

    Tries, in order:
      1. The first non-empty, non-heading paragraph of README.md /
         README.rst / README (case-insensitive).
      2. ``description`` field of ``package.json`` / ``pyproject.toml``
         (only if README didn't yield anything useful).
      3. Empty string.

    Truncated to ~280 chars so it fits in a Telegram message bubble
    without scrolling.
    """
    candidates = [
        "README.md",
        "README.MD",
        "Readme.md",
        "readme.md",
        "README.rst",
        "README",
        "readme",
    ]
    for name in candidates:
        path = repo_root / name
        if not path.is_file():
            continue
        try:
            text = path.read_text(errors="replace")
        except Exception:  # noqa: BLE001
            continue
        for raw in text.split("\n\n"):
            chunk = raw.strip()
            if not chunk:
                continue
            # Skip pure-heading paragraphs ("# Foo", "## Bar", "===").
            stripped = chunk.lstrip("#").strip()
            if not stripped:
                continue
            if set(stripped) <= set("=-~"):
                continue
            # Collapse whitespace, strip markdown link-syntax decorations.
            flat = " ".join(stripped.split())
            if len(flat) < 20:
                # Too short to be a description; might be a badge row.
                continue
            return flat[:280].rstrip()
    # Fallback: package.json description.
    pkg = repo_root / "package.json"
    if pkg.is_file():
        try:
            data = json.loads(pkg.read_text(errors="replace"))
            desc = str(data.get("description", "")).strip()
            if desc:
                return desc[:280]
        except Exception:  # noqa: BLE001
            pass
    return ""


async def clone_repo(url: str, dest_name: str | None = None) -> Path:
    if not dest_name:
        dest_name = url.rstrip("/").split("/")[-1].removesuffix(".git")
    if not dest_name or dest_name in ("..", "."):
        raise ToolError(f"invalid project name '{dest_name}'")
    if "/" in dest_name or "\\" in dest_name:
        raise ToolError(f"invalid project name '{dest_name}' (no slashes)")
    dest = PROJECTS_DIR / dest_name
    if dest.exists():
        raise ToolError(
            f"project '{dest_name}' already exists. Use switch_project to "
            "select it instead of cloning again."
        )
    code, out, err = await _run(
        ["git", "clone", "--depth", str(_CLONE_DEPTH), url, str(dest)],
        timeout=120,
    )
    if code != 0:
        raise ToolError(f"git clone failed: {err.strip() or out.strip()}")

    # Persist a one-line description + the source URL so the bot's
    # GitHub settings screen can show it later, and so the agent's
    # system prompt can list "what repos do I have" without reading
    # READMEs every turn. Best-effort: a failure here doesn't undo
    # the clone.
    try:
        from .storage import storage

        description = _extract_repo_description(dest)
        storage.set_github_repo_meta(dest_name, url, description=description)
    except Exception:  # noqa: BLE001
        pass

    return dest


def list_projects() -> list[str]:
    if not PROJECTS_DIR.exists():
        return []
    return sorted(p.name for p in PROJECTS_DIR.iterdir() if p.is_dir())


def switch_project(name: str) -> Path:
    if not name or name in ("..", "."):
        raise ToolError(f"invalid project name '{name}'")
    if "/" in name or "\\" in name:
        raise ToolError(f"invalid project name '{name}' (no slashes)")
    target = PROJECTS_DIR / name
    if not target.exists() or not target.is_dir():
        raise ToolError(
            f"project '{name}' not found. Call list_projects to see what's "
            "available, or clone_repo to fetch a new one."
        )
    return target


def delete_project(name: str) -> bool:
    """Remove a cloned project from disk AND from storage metadata.

    Returns True iff the on-disk folder existed and was removed. The
    storage metadata is forgotten regardless (so stale entries get
    cleaned up too). Defensively rejects names containing slashes or
    traversal segments so we never ``rm -rf`` outside ``PROJECTS_DIR``.
    """
    import shutil

    if not name or name in ("..", "."):
        raise ToolError(f"invalid project name '{name}'")
    if "/" in name or "\\" in name:
        raise ToolError(f"invalid project name '{name}' (no slashes)")
    target = (PROJECTS_DIR / name).resolve()
    if PROJECTS_DIR.resolve() not in target.parents:
        raise ToolError(f"refusing to delete outside projects dir: {target}")

    existed = target.exists() and target.is_dir()
    if existed:
        shutil.rmtree(target, ignore_errors=True)
    try:
        from .storage import storage

        storage.forget_github_repo(name)
    except Exception:  # noqa: BLE001
        pass
    return existed


async def dispatch_tool(
    name: str,
    args_json: str,
    cwd: Path | None,
    *,
    user_id: int | None = None,
) -> str:
    """Run a tool call.

    For tools that operate on the active project (``list_dir`` /
    ``read_file`` / ``write_file`` / ``exec_bash``), ``cwd`` must be a
    real path inside ``PROJECTS_DIR``. ``clone_repo`` /
    ``list_projects`` / ``switch_project`` do NOT require ``cwd`` —
    they're the way to acquire one. ``clone_repo`` and
    ``switch_project`` additionally update the per-user active project
    in storage so that *subsequent* tool calls in the same agent turn
    see the new ``cwd``.
    """
    try:
        args = json.loads(args_json or "{}")
    except json.JSONDecodeError as exc:
        raise ToolError(f"bad arguments json: {exc}") from exc

    # Project-management tools: no cwd required.
    if name == "clone_repo":
        url = args.get("url") or ""
        if not url:
            raise ToolError("clone_repo requires 'url'")
        dest = await clone_repo(url, args.get("name"))
        # Update active project for this user so subsequent tool calls
        # in the same turn see the new cwd.
        if user_id is not None:
            try:
                from .storage import storage

                storage.set_cwd(user_id, dest)
            except Exception:  # noqa: BLE001
                # Don't fail the clone just because storage write didn't
                # land — the user can /project <name> manually.
                pass
        return (
            f"Cloned {url} into '{dest.name}'. This is now the active "
            "project; use list_dir/read_file/exec_bash to explore it."
        )

    if name == "list_projects":
        projects = list_projects()
        if not projects:
            return (
                "No projects yet. Use clone_repo(url=...) to fetch one "
                "from a git URL."
            )
        return "Available projects:\n- " + "\n- ".join(projects)

    if name == "switch_project":
        proj_name = args.get("name") or ""
        if not proj_name:
            raise ToolError("switch_project requires 'name'")
        dest = switch_project(proj_name)
        if user_id is not None:
            try:
                from .storage import storage

                storage.set_cwd(user_id, dest)
            except Exception:  # noqa: BLE001
                pass
        return (
            f"Switched active project to '{dest.name}'. Subsequent "
            "list_dir/read_file/exec_bash calls target this project."
        )

    # External "research API" tools — no cwd required, all go through
    # ``bot/external_tools.py``. We import lazily so this module stays
    # importable in test environments that don't have httpx wired up.
    if name in {
        "tavily_search",
        "firecrawl_scrape",
        "brave_search",
        "exa_search",
        "apify_run_actor",
        "github_search_code",
        "github_get_file",
    }:
        from . import external_tools as ext

        if name == "tavily_search":
            return await ext.tavily_search(
                args.get("query") or "",
                depth=args.get("depth") or "basic",
                max_results=int(args.get("max_results") or 5),
            )
        if name == "firecrawl_scrape":
            return await ext.firecrawl_scrape(args.get("url") or "")
        if name == "brave_search":
            return await ext.brave_search(
                args.get("query") or "",
                count=int(args.get("count") or 10),
            )
        if name == "exa_search":
            return await ext.exa_search(
                args.get("query") or "",
                num_results=int(args.get("num_results") or 10),
                search_type=args.get("search_type") or "neural",
            )
        if name == "apify_run_actor":
            return await ext.apify_run_actor(
                args.get("actor_id") or "",
                run_input=args.get("run_input") or {},
            )
        if name == "github_search_code":
            return await ext.github_search_code(
                args.get("query") or "",
                per_page=int(args.get("per_page") or 10),
            )
        if name == "github_get_file":
            return await ext.github_get_file(
                args.get("repo") or "",
                args.get("path") or "",
                ref=args.get("ref"),
            )

    # All other tools work inside a project — require a cwd.
    if cwd is None:
        raise ToolError(
            "No project selected. Use clone_repo(url=...) for a new repo "
            "or switch_project(name=...) for an already-cloned one."
        )

    if name == "list_dir":
        return await list_dir(cwd, args.get("path", "."))
    if name == "read_file":
        return await read_file(cwd, args["path"])
    if name == "write_file":
        return await write_file(cwd, args["path"], args.get("content", ""))
    if name == "exec_bash":
        return await exec_bash(cwd, args["command"])
    raise ToolError(f"unknown tool: {name}")
