"""HTTP wrappers for the «🛠 Внешние API» integrations.

Each function takes the LLM-supplied arguments, reads the relevant API
key from ``storage.get_external_tool_key(...)``, performs the HTTP call,
and returns a plain-text string the LLM can paste into its reasoning.

Design notes
------------
* All functions are ``async`` and use :mod:`httpx` (already a transitive
  dependency via ``openai``) — no new wheel to install.
* They never raise on missing API key — they return a friendly message
  asking the user to set it. That way the LLM can apologise to the user
  instead of the whole tool-call failing with a 500.
* Response shape is always a string (truncated to ~4 KB) because the
  agent loop wraps the result in a ``role=tool`` message and we don't
  want to blow the context window. Network/transport errors are caught
  and surfaced as ``ERROR: ...`` strings on the same channel.
* We don't try to be clever about pagination — the LLM can call again
  with a more specific query if it needs more results.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from .storage import storage

logger = logging.getLogger(__name__)

# Conservative timeouts — these tools sit between the user's chat and an
# external network. If a provider is unreachable, we want to give up
# quickly so the agent can apologise instead of hanging the TG reply.
_HTTP_TIMEOUT = httpx.Timeout(connect=5.0, read=20.0, write=10.0, pool=5.0)

# Maximum characters we paste back into the model's context per tool
# call. The agent already caps tool results at 8000 chars; we cap a
# little lower so headers/footers fit.
_MAX_RESULT_CHARS = 6000


def _missing_key_msg(tool: str, url: str) -> str:
    return (
        f"API ключ {tool!r} не задан. Открой /setup → 🛠 Внешние API → {tool}, "
        f"вставь ключ (получить тут: {url}), и спроси меня ещё раз."
    )


def _truncate(text: str, *, limit: int = _MAX_RESULT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + f"\n... (truncated, {len(text) - limit} more chars)"


def _format_results(
    *,
    header: str,
    items: list[dict[str, Any]],
    title_keys: tuple[str, ...] = ("title",),
    url_keys: tuple[str, ...] = ("url",),
    body_keys: tuple[str, ...] = ("content", "snippet", "description", "text"),
) -> str:
    """Render a flat search-result list as markdown-ish text for the LLM."""
    if not items:
        return f"{header}\n(нет результатов)"
    lines: list[str] = [header]
    for i, item in enumerate(items, start=1):
        title = next(
            (str(item.get(k)) for k in title_keys if item.get(k)),
            "(без названия)",
        )
        url = next((str(item.get(k)) for k in url_keys if item.get(k)), "")
        body = next((str(item.get(k)) for k in body_keys if item.get(k)), "")
        head = f"{i}. {title}"
        if url:
            head += f" — {url}"
        lines.append(head)
        if body:
            # Indent the body a little so the LLM can tell title vs blurb.
            body = body.strip().replace("\n", " ")
            if len(body) > 500:
                body = body[:500].rstrip() + "…"
            lines.append(f"   {body}")
    return _truncate("\n".join(lines))


# ─── Tavily ───────────────────────────────────────────────────────────


async def tavily_search(query: str, *, depth: str = "basic", max_results: int = 5) -> str:
    """Run a Tavily ``/search`` call. Returns formatted results text."""
    key = storage.get_external_tool_key("tavily")
    if not key:
        return _missing_key_msg("tavily", "https://app.tavily.com/home")
    if not query.strip():
        return "ERROR: tavily_search requires a non-empty 'query'."
    depth = depth if depth in ("basic", "advanced") else "basic"
    max_results = max(1, min(int(max_results or 5), 10))
    payload = {
        "api_key": key,
        "query": query,
        "search_depth": depth,
        "max_results": max_results,
        "include_answer": True,
    }
    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            resp = await client.post("https://api.tavily.com/search", json=payload)
    except httpx.HTTPError as exc:
        logger.warning("tavily transport error: %s", exc)
        return f"ERROR: tavily network error: {exc}"
    if resp.status_code >= 400:
        return f"ERROR: tavily HTTP {resp.status_code}: {resp.text[:300]}"
    try:
        data = resp.json()
    except ValueError:
        return f"ERROR: tavily returned non-JSON: {resp.text[:300]}"
    answer = (data.get("answer") or "").strip()
    items = data.get("results") or []
    header = f"Tavily search for: {query!r}"
    if answer:
        header += f"\nQuick answer: {answer}"
    return _format_results(
        header=header,
        items=items,
        body_keys=("content", "snippet"),
    )


# ─── Firecrawl ────────────────────────────────────────────────────────


async def firecrawl_scrape(url: str) -> str:
    """Scrape ``url`` via Firecrawl and return the markdown body."""
    key = storage.get_external_tool_key("firecrawl")
    if not key:
        return _missing_key_msg("firecrawl", "https://firecrawl.dev/app/api-keys")
    if not url.strip():
        return "ERROR: firecrawl_scrape requires a non-empty 'url'."
    headers = {"Authorization": f"Bearer {key}"}
    payload = {"url": url, "formats": ["markdown"]}
    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            resp = await client.post(
                "https://api.firecrawl.dev/v1/scrape",
                json=payload,
                headers=headers,
            )
    except httpx.HTTPError as exc:
        logger.warning("firecrawl transport error: %s", exc)
        return f"ERROR: firecrawl network error: {exc}"
    if resp.status_code >= 400:
        return f"ERROR: firecrawl HTTP {resp.status_code}: {resp.text[:300]}"
    try:
        data = resp.json()
    except ValueError:
        return f"ERROR: firecrawl returned non-JSON: {resp.text[:300]}"
    if not data.get("success", True):
        return f"ERROR: firecrawl reported failure: {data.get('error') or data}"
    body = (data.get("data") or {}).get("markdown") or ""
    meta = (data.get("data") or {}).get("metadata") or {}
    title = meta.get("title") or url
    head = f"Firecrawl scrape: {title}\nURL: {url}\n"
    return _truncate(head + "\n" + body.strip())


# ─── Brave Search ─────────────────────────────────────────────────────


async def brave_search(query: str, *, count: int = 10) -> str:
    """Run a Brave Search ``/web/search`` and format the results."""
    key = storage.get_external_tool_key("brave")
    if not key:
        return _missing_key_msg(
            "brave", "https://api-dashboard.search.brave.com/app/keys"
        )
    if not query.strip():
        return "ERROR: brave_search requires a non-empty 'query'."
    count = max(1, min(int(count or 10), 20))
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": key,
    }
    params = {"q": query, "count": count}
    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            resp = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers=headers,
                params=params,
            )
    except httpx.HTTPError as exc:
        logger.warning("brave transport error: %s", exc)
        return f"ERROR: brave network error: {exc}"
    if resp.status_code >= 400:
        return f"ERROR: brave HTTP {resp.status_code}: {resp.text[:300]}"
    try:
        data = resp.json()
    except ValueError:
        return f"ERROR: brave returned non-JSON: {resp.text[:300]}"
    items = ((data.get("web") or {}).get("results")) or []
    return _format_results(
        header=f"Brave web search for: {query!r}",
        items=items,
        body_keys=("description",),
    )


# ─── Exa ──────────────────────────────────────────────────────────────


async def exa_search(
    query: str, *, num_results: int = 10, search_type: str = "neural"
) -> str:
    """Run an Exa ``/search`` (semantic). Returns formatted results."""
    key = storage.get_external_tool_key("exa")
    if not key:
        return _missing_key_msg("exa", "https://dashboard.exa.ai/api-keys")
    if not query.strip():
        return "ERROR: exa_search requires a non-empty 'query'."
    num_results = max(1, min(int(num_results or 10), 20))
    search_type = search_type if search_type in ("neural", "keyword", "auto") else "neural"
    headers = {"x-api-key": key, "Content-Type": "application/json"}
    payload = {"query": query, "numResults": num_results, "type": search_type}
    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            resp = await client.post(
                "https://api.exa.ai/search", json=payload, headers=headers
            )
    except httpx.HTTPError as exc:
        logger.warning("exa transport error: %s", exc)
        return f"ERROR: exa network error: {exc}"
    if resp.status_code >= 400:
        return f"ERROR: exa HTTP {resp.status_code}: {resp.text[:300]}"
    try:
        data = resp.json()
    except ValueError:
        return f"ERROR: exa returned non-JSON: {resp.text[:300]}"
    items = data.get("results") or []
    return _format_results(
        header=f"Exa {search_type} search for: {query!r}",
        items=items,
        body_keys=("text", "summary"),
    )


# ─── Apify ────────────────────────────────────────────────────────────


async def apify_run_actor(actor_id: str, run_input: dict | None = None) -> str:
    """Run an Apify actor synchronously and return the first items.

    ``actor_id`` is the slug like ``"clockworks/free-tiktok-scraper"`` or
    the numeric id. ``run_input`` is the actor-specific input JSON.

    We hit the ``run-sync-get-dataset-items`` endpoint so the bot
    doesn't have to poll a run-id; the actor must finish within
    Apify's sync timeout (~5 min) — which is also longer than our HTTP
    timeout, so very long actors will time out client-side. The user
    can re-run with smaller inputs in that case.
    """
    key = storage.get_external_tool_key("apify")
    if not key:
        return _missing_key_msg(
            "apify", "https://console.apify.com/account/integrations"
        )
    if not actor_id or not isinstance(actor_id, str):
        return "ERROR: apify_run_actor requires a non-empty 'actor_id' string."
    # Apify accepts actor IDs either as ``user~name`` or ``user/name``.
    # The path form requires the tilde. Normalise here so the LLM can
    # pass whatever feels natural.
    actor_path = actor_id.replace("/", "~")
    url = f"https://api.apify.com/v2/acts/{actor_path}/run-sync-get-dataset-items"
    params = {"token": key, "format": "json"}
    body = run_input or {}
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
            resp = await client.post(url, params=params, json=body)
    except httpx.HTTPError as exc:
        logger.warning("apify transport error: %s", exc)
        return f"ERROR: apify network error: {exc}"
    if resp.status_code >= 400:
        return f"ERROR: apify HTTP {resp.status_code}: {resp.text[:500]}"
    try:
        items = resp.json()
    except ValueError:
        return f"ERROR: apify returned non-JSON: {resp.text[:300]}"
    if not isinstance(items, list):
        # Some endpoints return ``{"data": {...}}`` on error.
        return f"ERROR: apify unexpected payload: {str(items)[:300]}"
    header = f"Apify actor {actor_id!r} → {len(items)} item(s)"
    if not items:
        return header + "\n(пустой dataset)"
    # Show the first 3 items in pretty JSON, truncated.
    preview_items = items[:3]
    rendered = json.dumps(preview_items, ensure_ascii=False, indent=2)
    tail = ""
    if len(items) > len(preview_items):
        tail = f"\n... ({len(items) - len(preview_items)} more items omitted)"
    return _truncate(f"{header}\n{rendered}{tail}")


# ─── GitHub ───────────────────────────────────────────────────────────


def _github_headers(pat: str) -> dict[str, str]:
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {pat}",
        "X-GitHub-Api-Version": "2022-11-28",
    }


async def github_search_code(query: str, *, per_page: int = 10) -> str:
    """Search GitHub code via the ``/search/code`` endpoint.

    Requires an authenticated PAT — anonymous calls to this endpoint
    are forbidden by GitHub. The query syntax follows the docs at
    https://docs.github.com/en/search-github/searching-on-github/searching-code .
    """
    key = storage.get_external_tool_key("github_pat")
    if not key:
        return _missing_key_msg("github_pat", "https://github.com/settings/tokens")
    if not query.strip():
        return "ERROR: github_search_code requires a non-empty 'query'."
    per_page = max(1, min(int(per_page or 10), 20))
    params = {"q": query, "per_page": per_page}
    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            resp = await client.get(
                "https://api.github.com/search/code",
                headers=_github_headers(key),
                params=params,
            )
    except httpx.HTTPError as exc:
        logger.warning("github transport error: %s", exc)
        return f"ERROR: github network error: {exc}"
    if resp.status_code >= 400:
        return f"ERROR: github HTTP {resp.status_code}: {resp.text[:300]}"
    try:
        data = resp.json()
    except ValueError:
        return f"ERROR: github returned non-JSON: {resp.text[:300]}"
    items = data.get("items") or []
    formatted: list[dict[str, Any]] = []
    for it in items:
        repo = (it.get("repository") or {}).get("full_name") or ""
        formatted.append({
            "title": f"{repo} :: {it.get('path') or it.get('name') or ''}",
            "url": it.get("html_url") or "",
            "description": "",
        })
    return _format_results(
        header=f"GitHub code search for: {query!r}",
        items=formatted,
        body_keys=("description",),
    )


async def github_get_file(repo: str, path: str, *, ref: str | None = None) -> str:
    """Fetch a file from a GitHub repo via the contents API.

    ``repo`` is ``owner/name``. ``path`` is the path inside the repo.
    ``ref`` is an optional branch / tag / sha; omitted = default branch.
    Returns the decoded file content (truncated). Binary files are
    refused with an ERROR string — the LLM should call list_dir-like
    tools instead.
    """
    key = storage.get_external_tool_key("github_pat")
    if not key:
        return _missing_key_msg("github_pat", "https://github.com/settings/tokens")
    if "/" not in repo:
        return "ERROR: github_get_file requires 'repo' like 'owner/name'."
    if not path:
        return "ERROR: github_get_file requires a non-empty 'path'."
    url = f"https://api.github.com/repos/{repo}/contents/{path.lstrip('/')}"
    params = {"ref": ref} if ref else None
    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            resp = await client.get(url, headers=_github_headers(key), params=params)
    except httpx.HTTPError as exc:
        logger.warning("github transport error: %s", exc)
        return f"ERROR: github network error: {exc}"
    if resp.status_code >= 400:
        return f"ERROR: github HTTP {resp.status_code}: {resp.text[:300]}"
    try:
        data = resp.json()
    except ValueError:
        return f"ERROR: github returned non-JSON: {resp.text[:300]}"
    encoding = data.get("encoding")
    if encoding != "base64":
        return f"ERROR: github returned unexpected encoding: {encoding!r}"
    import base64

    try:
        raw = base64.b64decode(data.get("content") or "")
    except (ValueError, TypeError) as exc:
        return f"ERROR: github base64 decode failed: {exc}"
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return (
            f"ERROR: file appears to be binary ({len(raw)} bytes) — "
            "the LLM can't usefully consume it via github_get_file."
        )
    head = f"GitHub: {repo}/{path}" + (f" @ {ref}" if ref else "")
    return _truncate(f"{head}\n\n{text}")
