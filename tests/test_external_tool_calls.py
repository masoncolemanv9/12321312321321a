"""Tests for the HTTP wrappers in :mod:`bot.external_tools`.

We mock ``httpx.AsyncClient`` at the module level so no real network
traffic happens. Each test verifies:

* Missing API key → friendly «set it in /setup» message (not a crash).
* Happy path → response is parsed and formatted.
* HTTP error → ``ERROR: ...`` string is returned.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

import bot.external_tools as ext
from bot.storage import Storage


@pytest.fixture
def fresh_storage(tmp_path: Path, monkeypatch) -> Storage:
    """A clean Storage pointed at tmp_path, swapped into external_tools."""
    s = Storage(data_dir=tmp_path)
    monkeypatch.setattr(ext, "storage", s)
    return s


class _FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        json_body: Any = None,
        text_body: str = "",
    ) -> None:
        self.status_code = status_code
        self._json = json_body
        self.text = text_body or (json.dumps(json_body) if json_body is not None else "")

    def json(self) -> Any:
        if self._json is None:
            raise ValueError("not json")
        return self._json


class _FakeClient:
    """Stand-in for ``httpx.AsyncClient`` used as an async context manager.

    Records the last request so tests can assert on URL / payload, and
    returns a pre-canned ``_FakeResponse``.
    """

    def __init__(self, response: _FakeResponse) -> None:
        self.response = response
        self.last_post: tuple[str, dict[str, Any]] | None = None
        self.last_get: tuple[str, dict[str, Any]] | None = None

    async def __aenter__(self) -> "_FakeClient":
        return self

    async def __aexit__(self, *_exc) -> None:
        return None

    async def post(self, url: str, **kwargs: Any) -> _FakeResponse:
        self.last_post = (url, kwargs)
        return self.response

    async def get(self, url: str, **kwargs: Any) -> _FakeResponse:
        self.last_get = (url, kwargs)
        return self.response


def _patch_client(monkeypatch, response: _FakeResponse) -> _FakeClient:
    """Patch :class:`httpx.AsyncClient` used by ``external_tools``."""
    fake = _FakeClient(response)
    monkeypatch.setattr(
        ext.httpx, "AsyncClient", lambda *args, **kwargs: fake
    )
    return fake


# ─── Missing-key behaviour ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_tavily_missing_key(fresh_storage) -> None:
    out = await ext.tavily_search("any query")
    assert "не задан" in out.lower() and "tavily" in out.lower()


@pytest.mark.asyncio
async def test_firecrawl_missing_key(fresh_storage) -> None:
    out = await ext.firecrawl_scrape("https://example.com")
    assert "не задан" in out.lower() and "firecrawl" in out.lower()


@pytest.mark.asyncio
async def test_brave_missing_key(fresh_storage) -> None:
    out = await ext.brave_search("anything")
    assert "не задан" in out.lower() and "brave" in out.lower()


@pytest.mark.asyncio
async def test_exa_missing_key(fresh_storage) -> None:
    out = await ext.exa_search("anything")
    assert "не задан" in out.lower() and "exa" in out.lower()


@pytest.mark.asyncio
async def test_apify_missing_key(fresh_storage) -> None:
    out = await ext.apify_run_actor("apify/instagram-scraper", {"x": 1})
    assert "не задан" in out.lower() and "apify" in out.lower()


@pytest.mark.asyncio
async def test_github_missing_key(fresh_storage) -> None:
    out = await ext.github_search_code("foo")
    assert "не задан" in out.lower()


# ─── Happy paths ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_tavily_happy_path(fresh_storage, monkeypatch) -> None:
    fresh_storage.set_external_tool_key("tavily", "tvly-test-aaaa")
    fake = _patch_client(
        monkeypatch,
        _FakeResponse(
            json_body={
                "answer": "The sky is blue.",
                "results": [
                    {
                        "title": "Why is the sky blue?",
                        "url": "https://example.com/sky",
                        "content": "Rayleigh scattering bends short wavelengths.",
                    }
                ],
            }
        ),
    )
    out = await ext.tavily_search("why is the sky blue", depth="advanced", max_results=3)
    assert fake.last_post is not None
    url, kwargs = fake.last_post
    assert url == "https://api.tavily.com/search"
    body = kwargs["json"]
    assert body["query"] == "why is the sky blue"
    assert body["search_depth"] == "advanced"
    assert body["max_results"] == 3
    assert body["api_key"] == "tvly-test-aaaa"
    assert "Tavily search for: 'why is the sky blue'" in out
    assert "Quick answer: The sky is blue." in out
    assert "Why is the sky blue?" in out


@pytest.mark.asyncio
async def test_firecrawl_happy_path(fresh_storage, monkeypatch) -> None:
    fresh_storage.set_external_tool_key("firecrawl", "fc-test-bbbb")
    fake = _patch_client(
        monkeypatch,
        _FakeResponse(
            json_body={
                "success": True,
                "data": {
                    "markdown": "# Hello\nThis is the page.",
                    "metadata": {"title": "Hello"},
                },
            }
        ),
    )
    out = await ext.firecrawl_scrape("https://example.com")
    assert fake.last_post is not None
    url, kwargs = fake.last_post
    assert url == "https://api.firecrawl.dev/v1/scrape"
    assert kwargs["headers"]["Authorization"] == "Bearer fc-test-bbbb"
    assert kwargs["json"]["url"] == "https://example.com"
    assert "# Hello" in out
    assert "Firecrawl scrape: Hello" in out


@pytest.mark.asyncio
async def test_brave_happy_path(fresh_storage, monkeypatch) -> None:
    fresh_storage.set_external_tool_key("brave", "brave-test-cccc")
    fake = _patch_client(
        monkeypatch,
        _FakeResponse(
            json_body={
                "web": {
                    "results": [
                        {
                            "title": "Example",
                            "url": "https://example.com",
                            "description": "Example site",
                        }
                    ]
                }
            }
        ),
    )
    out = await ext.brave_search("example", count=5)
    assert fake.last_get is not None
    url, kwargs = fake.last_get
    assert url == "https://api.search.brave.com/res/v1/web/search"
    assert kwargs["headers"]["X-Subscription-Token"] == "brave-test-cccc"
    assert kwargs["params"]["q"] == "example"
    assert kwargs["params"]["count"] == 5
    assert "Brave web search for: 'example'" in out
    assert "Example site" in out


@pytest.mark.asyncio
async def test_exa_happy_path(fresh_storage, monkeypatch) -> None:
    fresh_storage.set_external_tool_key("exa", "exa-test-dddd")
    fake = _patch_client(
        monkeypatch,
        _FakeResponse(
            json_body={
                "results": [
                    {
                        "title": "Cool startup",
                        "url": "https://cool.ai",
                        "text": "We do AI.",
                    }
                ]
            }
        ),
    )
    out = await ext.exa_search("AI startups", num_results=5, search_type="keyword")
    assert fake.last_post is not None
    url, kwargs = fake.last_post
    assert url == "https://api.exa.ai/search"
    assert kwargs["headers"]["x-api-key"] == "exa-test-dddd"
    body = kwargs["json"]
    assert body["query"] == "AI startups"
    assert body["numResults"] == 5
    assert body["type"] == "keyword"
    assert "Exa keyword search for: 'AI startups'" in out
    assert "Cool startup" in out


@pytest.mark.asyncio
async def test_apify_happy_path(fresh_storage, monkeypatch) -> None:
    fresh_storage.set_external_tool_key("apify", "apify_api_eeee")
    fake = _patch_client(
        monkeypatch,
        _FakeResponse(json_body=[{"foo": "bar"}, {"foo": "baz"}, {"foo": "qux"}, {"foo": "n"}]),
    )
    out = await ext.apify_run_actor(
        "apify/instagram-scraper", {"username": "openai"}
    )
    assert fake.last_post is not None
    url, kwargs = fake.last_post
    assert url == (
        "https://api.apify.com/v2/acts/apify~instagram-scraper/run-sync-get-dataset-items"
    )
    assert kwargs["params"]["token"] == "apify_api_eeee"
    assert kwargs["json"] == {"username": "openai"}
    assert "Apify actor 'apify/instagram-scraper' → 4 item(s)" in out
    assert "1 more items omitted" in out


@pytest.mark.asyncio
async def test_github_search_code_happy_path(fresh_storage, monkeypatch) -> None:
    fresh_storage.set_external_tool_key("github_pat", "ghp_test_ffff")
    fake = _patch_client(
        monkeypatch,
        _FakeResponse(
            json_body={
                "items": [
                    {
                        "name": "main.py",
                        "path": "src/main.py",
                        "html_url": "https://github.com/foo/bar/blob/main/src/main.py",
                        "repository": {"full_name": "foo/bar"},
                    }
                ]
            }
        ),
    )
    out = await ext.github_search_code("asyncio.create_subprocess_exec", per_page=5)
    assert fake.last_get is not None
    url, kwargs = fake.last_get
    assert url == "https://api.github.com/search/code"
    assert kwargs["headers"]["Authorization"] == "Bearer ghp_test_ffff"
    assert kwargs["params"]["q"] == "asyncio.create_subprocess_exec"
    assert kwargs["params"]["per_page"] == 5
    assert "foo/bar :: src/main.py" in out


@pytest.mark.asyncio
async def test_github_get_file_decodes_base64(fresh_storage, monkeypatch) -> None:
    fresh_storage.set_external_tool_key("github_pat", "ghp_test_gggg")
    content = base64.b64encode(b"hello world\n").decode("ascii")
    fake = _patch_client(
        monkeypatch,
        _FakeResponse(json_body={"encoding": "base64", "content": content}),
    )
    out = await ext.github_get_file("foo/bar", "README.md")
    assert fake.last_get is not None
    url, _kwargs = fake.last_get
    assert url == "https://api.github.com/repos/foo/bar/contents/README.md"
    assert "hello world" in out
    assert "GitHub: foo/bar/README.md" in out


# ─── Error surfacing ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_tavily_http_error(fresh_storage, monkeypatch) -> None:
    fresh_storage.set_external_tool_key("tavily", "tvly-test-aaaa")
    _patch_client(
        monkeypatch,
        _FakeResponse(status_code=401, text_body='{"detail":"unauthorized"}'),
    )
    out = await ext.tavily_search("x")
    assert out.startswith("ERROR: tavily HTTP 401")


@pytest.mark.asyncio
async def test_firecrawl_reports_failure(fresh_storage, monkeypatch) -> None:
    fresh_storage.set_external_tool_key("firecrawl", "fc-key")
    _patch_client(
        monkeypatch,
        _FakeResponse(json_body={"success": False, "error": "rate-limited"}),
    )
    out = await ext.firecrawl_scrape("https://example.com")
    assert "ERROR" in out and "rate-limited" in out


@pytest.mark.asyncio
async def test_github_get_file_rejects_binary(fresh_storage, monkeypatch) -> None:
    fresh_storage.set_external_tool_key("github_pat", "ghp_test")
    binary = base64.b64encode(b"\xff\xfe\x00\x01PNG\x00").decode("ascii")
    _patch_client(
        monkeypatch,
        _FakeResponse(json_body={"encoding": "base64", "content": binary}),
    )
    out = await ext.github_get_file("foo/bar", "logo.png")
    assert out.startswith("ERROR") and "binary" in out


@pytest.mark.asyncio
async def test_apify_invalid_payload(fresh_storage, monkeypatch) -> None:
    fresh_storage.set_external_tool_key("apify", "apify_api")
    _patch_client(monkeypatch, _FakeResponse(json_body={"data": "boom"}))
    out = await ext.apify_run_actor("foo/bar")
    assert out.startswith("ERROR")


# ─── Empty / edge-case input ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_tavily_empty_query_is_rejected(fresh_storage) -> None:
    fresh_storage.set_external_tool_key("tavily", "tvly-test")
    out = await ext.tavily_search("   ")
    assert out.startswith("ERROR")


@pytest.mark.asyncio
async def test_github_get_file_rejects_bad_repo(fresh_storage) -> None:
    fresh_storage.set_external_tool_key("github_pat", "ghp_test")
    out = await ext.github_get_file("no-slash", "README.md")
    assert out.startswith("ERROR") and "owner/name" in out
