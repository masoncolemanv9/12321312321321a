"""Unit tests for :class:`bot.workers.seo.SeoWorker`.

pytrends and OpenRouter HTTP calls are mocked. No network in CI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from bot.jobs import Job
from bot.workers import seo as seo_mod
from bot.workers.seo import (
    SeoWorker,
    _to_hashtag,
    fetch_trends,
)


def _make_seo_job(payload: dict[str, Any], job_id: int = 21) -> Job:
    return Job(
        id=job_id,
        kind="seo",
        status="running",
        parent_id=None,
        payload=payload,
        result=None,
        error=None,
        retries=0,
        chat_id=42,
        created_at="now",
        updated_at="now",
    )


@pytest.fixture
def transcript_file(tmp_path: Path) -> Path:
    path = tmp_path / "transcript.json"
    path.write_text(
        json.dumps(
            {
                "language": "ru",
                "segments": [
                    {"start": 0, "end": 3, "text": "Привет, это тест."},
                    {"start": 3, "end": 6, "text": "Здесь много слов."},
                ],
            },
            ensure_ascii=False,
        )
    )
    return path


@pytest.fixture
def stub_trends(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace ``fetch_trends`` with a controllable stub."""
    state: dict[str, Any] = {"calls": [], "value": ["AI", "GPT", "music"]}

    def fake_fetch(*, language: str) -> list[str]:
        state["calls"].append(language)
        return list(state["value"])

    monkeypatch.setattr(seo_mod, "fetch_trends", fake_fetch)
    return state


async def test_uses_template_without_api_key(
    transcript_file: Path,
    stub_trends: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(seo_mod._config, "OPENROUTER_API_KEY", "")

    worker = SeoWorker(MagicMock())
    result = await worker.process(
        _make_seo_job(
            {
                "clip_path": "/tmp/clip.mp4",
                "hook": "wow opener",
                "transcript_path": str(transcript_file),
            }
        )
    )

    assert result["seo_method"] == "template"
    assert result["seo_title"]
    assert "wow opener" in result["seo_title"]
    assert "#" in result["seo_description"]
    assert result["language"] == "ru"
    assert "AI" in result["trends_used"]
    assert len(result["tags"]) <= 15
    # Tags don't include the leading '#'.
    for tag in result["tags"]:
        assert not tag.startswith("#")


async def test_uses_llm_when_key_present(
    transcript_file: Path,
    stub_trends: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(seo_mod._config, "OPENROUTER_API_KEY", "sk-test")

    captured: dict[str, Any] = {}

    class _FakeResp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "title": "Insane plot twist you missed",
                                    "description": "Wait til the end #shorts #wow",
                                    "tags": ["plot twist", "shorts", "viral"],
                                }
                            )
                        }
                    }
                ]
            }

    def fake_post(url: str, **kwargs: Any) -> _FakeResp:
        captured["url"] = url
        captured["kwargs"] = kwargs
        return _FakeResp()

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)

    worker = SeoWorker(MagicMock())
    result = await worker.process(
        _make_seo_job(
            {
                "clip_path": "/tmp/clip.mp4",
                "hook": "kicker hook",
                "transcript_path": str(transcript_file),
            }
        )
    )

    assert result["seo_method"] == "llm"
    assert result["seo_title"] == "Insane plot twist you missed"
    assert "plot twist" in result["tags"]
    assert captured["url"].endswith("/chat/completions")
    auth = captured["kwargs"]["headers"]["Authorization"]
    assert auth == "Bearer sk-test"


async def test_invalid_llm_response_falls_back(
    transcript_file: Path,
    stub_trends: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Garbage from the LLM → template fallback, no crash."""
    monkeypatch.setattr(seo_mod._config, "OPENROUTER_API_KEY", "sk-test")

    class _BadResp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"choices": [{"message": {"content": "not json at all <html>"}}]}

    import httpx

    monkeypatch.setattr(httpx, "post", lambda url, **kwargs: _BadResp())

    worker = SeoWorker(MagicMock())
    result = await worker.process(
        _make_seo_job(
            {
                "clip_path": "/tmp/clip.mp4",
                "hook": "fallback hook",
                "transcript_path": str(transcript_file),
            }
        )
    )

    assert result["seo_method"] == "template"
    assert result["seo_title"]


async def test_language_propagated(
    transcript_file: Path, stub_trends: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explicit ``language`` in payload wins over transcript-detected one."""
    monkeypatch.setattr(seo_mod._config, "OPENROUTER_API_KEY", "")

    worker = SeoWorker(MagicMock())
    result = await worker.process(
        _make_seo_job(
            {
                "clip_path": "/tmp/clip.mp4",
                "hook": "x",
                "language": "EN",  # uppercase: should normalize
                "transcript_path": str(transcript_file),
            }
        )
    )
    assert result["language"] == "en"
    assert stub_trends["calls"] == ["en"]


async def test_works_without_transcript(
    stub_trends: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(seo_mod._config, "OPENROUTER_API_KEY", "")

    worker = SeoWorker(MagicMock())
    result = await worker.process(
        _make_seo_job(
            {
                "clip_path": "/tmp/clip.mp4",
                "hook": "no transcript here",
            }
        )
    )

    assert result["language"] == "en"
    assert "no transcript here" in result["seo_title"]


# ---- pure-function tests -------------------------------------------------


def test_fetch_trends_handles_pytrends_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Any error from pytrends should yield an empty list, not raise."""

    class _BoomTrend:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def trending_searches(self, pn: str) -> Any:
            raise RuntimeError("rate limited 429")

    import pytrends.request

    monkeypatch.setattr(pytrends.request, "TrendReq", _BoomTrend)

    assert fetch_trends(language="en") == []


def test_fetch_trends_returns_top_strings(monkeypatch: pytest.MonkeyPatch) -> None:
    import pandas as pd

    class _OkTrend:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def trending_searches(self, pn: str) -> pd.DataFrame:
            return pd.DataFrame({0: ["alpha", "beta", "gamma"]})

    import pytrends.request

    monkeypatch.setattr(pytrends.request, "TrendReq", _OkTrend)

    out = fetch_trends(language="en")
    assert out == ["alpha", "beta", "gamma"]


def test_to_hashtag_strips_punctuation() -> None:
    assert _to_hashtag("plot twist!") == "plottwist"
    assert _to_hashtag("#already") == "already"
    assert _to_hashtag("emoji 🎉 strip") == "emojistrip"


def test_passes_payload_through_to_publisher(
    transcript_file: Path,
    stub_trends: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``enqueue_next`` produces a publish job with seo metadata stamped on."""
    monkeypatch.setattr(seo_mod._config, "OPENROUTER_API_KEY", "")

    enqueued: list[tuple[str, dict[str, Any], dict[str, Any]]] = []

    class _FakeQueue:
        async def enqueue(
            self,
            kind: str,
            payload: dict[str, Any],
            *,
            chat_id: int | None = None,
            parent_id: int | None = None,
        ) -> int:
            enqueued.append((kind, payload, {"chat_id": chat_id, "parent_id": parent_id}))
            return len(enqueued)

    worker = SeoWorker(_FakeQueue())  # type: ignore[arg-type]
    job = _make_seo_job(
        {
            "clip_path": "/tmp/clip.mp4",
            "hook": "the hook",
            "transcript_path": str(transcript_file),
        }
    )

    import asyncio

    result = asyncio.run(worker.process(job))
    asyncio.run(worker.enqueue_next(job, result))

    assert len(enqueued) == 1
    kind, payload, meta = enqueued[0]
    assert kind == "publish"
    assert payload["clip_path"] == "/tmp/clip.mp4"
    assert "seo_title" in payload
    assert "tags" in payload
    assert meta["parent_id"] == job.id
    assert meta["chat_id"] == job.chat_id
