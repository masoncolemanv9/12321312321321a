"""Unit tests for :class:`bot.workers.publisher.PublisherWorker`.

DRY-RUN flow only — no real uploads. We verify file staging, metadata
formats, and terminal-stage behavior.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from bot.jobs import Job
from bot.workers import publisher as publisher_mod
from bot.workers.publisher import (
    PublisherWorker,
    build_instagram_metadata,
    build_tiktok_metadata,
    build_youtube_metadata,
)


def _make_publish_job(payload: dict[str, Any], job_id: int = 31) -> Job:
    return Job(
        id=job_id,
        kind="publish",
        status="running",
        parent_id=None,
        payload=payload,
        result=None,
        error=None,
        retries=0,
        chat_id=None,
        created_at="now",
        updated_at="now",
    )


@pytest.fixture
def fake_clip(tmp_path: Path) -> Path:
    p = tmp_path / "edited.mp4"
    p.write_bytes(b"\x00mp4_content")
    return p


@pytest.fixture(autouse=True)
def releases_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Redirect releases dir to a temp path for each test."""
    base = tmp_path / "releases"
    base.mkdir()
    monkeypatch.setattr(publisher_mod._config, "RELEASES_DIR", base)
    return base


async def test_creates_release_directory_per_job(
    fake_clip: Path, releases_dir: Path
) -> None:
    worker = PublisherWorker(MagicMock())
    result = await worker.process(
        _make_publish_job(
            {
                "clip_path": str(fake_clip),
                "seo_title": "test title",
                "seo_description": "test desc",
                "tags": ["one", "two"],
                "language": "en",
            },
            job_id=42,
        )
    )

    assert Path(result["release_dir"]) == releases_dir / "42"
    assert Path(result["release_dir"]).is_dir()
    assert result["dry_run"] is True
    assert result["post_url"] is None
    assert result["platforms"] == ["youtube", "tiktok", "instagram"]


async def test_links_or_copies_clip_into_release(
    fake_clip: Path, releases_dir: Path
) -> None:
    worker = PublisherWorker(MagicMock())
    result = await worker.process(
        _make_publish_job(
            {
                "clip_path": str(fake_clip),
                "seo_title": "x",
                "tags": [],
            }
        )
    )

    staged = Path(result["clip_path"])
    assert staged.exists()
    assert staged.read_bytes() == fake_clip.read_bytes()
    assert staged.parent == Path(result["release_dir"])
    assert staged.name == "clip.mp4"


async def test_youtube_metadata_format(fake_clip: Path) -> None:
    worker = PublisherWorker(MagicMock())
    result = await worker.process(
        _make_publish_job(
            {
                "clip_path": str(fake_clip),
                "seo_title": "wow opener",
                "seo_description": "wait til the end #shorts",
                "tags": ["plot twist", "viral"],
                "language": "en",
            }
        )
    )

    yt_path = Path(result["metadata_paths"]["youtube"])
    assert yt_path.exists()
    yt = json.loads(yt_path.read_text())
    assert yt["snippet"]["title"] == "wow opener"
    assert yt["snippet"]["description"] == "wait til the end #shorts"
    assert yt["snippet"]["tags"] == ["plot twist", "viral"]
    assert yt["snippet"]["defaultLanguage"] == "en"
    assert yt["status"]["privacyStatus"] == "private"
    assert yt["status"]["selfDeclaredMadeForKids"] is False


async def test_tiktok_metadata_format(fake_clip: Path) -> None:
    worker = PublisherWorker(MagicMock())
    result = await worker.process(
        _make_publish_job(
            {
                "clip_path": str(fake_clip),
                "seo_title": "wow opener",
                "seo_description": "wait til the end",
                "tags": ["plot twist!", "viral"],
                "language": "en",
            }
        )
    )

    tiktok_path = Path(result["metadata_paths"]["tiktok"])
    tiktok = json.loads(tiktok_path.read_text())
    assert tiktok["caption"].startswith("wow opener")
    # Punctuation stripped from hashtags.
    assert "plottwist" in tiktok["hashtags"]
    assert tiktok["language"] == "en"
    assert tiktok["private"] is True


async def test_instagram_metadata_format(fake_clip: Path) -> None:
    worker = PublisherWorker(MagicMock())
    result = await worker.process(
        _make_publish_job(
            {
                "clip_path": str(fake_clip),
                "seo_title": "title",
                "seo_description": "description body",
                "tags": ["one", "two"],
                "language": "ru",
            }
        )
    )

    ig_path = Path(result["metadata_paths"]["instagram"])
    ig = json.loads(ig_path.read_text())
    assert "title" in ig["caption"]
    assert "description body" in ig["caption"]
    assert "#one" in ig["caption"]
    assert "#two" in ig["caption"]
    assert ig["language"] == "ru"
    assert ig["share_to_feed"] is True


async def test_terminal_stage_no_enqueue_next(fake_clip: Path) -> None:
    """Worker has next_kind = None and enqueue_next short-circuits."""
    queue = MagicMock()
    queue.enqueue = MagicMock()  # would track calls if any happened
    worker = PublisherWorker(queue)
    job = _make_publish_job({"clip_path": str(fake_clip), "seo_title": "x"})
    result = await worker.process(job)

    await worker.enqueue_next(job, result)
    queue.enqueue.assert_not_called()
    assert PublisherWorker.next_kind is None


async def test_handles_missing_clip() -> None:
    worker = PublisherWorker(MagicMock())
    with pytest.raises(FileNotFoundError):
        await worker.process(
            _make_publish_job(
                {"clip_path": "/no/such/file.mp4", "seo_title": "x"}
            )
        )


async def test_writes_upload_instructions(fake_clip: Path) -> None:
    worker = PublisherWorker(MagicMock())
    result = await worker.process(
        _make_publish_job(
            {
                "clip_path": str(fake_clip),
                "seo_title": "title",
                "tags": [],
                "hook": "the hook",
                "clip_index": 2,
            }
        )
    )

    instr_path = Path(result["instructions_path"])
    assert instr_path.exists()
    text = instr_path.read_text()
    assert "YouTube Shorts" in text
    assert "TikTok" in text
    assert "Instagram" in text
    assert "the hook" in text


# ---- pure-function tests -------------------------------------------------


def test_build_youtube_metadata_falls_back_for_empty_title() -> None:
    yt = build_youtube_metadata(title="", description="x", tags=[], language="en")
    assert yt["snippet"]["title"] == "Untitled clip"


def test_build_tiktok_caption_caps_at_2200_chars() -> None:
    long = "a" * 5000
    tiktok = build_tiktok_metadata(
        title="x", description=long, tags=["t"], language="en"
    )
    assert len(tiktok["caption"]) == 2200


def test_build_instagram_includes_hashtag_block_once() -> None:
    ig = build_instagram_metadata(
        title="x", description="y", tags=["one", "two"], language="en"
    )
    # Caption should include #one and #two exactly once each.
    assert ig["caption"].count("#one") == 1
    assert ig["caption"].count("#two") == 1
