"""Unit tests for :class:`bot.workers.downloader.DownloaderWorker`.

The tests fully mock the ``yt_dlp.YoutubeDL`` class — no network access,
no actual ffmpeg merge — so they're safe to run on any machine and in CI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from bot.jobs import Job
from bot.workers import downloader as downloader_mod
from bot.workers.downloader import DownloaderWorker


def _make_job(payload: dict[str, Any], job_id: int = 1) -> Job:
    return Job(
        id=job_id,
        kind="download",
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


class _FakeYDL:
    """Stand-in for ``yt_dlp.YoutubeDL`` configurable per-test."""

    instances: list[_FakeYDL] = []

    def __init__(
        self,
        *,
        preflight_info: dict[str, Any] | None = None,
        download_info: dict[str, Any] | None = None,
        creates_file: Path | None = None,
        download_raises: Exception | None = None,
        preflight_raises: Exception | None = None,
    ) -> None:
        self.preflight_info = preflight_info or {}
        self.download_info = download_info or self.preflight_info
        self.creates_file = creates_file
        self.download_raises = download_raises
        self.preflight_raises = preflight_raises
        self.calls: list[tuple[str, bool]] = []

    def __call__(self, opts: dict[str, Any] | None = None) -> _FakeYDL:
        # YoutubeDL(opts) — the worker constructs us; return self so the
        # ``with YoutubeDL(opts) as ydl`` block works.
        self.last_opts = opts or {}
        _FakeYDL.instances.append(self)
        return self

    def __enter__(self) -> _FakeYDL:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def extract_info(self, url: str, download: bool = True) -> dict[str, Any]:
        self.calls.append((url, download))
        if not download:
            if self.preflight_raises:
                raise self.preflight_raises
            return self.preflight_info
        if self.download_raises:
            raise self.download_raises
        if self.creates_file is not None:
            self.creates_file.parent.mkdir(parents=True, exist_ok=True)
            self.creates_file.write_bytes(b"\x00fake-mp4-bytes")
        info = dict(self.download_info)
        if self.creates_file is not None and "_filename" not in info:
            info["_filename"] = str(self.creates_file)
        return info

    def sanitize_info(self, info: dict[str, Any]) -> dict[str, Any]:
        return info


@pytest.fixture(autouse=True)
def isolate_downloads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Each test gets its own DOWNLOADS_DIR under ``tmp_path``."""
    dl = tmp_path / "downloads"
    dl.mkdir()
    monkeypatch.setattr(downloader_mod, "DOWNLOADS_DIR", dl)
    _FakeYDL.instances.clear()
    return dl


async def test_resolves_metadata_only_first(
    isolate_downloads: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pre-flight (download=False) must run before any actual download."""
    target = isolate_downloads / "1" / "source.mp4"
    fake = _FakeYDL(
        preflight_info={
            "title": "T",
            "duration": 120.0,
            "filesize_approx": 10 * 1024 * 1024,  # 10 MB
            "uploader": "U",
            "webpage_url": "https://example.com/v",
            "width": 1920,
            "height": 1080,
            "fps": 30,
            "language": "en",
        },
        creates_file=target,
    )
    monkeypatch.setattr(downloader_mod, "YoutubeDL", fake)

    worker = DownloaderWorker(MagicMock())
    result = await worker.process(_make_job({"url": "https://example.com/v"}))

    # Pre-flight (download=False) must come strictly before the real download.
    assert fake.calls == [
        ("https://example.com/v", False),
        ("https://example.com/v", True),
    ]
    assert result["source_path"] == str(target)
    assert result["title"] == "T"
    assert result["duration_s"] == 120.0
    assert result["height"] == 1080


async def test_rejects_oversized_source(
    isolate_downloads: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeYDL(
        preflight_info={"filesize_approx": 10 * 1024 * 1024 * 1024},  # 10 GB
    )
    monkeypatch.setattr(downloader_mod, "YoutubeDL", fake)

    worker = DownloaderWorker(MagicMock())
    job = _make_job({"url": "https://example.com/big", "max_filesize_mb": 5000})

    with pytest.raises(RuntimeError, match="too big"):
        await worker.process(job)
    # Only the pre-flight call ran; we bailed before download.
    assert fake.calls == [("https://example.com/big", False)]


async def test_writes_metadata_json(
    isolate_downloads: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = isolate_downloads / "7" / "source.mp4"
    fake = _FakeYDL(
        preflight_info={"title": "Hello", "filesize_approx": 1024},
        download_info={"title": "Hello", "uploader": "U", "duration": 30.0},
        creates_file=target,
    )
    monkeypatch.setattr(downloader_mod, "YoutubeDL", fake)

    worker = DownloaderWorker(MagicMock())
    result = await worker.process(_make_job({"url": "u"}, job_id=7))

    metadata_path = Path(result["metadata_path"])
    assert metadata_path.exists()
    payload = json.loads(metadata_path.read_text())
    assert payload["title"] == "Hello"


async def test_progress_hook_writes_log(
    isolate_downloads: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = isolate_downloads / "11" / "source.mp4"

    class _ProgressYDL(_FakeYDL):
        def extract_info(self, url: str, download: bool = True) -> dict[str, Any]:
            if download:
                # Simulate yt-dlp firing the hook a few times.
                hook = self.last_opts["progress_hooks"][0]
                hook({"status": "downloading", "_percent_str": "10%",
                      "_speed_str": "1MiB/s", "_eta_str": "30s"})
                hook({"status": "finished", "filename": str(target)})
            return super().extract_info(url, download)

    fake = _ProgressYDL(
        preflight_info={"filesize_approx": 1024, "title": "X"},
        download_info={"title": "X", "duration": 5.0},
        creates_file=target,
    )
    monkeypatch.setattr(downloader_mod, "YoutubeDL", fake)

    worker = DownloaderWorker(MagicMock())
    await worker.process(_make_job({"url": "u"}, job_id=11))

    log_path = isolate_downloads / "11" / "download.log"
    text = log_path.read_text()
    assert "[downloading]" in text
    assert "10%" in text
    assert "[finished]" in text


async def test_pre_flight_failure_surfaces_friendly_error(
    isolate_downloads: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from yt_dlp.utils import DownloadError as RealDownloadError

    fake = _FakeYDL(preflight_raises=RealDownloadError("403 Forbidden"))
    monkeypatch.setattr(downloader_mod, "YoutubeDL", fake)

    worker = DownloaderWorker(MagicMock())
    with pytest.raises(RuntimeError, match="pre-flight failed"):
        await worker.process(_make_job({"url": "u"}))
