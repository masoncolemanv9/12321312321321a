"""Unit tests for :class:`bot.workers.editor.EditorWorker`.

ffmpeg invocations are mocked via ``subprocess.run``. No real encoding
runs in CI.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from bot.jobs import Job
from bot.workers import editor as editor_mod
from bot.workers.editor import (
    EditorWorker,
    build_ffmpeg_command,
    compute_crop_filter,
)


def _make_edit_job(payload: dict[str, Any], job_id: int = 11) -> Job:
    return Job(
        id=job_id,
        kind="edit",
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
def fake_source(tmp_path: Path) -> Path:
    src = tmp_path / "source.mp4"
    src.write_bytes(b"\x00fake")
    return src


@pytest.fixture(autouse=True)
def stub_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Capture ``subprocess.run`` calls instead of actually running ffmpeg."""
    state: dict[str, Any] = {"calls": []}

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        state["calls"].append(cmd)
        # Touch the output file so callers can verify it exists.
        if cmd and cmd[-1].endswith(".mp4"):
            Path(cmd[-1]).write_bytes(b"\x00mp4")
        return subprocess.CompletedProcess(cmd, returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    # Default: no overlay.
    monkeypatch.setattr(editor_mod._config, "OVERLAY_LOGO_PATH", "")
    return state


async def test_runs_ffmpeg_with_correct_filter_chain(
    fake_source: Path, stub_ffmpeg: dict[str, Any]
) -> None:
    worker = EditorWorker(MagicMock())
    result = await worker.process(
        _make_edit_job(
            {
                "source_path": str(fake_source),
                "start_s": 30.0,
                "end_s": 60.5,
                "clip_index": 2,
                "hook": "wow",
            }
        )
    )

    assert result["width"] == 1080
    assert result["height"] == 1920
    assert result["clip_index"] == 2
    assert result["hook"] == "wow"
    assert abs(result["duration_s"] - 30.5) < 1e-6
    assert result["overlay_used"] is False

    assert len(stub_ffmpeg["calls"]) == 1
    cmd = stub_ffmpeg["calls"][0]
    assert cmd[0] == "ffmpeg"
    assert "-ss" in cmd and "30.000" in cmd
    assert "-to" in cmd and "60.500" in cmd
    cmd_str = " ".join(cmd)
    assert "crop=ih*1080/1920:ih" in cmd_str
    assert "scale=1080:1920" in cmd_str
    assert "libx264" in cmd_str
    assert "yuv420p" in cmd_str


async def test_writes_output_under_source_clips_dir(
    fake_source: Path, stub_ffmpeg: dict[str, Any]
) -> None:
    worker = EditorWorker(MagicMock())
    result = await worker.process(
        _make_edit_job(
            {
                "source_path": str(fake_source),
                "start_s": 0.0,
                "end_s": 15.0,
                "clip_index": 0,
            },
            job_id=99,
        )
    )

    clip_path = Path(result["clip_path"])
    assert clip_path.parent == fake_source.parent / "clips"
    assert clip_path.name == "99_0.mp4"
    assert clip_path.exists()


async def test_overlay_added_when_logo_exists(
    fake_source: Path,
    stub_ffmpeg: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logo = tmp_path / "logo.png"
    logo.write_bytes(b"\x89PNG\r\n\x1a\n")  # PNG header is enough for existence check
    monkeypatch.setattr(editor_mod._config, "OVERLAY_LOGO_PATH", str(logo))

    worker = EditorWorker(MagicMock())
    result = await worker.process(
        _make_edit_job(
            {
                "source_path": str(fake_source),
                "start_s": 5.0,
                "end_s": 25.0,
                "clip_index": 0,
            }
        )
    )

    assert result["overlay_used"] is True
    cmd = stub_ffmpeg["calls"][0]
    cmd_str = " ".join(cmd)
    # Logo passed as a second -i input.
    assert cmd.count("-i") == 2
    assert str(logo) in cmd
    assert "overlay=" in cmd_str
    assert "filter_complex" in cmd_str


async def test_overlay_skipped_when_logo_missing(
    fake_source: Path,
    stub_ffmpeg: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(editor_mod._config, "OVERLAY_LOGO_PATH", "/no/such/logo.png")

    worker = EditorWorker(MagicMock())
    result = await worker.process(
        _make_edit_job(
            {
                "source_path": str(fake_source),
                "start_s": 0.0,
                "end_s": 10.0,
                "clip_index": 0,
            }
        )
    )

    assert result["overlay_used"] is False
    cmd = stub_ffmpeg["calls"][0]
    assert cmd.count("-i") == 1
    assert "overlay=" not in " ".join(cmd)


async def test_ffmpeg_failure_surfaces_friendly_error(
    fake_source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        raise subprocess.CalledProcessError(
            returncode=1, cmd=cmd, output=b"", stderr=b"Invalid data found"
        )

    monkeypatch.setattr(subprocess, "run", boom)
    monkeypatch.setattr(editor_mod._config, "OVERLAY_LOGO_PATH", "")

    worker = EditorWorker(MagicMock())
    with pytest.raises(RuntimeError, match="ffmpeg failed: Invalid data found"):
        await worker.process(
            _make_edit_job(
                {
                    "source_path": str(fake_source),
                    "start_s": 0.0,
                    "end_s": 10.0,
                    "clip_index": 0,
                }
            )
        )


async def test_invalid_window_raises(fake_source: Path) -> None:
    worker = EditorWorker(MagicMock())
    with pytest.raises(ValueError, match="non-positive window"):
        await worker.process(
            _make_edit_job(
                {
                    "source_path": str(fake_source),
                    "start_s": 30.0,
                    "end_s": 30.0,
                    "clip_index": 0,
                }
            )
        )


async def test_missing_source_raises() -> None:
    worker = EditorWorker(MagicMock())
    with pytest.raises(FileNotFoundError):
        await worker.process(
            _make_edit_job(
                {
                    "source_path": "/no/such/file.mp4",
                    "start_s": 0.0,
                    "end_s": 5.0,
                    "clip_index": 0,
                }
            )
        )


# ---- pure-function tests -------------------------------------------------


def test_compute_crop_filter_for_1080x1920() -> None:
    f = compute_crop_filter(width=1080, height=1920)
    assert f == "crop=ih*1080/1920:ih"


def test_build_ffmpeg_command_no_overlay(
    fake_source: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(editor_mod._config, "OVERLAY_LOGO_PATH", "")
    out = tmp_path / "out.mp4"
    cmd, used = build_ffmpeg_command(
        source_path=fake_source, clip_path=out, start_s=1.0, end_s=2.0
    )
    assert used is False
    assert "-vf" in cmd
    assert "filter_complex" not in cmd
