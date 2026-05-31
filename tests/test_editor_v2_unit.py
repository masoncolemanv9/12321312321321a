"""Unit tests for :class:`bot.workers.editor_v2.EditorV2Worker`.

``ffprobe`` / ``ffmpeg`` / ``cv2`` are all mocked. The integration test
in ``test_editor_v2_integration.py`` exercises a real ffmpeg run.

Coverage matrix:

* payload validation — minimal v1 payload and full v2 payload accepted;
* missing source path / non-positive window → ``EditorV2PayloadError``;
* probe failure → abort (raise);
* face sampling missing → warn + fallback recorded in manifest;
* simplified retry triggered on encode failure, succeeds on retry;
* simplified retry exhausted → abort;
* manifest fields per §9.12 present;
* result payload preserves v1 keys (§9.3 pass-through);
* worker factory: EDITOR_VERSION dispatch.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from bot.jobs import Job
from bot.uniqueization import FaceSampling, ProbeResult
from bot.uniqueization.runner import FfmpegError, FfmpegResult
from bot.workers import EditorV2Worker, EditorWorker, build_editor_worker
from bot.workers import editor_v2 as ev2_mod
from bot.workers.editor_v2 import EditorV2PayloadError, _validate_payload

# ---- fixtures ------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_editor_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for k in list(os.environ):
        if k.startswith("EDITOR_") or k == "OVERLAY_LOGO_PATH":
            monkeypatch.delenv(k, raising=False)
    yield


@pytest.fixture
def fake_source(tmp_path: Path) -> Path:
    src = tmp_path / "source.mp4"
    src.write_bytes(b"\x00fake")
    return src


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


def _stub_probe(monkeypatch: pytest.MonkeyPatch, source: Path) -> None:
    async def fake_probe(p: Path | str) -> ProbeResult:
        return ProbeResult(
            path=Path(p),
            duration_s=120.0,
            width=1920,
            height=1080,
            fps=30.0,
            has_audio=True,
        )

    monkeypatch.setattr(ev2_mod, "probe_source", fake_probe)


def _stub_face_center(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_sampler(*args: Any, **kwargs: Any) -> FaceSampling:  # noqa: ARG001
        return FaceSampling(
            samples=(),
            median_center_pct=(0.5, 0.5),
            fallback=False,
            reason="",
        )

    monkeypatch.setattr(ev2_mod, "sample_faces", fake_sampler)


def _stub_phash_thumbnail(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ev2_mod,
        "extract_phash_samples",
        lambda *a, **kw: pytest.skip("should not be reached when phash disabled"),
    )
    monkeypatch.setattr(
        ev2_mod,
        "extract_thumbnail",
        lambda *a, **kw: pytest.skip("should not be reached when thumb disabled"),
    )


def _stub_encode_success(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {"calls": []}

    def fake_runner(
        argv_prefix: list[str],
        *,
        output_path: Path,
        timeout_s: float,
        debug_dir: Path,
        keep_artifacts: bool = False,
    ) -> FfmpegResult:
        state["calls"].append(list(argv_prefix))
        state["timeout_s"] = timeout_s
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"\x00mp4")
        return FfmpegResult(
            argv=list(argv_prefix),
            elapsed_ms=42,
            stderr="",
            output_path=output_path,
        )

    monkeypatch.setattr(ev2_mod, "run_ffmpeg_atomic", fake_runner)
    return state


# ---- payload validation --------------------------------------------


def test_validate_payload_accepts_minimal_v1_shape(fake_source: Path) -> None:
    out = _validate_payload(
        {
            "source_path": str(fake_source),
            "start_s": 1.0,
            "end_s": 5.5,
            "clip_index": 2,
        }
    )
    src, start_s, end_s, idx, hook = out
    assert src == fake_source
    assert start_s == 1.0
    assert end_s == 5.5
    assert idx == 2
    assert hook == ""


def test_validate_payload_accepts_full_v2_shape(fake_source: Path) -> None:
    out = _validate_payload(
        {
            "source_path": str(fake_source),
            "duration_s": 5400.0,
            "language": "en",
            "transcript_path": "/tmp/transcript.json",
            "clip_index": 3,
            "start_s": 12.5,
            "end_s": 42.5,
            "hook": "wow opener",
            "score": 0.81,
        }
    )
    src, start_s, end_s, idx, hook = out
    assert src == fake_source
    assert (start_s, end_s, idx, hook) == (12.5, 42.5, 3, "wow opener")


def test_validate_payload_missing_source() -> None:
    with pytest.raises(EditorV2PayloadError, match="source_path"):
        _validate_payload({"start_s": 0.0, "end_s": 1.0})


def test_validate_payload_non_positive_window(fake_source: Path) -> None:
    with pytest.raises(EditorV2PayloadError, match="non-positive"):
        _validate_payload(
            {"source_path": str(fake_source), "start_s": 5.0, "end_s": 5.0}
        )


# ---- happy path -----------------------------------------------------


async def test_process_writes_manifest_and_clip(
    fake_source: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_probe(monkeypatch, fake_source)
    _stub_face_center(monkeypatch)
    state = _stub_encode_success(monkeypatch)
    # Make phash / thumbnail no-ops that still record stages.
    monkeypatch.setattr(
        ev2_mod,
        "extract_phash_samples",
        lambda *a, **kw: [],  # type: ignore[misc]
    )
    monkeypatch.setattr(
        ev2_mod,
        "extract_thumbnail",
        lambda src, out, **kw: out.write_bytes(b"\xff\xd8\xff") or out,  # type: ignore[misc,return-value]
    )

    worker = EditorV2Worker(MagicMock())
    result = await worker.process(
        _make_edit_job(
            {
                "source_path": str(fake_source),
                "start_s": 30.0,
                "end_s": 60.0,
                "clip_index": 0,
                "hook": "wow",
                "language": "en",
                "transcript_path": "/tmp/transcript.json",
                "score": 0.81,
            }
        )
    )
    # Result shape: §9.3 keys present.
    assert result["clip_path"].endswith("/clips/11_0.mp4")
    assert result["width"] == 1080
    assert result["height"] == 1920
    assert result["duration_s"] == 30.0
    assert result["editor_version"] == "v1"  # default; no env override here
    # Pass-through (§9.3).
    assert result["language"] == "en"
    assert result["transcript_path"] == "/tmp/transcript.json"
    assert result["score"] == 0.81
    # Manifest written.
    manifest_path = Path(result["uniqueization_manifest_path"])
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["job"]["id"] == 11
    assert manifest["source"]["duration_s"] == 120.0
    assert manifest["output"]["clip_path"] == result["clip_path"]
    stage_names = {s["name"] for s in manifest["stages"]}
    assert "ffprobe" in stage_names
    assert "final_encode" in stage_names
    # Encoder invoked exactly once.
    assert len(state["calls"]) == 1
    # Argv has -filter_complex and the runner adds the temp output path.
    assert "-filter_complex" in state["calls"][0]


# ---- failure modes --------------------------------------------------


async def test_probe_failure_aborts(
    fake_source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from bot.uniqueization.probe import ProbeError

    async def fake_probe(_p: Any) -> ProbeResult:
        raise ProbeError("ffprobe unavailable")

    monkeypatch.setattr(ev2_mod, "probe_source", fake_probe)

    worker = EditorV2Worker(MagicMock())
    with pytest.raises(EditorV2PayloadError, match="ffprobe failed"):
        await worker.process(
            _make_edit_job(
                {
                    "source_path": str(fake_source),
                    "start_s": 0.0,
                    "end_s": 5.0,
                    "clip_index": 0,
                }
            )
        )


async def test_face_sampling_failure_warn_and_continue(
    fake_source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_probe(monkeypatch, fake_source)

    def angry_sampler(*a: Any, **kw: Any) -> FaceSampling:  # noqa: ARG001
        raise RuntimeError("cv2 model corrupted")

    monkeypatch.setattr(ev2_mod, "sample_faces", angry_sampler)
    _stub_encode_success(monkeypatch)
    monkeypatch.setattr(
        ev2_mod, "extract_phash_samples", lambda *a, **kw: []
    )
    monkeypatch.setattr(
        ev2_mod,
        "extract_thumbnail",
        lambda src, out, **kw: out.write_bytes(b"\xff") or out,  # type: ignore[misc,return-value]
    )

    worker = EditorV2Worker(MagicMock())
    result = await worker.process(
        _make_edit_job(
            {
                "source_path": str(fake_source),
                "start_s": 0.0,
                "end_s": 5.0,
                "clip_index": 0,
            }
        )
    )
    manifest = json.loads(
        Path(result["uniqueization_manifest_path"]).read_text(encoding="utf-8")
    )
    warnings = manifest["warnings"]
    assert any("face_fallback" in w for w in warnings)
    # Stage row records the failure.
    sample_rows = [s for s in manifest["stages"] if s["name"] == "zoom-reframe-agent.sample"]
    assert sample_rows
    assert sample_rows[0]["status"] == "failed"


async def test_encode_failure_triggers_simplified_retry(
    fake_source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_probe(monkeypatch, fake_source)
    _stub_face_center(monkeypatch)

    state: dict[str, Any] = {"calls": []}

    def flaky_runner(
        argv_prefix: list[str], **kwargs: Any
    ) -> FfmpegResult:
        state["calls"].append(list(argv_prefix))
        output_path: Path = kwargs["output_path"]
        if len(state["calls"]) == 1:
            raise FfmpegError(
                "ffmpeg exited 1", argv=argv_prefix, stderr="boom"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"\x00mp4")
        return FfmpegResult(
            argv=list(argv_prefix), elapsed_ms=99, stderr="", output_path=output_path
        )

    monkeypatch.setattr(ev2_mod, "run_ffmpeg_atomic", flaky_runner)
    monkeypatch.setattr(ev2_mod, "extract_phash_samples", lambda *a, **kw: [])
    monkeypatch.setattr(
        ev2_mod,
        "extract_thumbnail",
        lambda src, out, **kw: out.write_bytes(b"\xff") or out,  # type: ignore[misc,return-value]
    )

    worker = EditorV2Worker(MagicMock())
    result = await worker.process(
        _make_edit_job(
            {
                "source_path": str(fake_source),
                "start_s": 0.0,
                "end_s": 5.0,
                "clip_index": 0,
            }
        )
    )
    assert len(state["calls"]) == 2  # primary + simplified retry.
    manifest = json.loads(
        Path(result["uniqueization_manifest_path"]).read_text(encoding="utf-8")
    )
    stage_names = [s["name"] for s in manifest["stages"]]
    assert "final_encode" in stage_names
    assert "final_encode_simplified" in stage_names
    assert any("final_encode_failed_retry_simplified" in w for w in manifest["warnings"])


async def test_encode_failure_after_retry_aborts(
    fake_source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_probe(monkeypatch, fake_source)
    _stub_face_center(monkeypatch)

    def always_fails(
        argv_prefix: list[str], **kwargs: Any
    ) -> FfmpegResult:
        raise FfmpegError("ffmpeg exited 1", argv=argv_prefix, stderr="boom")

    monkeypatch.setattr(ev2_mod, "run_ffmpeg_atomic", always_fails)
    monkeypatch.setattr(ev2_mod, "extract_phash_samples", lambda *a, **kw: [])
    monkeypatch.setattr(
        ev2_mod,
        "extract_thumbnail",
        lambda *a, **kw: pytest.skip("never reached"),
    )

    worker = EditorV2Worker(MagicMock())
    with pytest.raises(FfmpegError):
        await worker.process(
            _make_edit_job(
                {
                    "source_path": str(fake_source),
                    "start_s": 0.0,
                    "end_s": 5.0,
                    "clip_index": 0,
                }
            )
        )


async def test_thumbnail_failure_warn_only(
    fake_source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_probe(monkeypatch, fake_source)
    _stub_face_center(monkeypatch)
    _stub_encode_success(monkeypatch)
    monkeypatch.setattr(ev2_mod, "extract_phash_samples", lambda *a, **kw: [])

    def bad_thumb(*a: Any, **kw: Any) -> Path:  # noqa: ARG001
        raise RuntimeError("ffmpeg thumbnail failed")

    monkeypatch.setattr(ev2_mod, "extract_thumbnail", bad_thumb)

    worker = EditorV2Worker(MagicMock())
    result = await worker.process(
        _make_edit_job(
            {
                "source_path": str(fake_source),
                "start_s": 0.0,
                "end_s": 5.0,
                "clip_index": 0,
            }
        )
    )
    assert result["thumbnail_path"] == ""
    manifest = json.loads(
        Path(result["uniqueization_manifest_path"]).read_text(encoding="utf-8")
    )
    assert any("thumbnail_skipped" in w for w in manifest["warnings"])


# ---- factory --------------------------------------------------------


def test_factory_default_returns_v1(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EDITOR_VERSION", raising=False)
    from bot import config as bot_config
    monkeypatch.setattr(bot_config, "EDITOR_VERSION", "v1")
    w = build_editor_worker(MagicMock())
    assert isinstance(w, EditorWorker)


def test_factory_v2_returns_v2(monkeypatch: pytest.MonkeyPatch) -> None:
    from bot import config as bot_config
    monkeypatch.setattr(bot_config, "EDITOR_VERSION", "v2")
    w = build_editor_worker(MagicMock())
    assert isinstance(w, EditorV2Worker)


def test_factory_v21_returns_v2(monkeypatch: pytest.MonkeyPatch) -> None:
    from bot import config as bot_config
    monkeypatch.setattr(bot_config, "EDITOR_VERSION", "v2.1")
    w = build_editor_worker(MagicMock())
    assert isinstance(w, EditorV2Worker)


def test_factory_v6_returns_v2(monkeypatch: pytest.MonkeyPatch) -> None:
    from bot import config as bot_config
    monkeypatch.setattr(bot_config, "EDITOR_VERSION", "v6")
    w = build_editor_worker(MagicMock())
    assert isinstance(w, EditorV2Worker)


def test_factory_unknown_falls_back_to_v1(monkeypatch: pytest.MonkeyPatch) -> None:
    from bot import config as bot_config
    monkeypatch.setattr(bot_config, "EDITOR_VERSION", "v99")
    w = build_editor_worker(MagicMock())
    assert isinstance(w, EditorWorker)
