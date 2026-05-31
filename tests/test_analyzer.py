"""Unit tests for :class:`bot.workers.analyzer.AnalyzerWorker`.

All heavy dependencies (``ffmpeg``, ``faster-whisper``, ``pyscenedetect``,
``httpx`` to OpenRouter) are mocked. Tests run offline in CI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from bot.jobs import Job
from bot.workers import analyzer as analyzer_mod
from bot.workers.analyzer import (
    AnalyzerWorker,
    _Clip,
    _Scene,
    _Segment,
    build_candidate_windows,
    pick_top_clips,
)


def _make_job(payload: dict[str, Any], job_id: int = 7) -> Job:
    return Job(
        id=job_id,
        kind="analyze",
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
def stub_heavy_deps(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace ffmpeg / whisper / scenedetect with controllable stubs."""
    state: dict[str, Any] = {
        "extract_calls": 0,
        "transcribe_calls": 0,
        "scenes_calls": 0,
        "fake_segments": [
            _Segment(0.0, 5.0, "First scene with lots of dialog right here."),
            _Segment(5.0, 10.0, "More words."),
            _Segment(60.0, 65.0, "Quiet stretch in the middle."),
            _Segment(120.0, 125.0, "Final big talky moment with payoff."),
        ],
        "fake_scenes": [
            _Scene(0.0, 30.0),
            _Scene(30.0, 90.0),
            _Scene(90.0, 150.0),
        ],
        "fake_lang": "en",
    }

    def fake_extract(src: Path, dst: Path) -> None:
        state["extract_calls"] += 1
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"\x00wav")

    def fake_transcribe(audio: Path, *, language: Any) -> Any:
        state["transcribe_calls"] += 1
        return state["fake_segments"], state["fake_lang"]

    def fake_scenes(src: Path) -> Any:
        state["scenes_calls"] += 1
        return state["fake_scenes"]

    monkeypatch.setattr(analyzer_mod, "extract_audio", fake_extract)
    monkeypatch.setattr(analyzer_mod, "transcribe", fake_transcribe)
    monkeypatch.setattr(analyzer_mod, "detect_scenes", fake_scenes)
    return state


async def test_falls_back_to_heuristic_without_api_key(
    fake_source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(analyzer_mod._config, "OPENROUTER_API_KEY", "")

    worker = AnalyzerWorker(MagicMock())
    result = await worker.process(
        _make_job({"source_path": str(fake_source), "duration_s": 150.0})
    )

    assert result["analysis_method"] == "heuristic"
    assert result["language"] == "en"
    assert len(result["clips"]) <= 3
    # All clips lie within the source duration.
    for clip in result["clips"]:
        assert 0 <= clip["start_s"] < clip["end_s"] <= 150.0
    # Transcript + scenes JSONs got written.
    assert Path(result["transcript_path"]).exists()
    assert Path(result["scenes_path"]).exists()


async def test_uses_llm_ranker_when_key_present(
    fake_source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(analyzer_mod._config, "OPENROUTER_API_KEY", "sk-test")

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
                                    "picks": [
                                        {"id": 0, "hook": "wow opener"},
                                        {"id": 1, "hook": "mid moment"},
                                        {"id": 2, "hook": "killer ending"},
                                    ]
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

    worker = AnalyzerWorker(MagicMock())
    result = await worker.process(
        _make_job({"source_path": str(fake_source), "duration_s": 150.0})
    )

    assert result["analysis_method"] == "llm"
    assert len(result["clips"]) == 3
    hooks = [c["hook"] for c in result["clips"]]
    assert "wow opener" in hooks
    assert captured["url"].endswith("/chat/completions")
    auth = captured["kwargs"]["headers"]["Authorization"]
    assert auth == "Bearer sk-test"


async def test_clip_windows_are_within_bounds(
    fake_source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(analyzer_mod._config, "OPENROUTER_API_KEY", "")

    worker = AnalyzerWorker(MagicMock())
    result = await worker.process(
        _make_job(
            {
                "source_path": str(fake_source),
                "duration_s": 150.0,
                "clip_min_s": 15.0,
                "clip_max_s": 45.0,
            }
        )
    )

    for clip in result["clips"]:
        length = clip["end_s"] - clip["start_s"]
        assert length >= 15.0 - 1e-6
        assert length <= 45.0 + 1e-6


async def test_payload_passes_through_to_editor(
    fake_source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``enqueue_next`` produces one edit job per detected clip."""
    monkeypatch.setattr(analyzer_mod._config, "OPENROUTER_API_KEY", "")

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

    worker = AnalyzerWorker(_FakeQueue())  # type: ignore[arg-type]
    job = _make_job({"source_path": str(fake_source), "duration_s": 150.0})
    result = await worker.process(job)
    await worker.enqueue_next(job, result)

    assert len(enqueued) == len(result["clips"])
    for kind, payload, meta in enqueued:
        assert kind == "edit"
        assert payload["source_path"] == str(fake_source)
        assert "start_s" in payload
        assert "end_s" in payload
        assert "hook" in payload
        assert meta["parent_id"] == job.id
        assert meta["chat_id"] == job.chat_id


async def test_missing_source_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = AnalyzerWorker(MagicMock())
    with pytest.raises(FileNotFoundError):
        await worker.process(
            _make_job({"source_path": "/no/such/path.mp4", "duration_s": 10.0})
        )


# ---- pure-function tests (no Worker, no mocks) ---------------------------


def test_build_candidate_windows_uses_scenes() -> None:
    scenes = [_Scene(0.0, 40.0), _Scene(40.0, 100.0), _Scene(100.0, 110.0)]
    windows = build_candidate_windows(
        scenes=scenes, duration_s=110.0, clip_min=15.0, clip_max=45.0
    )
    # Third scene (10s) is too short.
    assert len(windows) == 2
    for w in windows:
        assert w.end_s - w.start_s >= 15.0
        assert w.end_s - w.start_s <= 45.0


def test_build_candidate_windows_falls_back_to_strides() -> None:
    """When pyscenedetect returns nothing, walk the file in clip_max strides."""
    windows = build_candidate_windows(
        scenes=[], duration_s=120.0, clip_min=15.0, clip_max=30.0
    )
    assert len(windows) == 4  # 0..30, 30..60, 60..90, 90..120
    assert windows[0].start_s == 0.0
    assert windows[-1].end_s == 120.0


def test_pick_top_clips_returns_empty_for_no_candidates() -> None:
    clips, method = pick_top_clips(candidates=[], segments=[], target_count=3)
    assert clips == []
    assert method == "heuristic"


def test_heuristic_prefers_dense_transcript_windows() -> None:
    candidates = [
        _Scene(0.0, 30.0),  # talky
        _Scene(60.0, 90.0),  # silent
    ]
    segments = [
        _Segment(5.0, 10.0, "lots and lots of words said quickly here today."),
        _Segment(10.0, 15.0, "and even more text in this clip than the other one."),
    ]
    clips, method = pick_top_clips(
        candidates=candidates, segments=segments, target_count=1
    )
    assert method == "heuristic"
    assert len(clips) == 1
    assert isinstance(clips[0], _Clip)
    assert clips[0].start_s == 0.0  # the talky scene wins
