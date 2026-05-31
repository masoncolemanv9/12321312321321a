"""Unit tests for the Part 1 (foundations) layer of the Editor Agent v2.

Covers ``bot.uniqueization.*`` except face-detection / filtergraph (those
are Part 2). The whole suite runs without invoking real ``ffmpeg`` /
``ffprobe`` — heavy work is monkeypatched following the existing repo
convention in ``tests/test_editor_worker.py``.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from bot.uniqueization import (
    AssetError,
    FfmpegError,
    FfmpegTimeoutError,
    ManifestStage,
    PhashError,
    ProbeError,
    UniqManifest,
    config_origin,
    ensure_yunet_model,
    hamming_distance,
    load_config,
    resolve_profile,
    run_ffmpeg_atomic,
    schemas_dir,
    write_manifest_atomic,
    yunet_model_path,
)
from bot.uniqueization.config import apply_profile
from bot.uniqueization.probe import parse_probe_json, probe_source

# ---------- helpers --------------------------------------------------


def _clear_editor_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop every ``EDITOR_*`` env var so :func:`load_config` only sees
    its hardcoded defaults."""
    for key in list(os.environ):
        if key.startswith("EDITOR_"):
            monkeypatch.delenv(key, raising=False)


# ---------- config ---------------------------------------------------


def test_load_config_returns_defaults_when_env_blank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_editor_env(monkeypatch)
    cfg = load_config()
    assert cfg.editor_version == "v1"
    assert cfg.editor_profile == "light"
    assert cfg.ffmpeg_timeout_s == pytest.approx(360.0)
    assert cfg.mirror_enabled is True
    assert cfg.v6_enabled is False
    assert config_origin(cfg, "editor_version") == "default"
    assert config_origin(cfg, "mirror_enabled") == "default"


def test_load_config_reads_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_editor_env(monkeypatch)
    monkeypatch.setenv("EDITOR_VERSION", "V2")
    monkeypatch.setenv("EDITOR_PROFILE", "Heavy")
    monkeypatch.setenv("EDITOR_FFMPEG_TIMEOUT_S", "900")
    monkeypatch.setenv("EDITOR_MIRROR_ENABLED", "false")
    monkeypatch.setenv("EDITOR_ZOOM", "0.5")
    cfg = load_config()
    assert cfg.editor_version == "v2"
    assert cfg.editor_profile == "heavy"
    assert cfg.ffmpeg_timeout_s == pytest.approx(900.0)
    assert cfg.mirror_enabled is False
    assert cfg.zoom == pytest.approx(0.5)
    assert config_origin(cfg, "editor_version") == "env"


def test_overrides_beat_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_editor_env(monkeypatch)
    monkeypatch.setenv("EDITOR_ZOOM", "0.25")
    cfg = load_config(overrides={"zoom": 0.9})
    assert cfg.zoom == pytest.approx(0.9)
    assert config_origin(cfg, "zoom") == "override"


def test_profile_beats_env_but_not_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_editor_env(monkeypatch)
    monkeypatch.setenv("EDITOR_ZOOM", "0.1")
    cfg = load_config(overrides={"video_crf": 18})
    cfg = apply_profile(cfg, resolve_profile("heavy"))
    assert cfg.zoom == pytest.approx(0.45)  # heavy profile wins over env
    assert config_origin(cfg, "zoom") == "profile:heavy"
    assert cfg.video_crf == 18  # explicit override survives
    assert config_origin(cfg, "video_crf") == "override"


# ---------- profiles -------------------------------------------------


def test_known_profiles_resolve() -> None:
    light = resolve_profile("light")
    medium = resolve_profile("medium")
    heavy = resolve_profile("heavy")
    assert light.name == "light"
    assert medium.video_crf < light.video_crf  # quality goes up
    assert heavy.video_crf < medium.video_crf
    assert heavy.zoom > light.zoom


def test_unknown_profile_falls_back_to_light() -> None:
    assert resolve_profile("nonsense").name == "light"
    assert resolve_profile("").name == "light"


# ---------- probe ----------------------------------------------------


def test_parse_probe_json_extracts_core_fields(tmp_path: Path) -> None:
    fake_path = tmp_path / "x.mp4"
    fake_path.write_bytes(b"\x00")
    payload = {
        "format": {"duration": "12.34"},
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "30000/1001",
                "sample_aspect_ratio": "1:1",
                "tags": {"rotate": "0"},
            },
            {"codec_type": "audio", "codec_name": "aac"},
        ],
    }
    result = parse_probe_json(json.dumps(payload), fake_path)
    assert result.duration_s == pytest.approx(12.34)
    assert result.width == 1920
    assert result.height == 1080
    assert 29.0 < result.fps < 30.5
    assert result.has_audio is True
    assert result.video_codec == "h264"
    assert result.audio_codec == "aac"
    assert result.is_landscape is True
    assert result.is_portrait is False
    assert result.is_square is False


def test_parse_probe_json_handles_video_only(tmp_path: Path) -> None:
    fake_path = tmp_path / "x.mp4"
    payload = {
        "format": {"duration": "5.0"},
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 720,
                "height": 1280,
                "avg_frame_rate": "60/1",
            }
        ],
    }
    result = parse_probe_json(json.dumps(payload), fake_path)
    assert result.has_audio is False
    assert result.is_portrait is True


def test_parse_probe_json_rejects_streams_without_video(tmp_path: Path) -> None:
    fake_path = tmp_path / "x.mp4"
    payload = {"format": {"duration": "1.0"}, "streams": [{"codec_type": "audio"}]}
    with pytest.raises(ProbeError):
        parse_probe_json(json.dumps(payload), fake_path)


def test_parse_probe_json_handles_garbage(tmp_path: Path) -> None:
    with pytest.raises(ProbeError):
        parse_probe_json("not json", tmp_path / "x.mp4")


@pytest.mark.asyncio
async def test_probe_source_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ProbeError):
        await probe_source(tmp_path / "missing.mp4")


# ---------- runner ---------------------------------------------------


class _FakeCompleted:
    def __init__(self, stderr: bytes = b"") -> None:
        self.stderr = stderr


def test_run_ffmpeg_atomic_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out = tmp_path / "out.mp4"
    seen: dict[str, Any] = {}

    def fake_run(
        argv: list[str], **kwargs: Any
    ) -> _FakeCompleted:  # noqa: ARG001
        seen["argv"] = argv
        Path(argv[-1]).write_bytes(b"FAKEMP4")
        return _FakeCompleted(stderr=b"all good")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = run_ffmpeg_atomic(
        ["ffmpeg", "-y", "-i", "in.mp4", "-c:v", "libx264"],
        output_path=out,
        timeout_s=5.0,
        debug_dir=tmp_path / "debug",
    )
    assert result.output_path == out
    assert out.exists()
    assert b"FAKEMP4" in out.read_bytes()
    # Temp path keeps the ``.mp4`` extension so ffmpeg's muxer probe
    # still recognises the format (``foo.mp4.tmp`` fails probing).
    assert seen["argv"][-1].endswith(".tmp.mp4")
    # Atomic rename should leave NO sibling temp.
    assert not (tmp_path / "out.tmp.mp4").exists()


def test_run_ffmpeg_atomic_timeout_cleans_up(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out = tmp_path / "out.mp4"
    temp = tmp_path / "out.mp4.tmp"

    def fake_run(argv: list[str], **kwargs: Any) -> _FakeCompleted:  # noqa: ARG001
        Path(argv[-1]).write_bytes(b"PARTIAL")
        raise subprocess.TimeoutExpired(cmd=argv, timeout=1.0, stderr=b"slow")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(FfmpegTimeoutError):
        run_ffmpeg_atomic(
            ["ffmpeg", "-i", "in.mp4"],
            output_path=out,
            timeout_s=1.0,
            debug_dir=tmp_path / "debug",
        )
    assert not out.exists()
    assert not temp.exists()
    assert (tmp_path / "debug" / "ffmpeg-stderr.txt").exists()


def test_run_ffmpeg_atomic_failure_writes_stderr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out = tmp_path / "out.mp4"

    def fake_run(argv: list[str], **kwargs: Any) -> _FakeCompleted:  # noqa: ARG001
        Path(argv[-1]).write_bytes(b"PARTIAL")
        raise subprocess.CalledProcessError(
            returncode=1, cmd=argv, output=b"", stderr=b"boom"
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(FfmpegError):
        run_ffmpeg_atomic(
            ["ffmpeg", "-i", "in.mp4"],
            output_path=out,
            timeout_s=5.0,
            debug_dir=tmp_path / "debug",
        )
    assert not out.exists()
    stderr_file = tmp_path / "debug" / "ffmpeg-stderr.txt"
    assert stderr_file.exists()
    assert "boom" in stderr_file.read_text()


def test_run_ffmpeg_atomic_missing_binary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_run(argv: list[str], **kwargs: Any) -> _FakeCompleted:  # noqa: ARG001
        raise FileNotFoundError("no ffmpeg")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(FfmpegError):
        run_ffmpeg_atomic(
            ["ffmpeg", "-i", "in.mp4"],
            output_path=tmp_path / "out.mp4",
            timeout_s=5.0,
            debug_dir=tmp_path / "debug",
        )


# ---------- manifest -------------------------------------------------


def test_manifest_atomic_write_roundtrips(tmp_path: Path) -> None:
    manifest = UniqManifest(
        schema_version=1,
        editor_version="v2",
        editor_profile="medium",
        job={"id": "job_abc", "clip_index": 0},
        source={"path": "downloads/job_abc/source.mp4", "duration_s": 25.0},
        output={"path": "clips/job_abc_clip_000.mp4", "width": 1080, "height": 1920},
        stages=(
            ManifestStage(
                name="zoom_reframe",
                enabled=True,
                status="ok",
                params={"zoom": 0.35},
                elapsed_ms=120,
            ),
            ManifestStage(
                name="mirror", enabled=False, status="skipped", reason="too_short"
            ),
        ),
        warnings=("phash_failed",),
    )
    out = tmp_path / "uniqueization.json"
    write_manifest_atomic(manifest, out)
    data = json.loads(out.read_text())
    assert data["schema_version"] == 1
    assert data["editor_version"] == "v2"
    assert len(data["stages"]) == 2
    assert data["stages"][0]["name"] == "zoom_reframe"
    assert data["warnings"] == ["phash_failed"]
    # No leftover temp file.
    assert not (tmp_path / "uniqueization.json.tmp").exists()


# ---------- phash ----------------------------------------------------


def test_hamming_distance_basic() -> None:
    assert hamming_distance("0" * 16, "0" * 16) == 0
    # Differ in low nibble only: 0x0 (0000) ^ 0xF (1111) => 4
    assert hamming_distance("0" * 16, "f" + "0" * 15) == 4
    assert hamming_distance("ff" * 8, "00" * 8) == 64


def test_hamming_distance_rejects_bad_input() -> None:
    with pytest.raises(PhashError):
        hamming_distance("ff", "ffff")
    with pytest.raises(PhashError):
        hamming_distance("zz" * 8, "00" * 8)


# ---------- assets ---------------------------------------------------


def test_schemas_dir_exists() -> None:
    sd = schemas_dir()
    assert sd.is_dir()
    assert (sd / "director_brief.schema.json").is_file()
    assert (sd / "edit_plan.schema.json").is_file()


def test_yunet_model_path_default_when_env_blank() -> None:
    default = yunet_model_path("")
    assert default.name.endswith(".onnx")
    # Default lives inside the package, not in user-supplied location.
    assert "uniqueization" in str(default)


def test_yunet_model_path_respects_env_override(tmp_path: Path) -> None:
    chosen = tmp_path / "custom-model.onnx"
    chosen.write_bytes(b"fake")
    assert yunet_model_path(str(chosen)) == chosen


def test_ensure_yunet_model_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(AssetError):
        ensure_yunet_model(tmp_path / "nope.onnx")


def test_ensure_yunet_model_verifies_sha256(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"hello")
    # sha256("hello") = 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824
    good = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    assert ensure_yunet_model(model, expected_sha256=good) == model
    with pytest.raises(AssetError):
        ensure_yunet_model(model, expected_sha256="00" * 32)


# ---------- v1 invariance --------------------------------------------


def test_default_editor_version_is_v1(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sanity: with no env, the resolved version stays at v1 so the
    existing legacy worker keeps running. This is the contract that lets
    all 10 v2 parts land without breaking the production pipeline."""
    _clear_editor_env(monkeypatch)
    cfg = load_config()
    assert cfg.editor_version == "v1"
    assert cfg.v6_enabled is False


def test_replace_keeps_origin_for_unchanged_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_editor_env(monkeypatch)
    cfg = load_config()
    cfg2 = replace(cfg, editor_profile="medium")
    # ``_origin`` is structural; dataclasses.replace keeps the same map.
    assert cfg._origin == cfg2._origin
