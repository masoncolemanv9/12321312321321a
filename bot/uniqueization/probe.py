"""Async :mod:`ffprobe` wrapper used by Editor Agent v2+.

Returns a typed :class:`ProbeResult` so call sites don't have to parse
``ffprobe -of json`` output themselves. See ``final_spec_FULL.md`` §9.5
step 2 for the role of this stage in the v2.0 pipeline.

This module deliberately stays narrow — it answers only the questions
the rest of the package needs (duration, dimensions, fps, audio
presence, codec names). Add more keys here when a downstream module
needs them; don't reparse ffprobe output ad-hoc elsewhere.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ProbeError(RuntimeError):
    """Raised when ``ffprobe`` is missing, exits non-zero, or returns
    unparseable JSON."""


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """Subset of ``ffprobe`` data the Editor Agent v2+ pipeline relies on."""

    path: Path
    duration_s: float
    width: int
    height: int
    fps: float
    has_audio: bool
    video_codec: str = ""
    audio_codec: str = ""
    sample_aspect_ratio: str = ""
    rotation_degrees: int = 0
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def aspect_ratio(self) -> float:
        """Width / height as a float (>1 = landscape, <1 = portrait)."""
        if self.height <= 0:
            return 0.0
        return self.width / self.height

    @property
    def is_portrait(self) -> bool:
        return self.height > self.width

    @property
    def is_landscape(self) -> bool:
        return self.width > self.height

    @property
    def is_square(self) -> bool:
        return self.width == self.height


def _parse_fps(stream: Mapping[str, Any]) -> float:
    """Decode ``avg_frame_rate`` (e.g. ``"30000/1001"``) into a float."""
    raw = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/0"
    try:
        num, _, den = raw.partition("/")
        n = float(num)
        d = float(den) if den else 1.0
        if d == 0.0:
            return 0.0
        return n / d
    except (ValueError, ZeroDivisionError):
        return 0.0


def _parse_rotation(stream: Mapping[str, Any]) -> int:
    """Extract ``rotate`` tag (legacy) or ``side_data_list`` rotation."""
    tags = stream.get("tags") or {}
    raw = tags.get("rotate")
    if raw is not None:
        try:
            return int(raw)
        except (TypeError, ValueError):
            pass
    side_data = stream.get("side_data_list") or []
    for entry in side_data:
        if isinstance(entry, dict) and "rotation" in entry:
            try:
                return int(entry["rotation"])
            except (TypeError, ValueError):
                continue
    return 0


def _select_stream(
    streams: list[Mapping[str, Any]], codec_type: str
) -> Mapping[str, Any] | None:
    for s in streams:
        if s.get("codec_type") == codec_type:
            return s
    return None


def parse_probe_json(raw_json: str | bytes, path: Path) -> ProbeResult:
    """Convert a raw ``ffprobe -of json`` payload into :class:`ProbeResult`.

    Exposed for unit tests so they can drive the parser without a real
    ``ffprobe`` binary on the box.
    """
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ProbeError(f"ffprobe returned non-JSON for {path}: {exc}") from exc

    streams = data.get("streams") or []
    fmt = data.get("format") or {}

    video = _select_stream(streams, "video")
    if video is None:
        raise ProbeError(f"ffprobe found no video stream in {path}")

    audio = _select_stream(streams, "audio")

    try:
        duration_s = float(fmt.get("duration") or video.get("duration") or 0.0)
    except (TypeError, ValueError):
        duration_s = 0.0

    try:
        width = int(video.get("width") or 0)
        height = int(video.get("height") or 0)
    except (TypeError, ValueError) as exc:
        raise ProbeError(f"invalid dimensions in {path}: {exc}") from exc

    return ProbeResult(
        path=path,
        duration_s=duration_s,
        width=width,
        height=height,
        fps=_parse_fps(video),
        has_audio=audio is not None,
        video_codec=str(video.get("codec_name") or ""),
        audio_codec=str((audio or {}).get("codec_name") or ""),
        sample_aspect_ratio=str(video.get("sample_aspect_ratio") or ""),
        rotation_degrees=_parse_rotation(video),
        raw=data,
    )


async def probe_source(
    path: Path | str,
    *,
    ffprobe_bin: str = "ffprobe",
    timeout_s: float = 30.0,
) -> ProbeResult:
    """Run ``ffprobe -show_format -show_streams -of json`` on ``path``.

    ``ffprobe`` must be on ``PATH``. Times out after ``timeout_s``
    seconds, killing the subprocess. Translates every non-success outcome
    into :class:`ProbeError`.
    """
    p = Path(path)
    if not p.exists():
        raise ProbeError(f"source not found: {p}")
    if shutil.which(ffprobe_bin) is None:
        raise ProbeError(f"{ffprobe_bin} not found on PATH")

    argv = [
        ffprobe_bin,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(p),
    ]
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except TimeoutError as exc:
        proc.kill()
        await proc.wait()
        raise ProbeError(f"ffprobe timed out after {timeout_s}s on {p}") from exc

    if proc.returncode != 0:
        raise ProbeError(
            f"ffprobe exited {proc.returncode} on {p}: {stderr.decode(errors='replace')}"
        )

    return parse_probe_json(stdout, p)
