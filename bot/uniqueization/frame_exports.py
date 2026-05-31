"""Publisher frame_exports/ writer (final_spec §11.1).

The editor-v6 worker calls :func:`write_frame_exports_metadata` after
the v2.1 / v6 render produces ``clip.mp4`` so the publisher has the
3-5 candidate thumbnails plus
``frame_exports/metadata.json`` to read.

This module does NOT extract jpegs itself — that requires an ffmpeg
invocation against the rendered clip, which is the worker's job
(see :mod:`bot.workers.editor_v2`). What we own here:

* ``select_frame_export_candidates`` — pick which timestamps to
  emit jpegs for, sourced from ``edit_plan.frame_export_hints[]``.
* ``write_frame_exports_metadata`` — emit
  ``frame_exports/metadata.json`` with the §11.1 schema.
* ``build_frame_export_argv`` — produce the ``ffmpeg`` argv string
  list that the worker should run for each candidate (extracted as
  a separate helper so it can be unit-tested without spawning
  ffmpeg).

All helpers are deterministic. The metadata writer overwrites the
file atomically (write to ``metadata.json.tmp`` then ``rename``).
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = (
    "DEFAULT_FRAMES_MIN",
    "DEFAULT_FRAMES_MAX",
    "FrameExportCandidate",
    "build_frame_export_argv",
    "select_frame_export_candidates",
    "write_frame_exports_metadata",
)


DEFAULT_FRAMES_MIN: int = 3
DEFAULT_FRAMES_MAX: int = 5


@dataclass(frozen=True, slots=True)
class FrameExportCandidate:
    """One §11.1 frame export."""

    filename: str
    source_timestamp_s: float
    beat_id: str = ""
    emotional_intent: str = ""
    face_count: int = 0
    face_bboxes: tuple[Mapping[str, float], ...] = ()
    has_text: bool = False
    suitability_score: float = 0.0


def _beat_emotional_intent(
    edit_plan: Mapping[str, Any], beat_id: str
) -> str:
    beat_sheet = edit_plan.get("beat_sheet")
    if not isinstance(beat_sheet, list):
        return ""
    for beat in beat_sheet:
        if not isinstance(beat, Mapping):
            continue
        if str(beat.get("beat_id", "")) == beat_id:
            return str(beat.get("emotional_intent", "") or "")
    return ""


def select_frame_export_candidates(
    edit_plan: Mapping[str, Any],
    *,
    min_frames: int = DEFAULT_FRAMES_MIN,
    max_frames: int = DEFAULT_FRAMES_MAX,
) -> list[FrameExportCandidate]:
    """Pick the §11.1 candidate list from ``edit_plan.frame_export_hints``.

    Hints with higher ``suitability_score`` win. Order is stable by
    ``(timestamp_s ascending)`` once selected so manifest output is
    deterministic. We always select **at least** ``min_frames`` (when
    enough hints exist) and at most ``max_frames``.
    """
    hints = edit_plan.get("frame_export_hints")
    if not isinstance(hints, list):
        return []

    cleaned: list[dict[str, Any]] = []
    for h in hints:
        if not isinstance(h, Mapping):
            continue
        try:
            ts = float(h["timestamp_s"])
        except (KeyError, TypeError, ValueError):
            continue
        cleaned.append(
            {
                "timestamp_s": ts,
                "beat_id": str(h.get("beat_id", "") or ""),
                "suitability_score": float(
                    h.get("suitability_score", 0.0) or 0.0
                ),
            }
        )

    if not cleaned:
        return []

    # Sort by suitability descending (stable on insertion order).
    cleaned.sort(key=lambda c: (-c["suitability_score"], c["timestamp_s"]))

    take_n = max(
        min(max_frames, len(cleaned)),
        min(min_frames, len(cleaned)),
    )
    chosen = cleaned[:take_n]
    chosen.sort(key=lambda c: c["timestamp_s"])

    out: list[FrameExportCandidate] = []
    for idx, h in enumerate(chosen, start=1):
        beat_id = str(h.get("beat_id", "") or "")
        emo = _beat_emotional_intent(edit_plan, beat_id) if beat_id else ""
        out.append(
            FrameExportCandidate(
                filename=f"frame_{idx:03d}.jpg",
                source_timestamp_s=float(h["timestamp_s"]),
                beat_id=beat_id,
                emotional_intent=emo,
                face_count=0,
                face_bboxes=(),
                has_text=False,
                suitability_score=float(h.get("suitability_score", 0.0)),
            )
        )
    return out


def build_frame_export_argv(
    *,
    source_clip: Path,
    timestamp_s: float,
    out_path: Path,
    ffmpeg_bin: str = "ffmpeg",
    jpeg_quality: int = 3,
) -> list[str]:
    """Return the ffmpeg argv to grab one jpeg at ``timestamp_s``.

    ``jpeg_quality`` follows ffmpeg's mjpeg semantics (2 = ~90%
    quality per §11.1; lower number = higher quality). The output is
    a single image (``-frames:v 1``). The worker is expected to
    invoke this argv via the usual atomic runner.
    """
    return [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{timestamp_s:.3f}",
        "-i",
        str(source_clip),
        "-frames:v",
        "1",
        "-q:v",
        str(jpeg_quality),
        str(out_path),
    ]


def write_frame_exports_metadata(
    *,
    clip_id: str,
    candidates: Sequence[FrameExportCandidate],
    out_dir: Path,
    filename: str = "metadata.json",
) -> Path:
    """Write ``frame_exports/metadata.json`` atomically.

    ``out_dir`` is created if missing. Returns the resolved path.
    Schema follows §11.1 verbatim.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "clip_id": clip_id,
        "frames": [
            {
                "filename": c.filename,
                "source_timestamp_s": round(c.source_timestamp_s, 3),
                "beat_id": c.beat_id,
                "emotional_intent": c.emotional_intent,
                "face_count": int(c.face_count),
                "face_bboxes": [dict(b) for b in c.face_bboxes],
                "has_text": bool(c.has_text),
                "suitability_score": round(c.suitability_score, 3),
            }
            for c in candidates
        ],
    }
    final_path = out_dir / filename
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{filename}.",
        suffix=".tmp",
        dir=str(out_dir),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=False)
            fh.write("\n")
        os.replace(tmp_path, final_path)
    except Exception:
        # Best-effort cleanup if we never reached replace().
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise
    return final_path
