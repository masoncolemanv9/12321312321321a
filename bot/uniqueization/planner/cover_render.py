"""Publisher cover render-jobs handler (final_spec §11.2).

Round-trip:

* Editor exports ``clip.mp4`` + ``frame_exports/``.
* Publisher decides per-platform crop + text overlay, emits
  ``publish_pack.json`` with ``render_jobs[]``.
* The editor-renderer (i.e. the worker) calls
  :func:`build_cover_render_jobs` to materialise an ffmpeg argv per
  job. This module is deliberately a *separate* code path from
  :mod:`bot.uniqueization.planner.creative_planner` (§11.2 last
  paragraph) — direct ffmpeg invocation, deterministic parameters,
  no LLM.

Implementation policy:

* Crop strategies are a *closed enum*. The mapping is published as
  :data:`CROP_STRATEGIES` so tests can assert it.
* Text overlay parameters MUST be sanitised (only ASCII / Cyrillic
  / common punctuation; no shell expansions) so the resulting ffmpeg
  argv is safe.
* Output paths are resolved against a base directory the caller
  supplies; the caller is responsible for ensuring the directory
  exists and is writable.

This module is *pure* — no ffmpeg is invoked here. Callers that
actually want to render call ``run_ffmpeg_atomic`` from
:mod:`bot.uniqueization.runner`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = (
    "CROP_STRATEGIES",
    "PLATFORMS",
    "CoverRenderJobError",
    "CoverRenderJob",
    "CoverRenderPlan",
    "parse_render_jobs",
    "build_cover_render_argv",
    "build_cover_render_plan",
)


# Closed enum of supported per-platform crop strategies. Each maps
# to an ffmpeg ``crop=W:H:X:Y`` recipe expressed as a function of
# the input frame size (width, height).
CROP_STRATEGIES: tuple[str, ...] = (
    "center_face_16_9",
    "center_face_1_1",
    "center_face_9_16",
    "letterbox_16_9",
    "letterbox_1_1",
    "letterbox_9_16",
    "cover_full",
)

PLATFORMS: tuple[str, ...] = (
    "youtube",
    "tiktok",
    "reels",
    "instagram",
    "x",
)


# Characters allowed in overlay text. Reject anything else so the
# argv passed to ffmpeg is safe (we still single-quote the value
# below, but defence-in-depth).
_TEXT_OVERLAY_PATTERN = re.compile(
    r"^[\w\s.,!?'\"\-:;()\[\]/А-Яа-я—–…@#$%&*+=]*$",
    re.UNICODE,
)


class CoverRenderJobError(RuntimeError):
    """Raised on malformed ``render_jobs[]`` input."""


@dataclass(frozen=True, slots=True)
class TextOverlay:
    """One §11.2 text overlay block."""

    text: str
    font: str = "InterBold"
    color: str = "#FFFFFF"
    stroke: str = "#000000"
    y_position: float = 0.10


@dataclass(frozen=True, slots=True)
class CoverRenderJob:
    """One §11.2 cover render job."""

    job_type: str
    base_frame: Path
    platform: str
    crop_strategy: str
    out_path: Path
    text_overlay: TextOverlay | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CoverRenderPlan:
    """Aggregated render-jobs ready for ffmpeg execution."""

    jobs: tuple[CoverRenderJob, ...] = ()
    warnings: tuple[str, ...] = ()


def _normalise_text(text: str) -> str:
    """Strip control bytes; reject text outside the safe character set."""
    text = (text or "").strip()
    if not text:
        return ""
    if not _TEXT_OVERLAY_PATTERN.match(text):
        raise CoverRenderJobError(
            f"text_overlay.text contains unsupported characters: {text!r}"
        )
    return text


def _parse_text_overlay(raw: Any) -> TextOverlay | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise CoverRenderJobError("text_overlay must be a mapping")
    text = _normalise_text(str(raw.get("text", "")))
    if not text:
        return None
    font = str(raw.get("font", "InterBold")) or "InterBold"
    color = str(raw.get("color", "#FFFFFF")) or "#FFFFFF"
    stroke = str(raw.get("stroke", "#000000")) or "#000000"
    try:
        y_position = float(raw.get("y_position", 0.10) or 0.10)
    except (TypeError, ValueError):
        y_position = 0.10
    y_position = max(0.0, min(0.95, y_position))
    return TextOverlay(
        text=text,
        font=font,
        color=color,
        stroke=stroke,
        y_position=y_position,
    )


def parse_render_jobs(
    raw_jobs: Sequence[Mapping[str, Any]],
    *,
    base_dir: Path | None = None,
) -> CoverRenderPlan:
    """Parse a publisher ``render_jobs[]`` list into a :class:`CoverRenderPlan`.

    Unknown ``job_type`` values are filtered out with a warning; this
    way the editor-renderer can ignore jobs it doesn't know how to
    handle (forward-compat — publisher may emit future job types).
    """
    base_dir = base_dir or Path(".")
    jobs: list[CoverRenderJob] = []
    warnings: list[str] = []

    for idx, raw in enumerate(raw_jobs):
        if not isinstance(raw, Mapping):
            warnings.append(f"job_{idx}_skipped:not_mapping")
            continue
        job_type = str(raw.get("job_type", "")).strip()
        if job_type != "cover_render":
            warnings.append(f"job_{idx}_skipped:job_type={job_type or '<empty>'}")
            continue

        base_frame_raw = raw.get("base_frame")
        if not isinstance(base_frame_raw, str) or not base_frame_raw:
            warnings.append(f"job_{idx}_skipped:missing_base_frame")
            continue
        base_frame = Path(base_frame_raw)
        if not base_frame.is_absolute():
            base_frame = (base_dir / base_frame).resolve()

        platform = str(raw.get("platform", "")).strip().lower()
        if platform not in PLATFORMS:
            warnings.append(
                f"job_{idx}_skipped:unsupported_platform={platform or '<empty>'}"
            )
            continue

        crop_strategy = str(raw.get("crop_strategy", "")).strip()
        if crop_strategy not in CROP_STRATEGIES:
            warnings.append(
                f"job_{idx}_skipped:unsupported_crop_strategy="
                f"{crop_strategy or '<empty>'}"
            )
            continue

        out_path_raw = raw.get("out_path")
        if not isinstance(out_path_raw, str) or not out_path_raw:
            warnings.append(f"job_{idx}_skipped:missing_out_path")
            continue
        out_path = Path(out_path_raw)
        if not out_path.is_absolute():
            out_path = (base_dir / out_path).resolve()

        try:
            overlay = _parse_text_overlay(raw.get("text_overlay"))
        except CoverRenderJobError as exc:
            warnings.append(f"job_{idx}_skipped:{exc}")
            continue

        extras = {
            k: v
            for k, v in raw.items()
            if k
            not in {
                "job_type",
                "base_frame",
                "platform",
                "crop_strategy",
                "out_path",
                "text_overlay",
            }
        }

        jobs.append(
            CoverRenderJob(
                job_type=job_type,
                base_frame=base_frame,
                platform=platform,
                crop_strategy=crop_strategy,
                out_path=out_path,
                text_overlay=overlay,
                extra=extras,
            )
        )

    return CoverRenderPlan(jobs=tuple(jobs), warnings=tuple(warnings))


def _crop_filter_for_strategy(strategy: str) -> str:
    """Map :data:`CROP_STRATEGIES` to an ffmpeg ``crop=`` expression."""
    # The expressions use ffmpeg variables ``iw`` / ``ih`` so output
    # is correct regardless of input frame size.
    if strategy == "center_face_16_9":
        return "crop='min(iw,ih*16/9):min(ih,iw*9/16):(iw-out_w)/2:(ih-out_h)/2'"
    if strategy == "center_face_1_1":
        return "crop='min(iw,ih):min(iw,ih):(iw-out_w)/2:(ih-out_h)/2'"
    if strategy == "center_face_9_16":
        return "crop='min(iw,ih*9/16):min(ih,iw*16/9):(iw-out_w)/2:(ih-out_h)/2'"
    if strategy == "letterbox_16_9":
        return (
            "pad='if(gt(a,16/9),iw,ih*16/9):if(gt(a,16/9),iw*9/16,ih):"
            "(ow-iw)/2:(oh-ih)/2:black'"
        )
    if strategy == "letterbox_1_1":
        return "pad='max(iw,ih):max(iw,ih):(ow-iw)/2:(oh-ih)/2:black'"
    if strategy == "letterbox_9_16":
        return (
            "pad='if(gt(a,9/16),iw,ih*9/16):if(gt(a,9/16),iw*16/9,ih):"
            "(ow-iw)/2:(oh-ih)/2:black'"
        )
    # cover_full → no crop, full frame.
    return "null"


def _drawtext_filter(overlay: TextOverlay) -> str:
    """Build a deterministic ffmpeg ``drawtext`` expression."""
    # Single-quote the text and escape colons / apostrophes which are
    # ffmpeg-significant.
    escaped = (
        overlay.text.replace("\\", r"\\")
        .replace(":", r"\:")
        .replace("'", r"\'")
    )
    return (
        "drawtext=text='" + escaped + "'"
        f":fontfile='{overlay.font}.ttf'"
        f":fontcolor={overlay.color}"
        f":bordercolor={overlay.stroke}:borderw=4"
        f":x=(w-text_w)/2:y=h*{overlay.y_position:.3f}"
    )


def build_cover_render_argv(
    job: CoverRenderJob,
    *,
    ffmpeg_bin: str = "ffmpeg",
    jpeg_quality: int = 3,
) -> list[str]:
    """Return the ffmpeg argv for a single :class:`CoverRenderJob`."""
    filters = [_crop_filter_for_strategy(job.crop_strategy)]
    if job.text_overlay is not None:
        filters.append(_drawtext_filter(job.text_overlay))
    vf = ",".join(f for f in filters if f and f != "null") or "null"
    return [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(job.base_frame),
        "-vf",
        vf,
        "-frames:v",
        "1",
        "-q:v",
        str(jpeg_quality),
        str(job.out_path),
    ]


def build_cover_render_plan(
    publish_pack: Mapping[str, Any],
    *,
    base_dir: Path | None = None,
) -> CoverRenderPlan:
    """Convenience wrapper: pull ``render_jobs[]`` from a publish_pack dict."""
    raw = publish_pack.get("render_jobs")
    if not isinstance(raw, list):
        return CoverRenderPlan(jobs=(), warnings=("missing_render_jobs",))
    return parse_render_jobs(raw, base_dir=base_dir)
