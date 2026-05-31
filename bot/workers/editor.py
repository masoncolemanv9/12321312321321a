"""Editor stage: cut + crop + scale a window into a 1080×1920 short.

Pipeline (single ffmpeg invocation per clip):

1. ``-ss start_s -to end_s`` extracts the chosen window.
2. ``crop=ih*9/16:ih`` keeps a centered vertical 9:16 strip.
3. ``scale=W:H`` normalizes to ``EDITOR_OUTPUT_WIDTH × EDITOR_OUTPUT_HEIGHT``.
4. (Optional) ``overlay`` burns a logo into the bottom-right corner.
5. Re-encode with ``libx264`` (CRF) + AAC for compatibility on every
   destination platform.

The output lives at ``<source_dir>/clips/<job_id>_<idx>.mp4`` so the
publisher stage can find it without extra plumbing.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from pathlib import Path
from typing import Any

from .. import config as _config
from ..jobs import Job
from .base import Worker

logger = logging.getLogger(__name__)


class EditorWorker(Worker):
    kind = "edit"
    next_kind = "seo"

    async def process(self, job: Job) -> dict[str, Any]:
        source_path_str = job.payload.get("source_path")
        if not source_path_str:
            raise ValueError("edit payload missing 'source_path'")
        source_path = Path(source_path_str)
        if not source_path.exists():
            raise FileNotFoundError(f"edit source not found: {source_path}")

        start_s = float(job.payload.get("start_s") or 0.0)
        end_s = float(job.payload.get("end_s") or 0.0)
        if end_s <= start_s:
            raise ValueError(
                f"edit payload has non-positive window: {start_s}..{end_s}"
            )

        clip_index = int(job.payload.get("clip_index", 0))
        hook = str(job.payload.get("hook", ""))

        clips_dir = source_path.parent / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)
        clip_path = clips_dir / f"{job.id}_{clip_index}.mp4"

        return await asyncio.to_thread(
            self._render_blocking,
            source_path=source_path,
            clip_path=clip_path,
            start_s=start_s,
            end_s=end_s,
            clip_index=clip_index,
            hook=hook,
        )

    @staticmethod
    def _render_blocking(
        *,
        source_path: Path,
        clip_path: Path,
        start_s: float,
        end_s: float,
        clip_index: int,
        hook: str,
    ) -> dict[str, Any]:
        cmd, overlay_used = build_ffmpeg_command(
            source_path=source_path,
            clip_path=clip_path,
            start_s=start_s,
            end_s=end_s,
        )
        run_ffmpeg(cmd)
        return {
            "source_path": str(source_path),
            "clip_index": clip_index,
            "clip_path": str(clip_path),
            "duration_s": end_s - start_s,
            "width": _config.EDITOR_OUTPUT_WIDTH,
            "height": _config.EDITOR_OUTPUT_HEIGHT,
            "hook": hook,
            "overlay_used": overlay_used,
        }


# ---- helpers -------------------------------------------------------------


def _logo_path() -> Path | None:
    """Resolve the optional overlay logo if it actually exists on disk."""
    raw = _config.OVERLAY_LOGO_PATH
    if not raw:
        return None
    path = Path(raw)
    if not path.exists():
        logger.warning("OVERLAY_LOGO_PATH=%s does not exist; skipping overlay", path)
        return None
    return path


def build_ffmpeg_command(
    *,
    source_path: Path,
    clip_path: Path,
    start_s: float,
    end_s: float,
) -> tuple[list[str], bool]:
    """Construct the ffmpeg argv for one clip render.

    Returns ``(cmd, overlay_used)``. ``overlay_used`` reflects whether
    a real logo file was attached.
    """
    width = _config.EDITOR_OUTPUT_WIDTH
    height = _config.EDITOR_OUTPUT_HEIGHT
    crop_filter = compute_crop_filter(width=width, height=height)
    scale_filter = f"scale={width}:{height}"

    logo = _logo_path()
    cmd: list[str] = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-ss",
        f"{start_s:.3f}",
        "-to",
        f"{end_s:.3f}",
        "-i",
        str(source_path),
    ]
    if logo is not None:
        cmd += ["-i", str(logo)]
        margin = _config.OVERLAY_MARGIN_PX
        # First chain: crop+scale the source into [base]; second chain:
        # overlay [base] with the second input at bottom-right.
        filter_complex = (
            f"[0:v]{crop_filter},{scale_filter},setsar=1[base];"
            f"[base][1:v]overlay=W-w-{margin}:H-h-{margin}[outv]"
        )
        cmd += [
            "-filter_complex",
            filter_complex,
            "-map",
            "[outv]",
            "-map",
            "0:a?",
        ]
        overlay_used = True
    else:
        cmd += [
            "-vf",
            f"{crop_filter},{scale_filter},setsar=1",
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
        ]
        overlay_used = False

    cmd += [
        "-c:v",
        "libx264",
        "-preset",
        _config.EDITOR_VIDEO_PRESET,
        "-crf",
        str(_config.EDITOR_VIDEO_CRF),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        _config.EDITOR_AUDIO_BITRATE,
        "-movflags",
        "+faststart",
        str(clip_path),
    ]
    return cmd, overlay_used


def compute_crop_filter(*, width: int, height: int) -> str:
    """Center-crop a 9:16 vertical strip out of the source frame.

    For 9:16 output, the crop width = input_height * 9/16 (assuming
    landscape input). This is centered horizontally.
    """
    aspect = f"{width}/{height}"  # e.g. "1080/1920" → 9:16
    return f"crop=ih*{aspect}:ih"


def run_ffmpeg(cmd: list[str]) -> None:
    """Run ffmpeg, raising :class:`RuntimeError` with a friendly message."""
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
        raise RuntimeError(f"ffmpeg failed: {stderr.strip()}") from exc
