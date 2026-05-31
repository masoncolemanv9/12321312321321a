"""Optional clip thumbnail extractor.

``final_spec_FULL.md`` §9.14: thumbnail extraction is optional and
classified ``warn only`` in the failure policy. Callers therefore wrap
this in a try/except and only attach the result if it succeeded.
"""

from __future__ import annotations

import logging
import subprocess  # noqa: S404 — argv-list only
from pathlib import Path

logger = logging.getLogger(__name__)


class ThumbnailError(RuntimeError):
    """Raised when thumbnail extraction fails. Warn-only per spec."""


def extract_thumbnail(
    source_path: Path,
    output_path: Path,
    *,
    timestamp_s: float | None = None,
    ffmpeg_bin: str = "ffmpeg",
    timeout_s: float = 60.0,
) -> Path:
    """Render a single PNG/JPEG frame from ``source_path``.

    Picks ``timestamp_s`` (or the middle of the file if None) and
    writes a single frame to ``output_path``. Format is inferred from
    ``output_path``'s suffix.

    Returns ``output_path`` on success; raises :class:`ThumbnailError`
    on any failure.
    """
    if not source_path.exists():
        raise ThumbnailError(f"source not found: {source_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    seek_args: list[str] = []
    if timestamp_s is not None and timestamp_s > 0:
        seek_args = ["-ss", f"{timestamp_s:.3f}"]

    argv: list[str] = [
        ffmpeg_bin,
        "-y",
        "-loglevel",
        "error",
        *seek_args,
        "-i",
        str(source_path),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(output_path),
    ]
    try:
        subprocess.run(  # noqa: S603 — argv list, no shell
            argv, check=True, capture_output=True, timeout=timeout_s
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise ThumbnailError(f"ffmpeg thumbnail failed: {exc}") from exc
    except FileNotFoundError as exc:
        raise ThumbnailError(f"ffmpeg not found: {ffmpeg_bin}") from exc

    if not output_path.exists():
        raise ThumbnailError(f"thumbnail file missing after ffmpeg: {output_path}")
    return output_path
