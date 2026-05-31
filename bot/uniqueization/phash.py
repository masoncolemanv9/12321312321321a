"""Perceptual-hash sampler used by the v2.0 manifest's
``signature_metrics`` block (``final_spec_FULL.md`` §9.14).

Sampling policy: 10 / 25 / 50 / 75 / 90 percent of clip duration.
Five frames balance outlier resistance with CPU cost. Each sample
returns an ``imagehash.ImageHash``; downstream code computes Hamming
distance against a paired source sample to expose drift in the
manifest.

Failures here are **observability-only** (§9.13 failure policy
classifies pHash / thumbnail as ``warn only``). Importers therefore
swallow :class:`PhashError` and record a warning rather than aborting
the render.

Optional deps (``Pillow``, ``imagehash``) are pulled in by the
``editor-v2`` extras group from §13.3. They are imported lazily so
``EDITOR_VERSION=v1`` deploys can skip the install entirely.
"""

from __future__ import annotations

import logging
import subprocess  # noqa: S404 — argv-list only
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_PERCENTS: tuple[int, ...] = (10, 25, 50, 75, 90)


class PhashError(RuntimeError):
    """Raised when pHash sampling fails. Callers must treat as warn-only."""


@dataclass(frozen=True, slots=True)
class PhashSample:
    """One sampled pHash row."""

    percent: int
    timestamp_s: float
    hash_hex: str

    def __post_init__(self) -> None:
        if not self.hash_hex:
            raise PhashError(f"empty hash for percent={self.percent}")


def _ensure_imagehash() -> Any:
    """Lazy import; raises :class:`PhashError` if extras not installed."""
    try:
        import imagehash
        from PIL import Image
    except ImportError as exc:
        raise PhashError(
            "imagehash / Pillow not installed (pip install '.[editor-v2]')"
        ) from exc
    return imagehash, Image


def _sample_timestamps(duration_s: float, percents: Sequence[int]) -> list[float]:
    return [max(0.0, duration_s * (p / 100.0)) for p in percents]


def extract_phash_samples(
    source_path: Path,
    *,
    duration_s: float,
    percents: Sequence[int] = DEFAULT_PERCENTS,
    ffmpeg_bin: str = "ffmpeg",
) -> list[PhashSample]:
    """Extract perceptual-hash samples for ``source_path``.

    Runs one short ``ffmpeg`` invocation per sample (cheap; the seek is
    fast-forward, no full decode). Returns one entry per percent in
    input order.

    :raises PhashError: when extras are missing, ffmpeg fails, or
        every sample failed. Partial failures (some succeeded) return
        only the successful subset.
    """
    if duration_s <= 0:
        raise PhashError(f"non-positive duration {duration_s} for {source_path}")

    imagehash, image_mod = _ensure_imagehash()
    timestamps = _sample_timestamps(duration_s, percents)

    samples: list[PhashSample] = []
    with tempfile.TemporaryDirectory(prefix="uniq-phash-") as tmpdir:
        for percent, ts in zip(percents, timestamps, strict=True):
            png = Path(tmpdir) / f"phash_{percent}.png"
            argv: list[str] = [
                ffmpeg_bin,
                "-y",
                "-loglevel",
                "error",
                "-ss",
                f"{ts:.3f}",
                "-i",
                str(source_path),
                "-frames:v",
                "1",
                "-q:v",
                "2",
                str(png),
            ]
            try:
                subprocess.run(  # noqa: S603 — argv list, no shell
                    argv,
                    check=True,
                    capture_output=True,
                    timeout=60.0,
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                logger.warning(
                    "phash sample failed at %.2fs (%d%%): %s",
                    ts,
                    percent,
                    exc,
                )
                continue
            except FileNotFoundError as exc:
                raise PhashError(f"ffmpeg binary not found: {ffmpeg_bin}") from exc

            try:
                with image_mod.open(png) as img:
                    h = imagehash.phash(img)
            except Exception as exc:  # noqa: BLE001 — Pillow/imagehash raise variants
                logger.warning("phash compute failed at %d%%: %s", percent, exc)
                continue
            samples.append(
                PhashSample(percent=percent, timestamp_s=ts, hash_hex=str(h))
            )

    if not samples:
        raise PhashError(f"all phash samples failed for {source_path}")
    return samples


def hamming_distance(a: str, b: str) -> int:
    """Hex-pHash Hamming distance, agnostic of imagehash being installed.

    Both inputs must be the same hex length (imagehash's ``phash``
    returns 16-char strings for 64-bit hashes). Used to score
    source/output drift for the manifest's ``phash_distances`` row.
    """
    if len(a) != len(b):
        raise PhashError(f"hash length mismatch: {len(a)} != {len(b)}")
    try:
        ai = int(a, 16)
        bi = int(b, 16)
    except ValueError as exc:
        raise PhashError(f"non-hex phash input: {a!r} / {b!r}") from exc
    return (ai ^ bi).bit_count()
