"""``FfmpegRunner`` — atomic, timeout-protected ffmpeg subprocess driver.

Implements the contract in ``final_spec_FULL.md`` §9.11:

1. ``argv`` is always ``list[str]`` (never ``shell=True``).
2. Renders to ``output_path.with_suffix(".tmp.mp4")`` then
   ``os.replace`` on success.
3. Failure / timeout deletes the temp file (no partial mp4 left).
4. ``stderr`` is written to ``debug_dir/ffmpeg-stderr.txt`` on failure
   or when ``EDITOR_KEEP_UNIQ_ARTIFACTS=true``.
5. Uses ``subprocess.run(check=True, capture_output=True, timeout=...)``.
6. Default timeout per call site: ``max(config.ffmpeg_timeout_s,
   (end_s - start_s) * 6)``.

Used by:

* :class:`bot.workers.editor_v2.EditorV2Worker` (Part 3) for the v2.0
  fused encode.
* :mod:`bot.uniqueization.phash` / :mod:`.thumbnail` for sampled frame
  extraction.
* Various v6 renderer adapters in Part 10.
"""

from __future__ import annotations

import logging
import os
import subprocess  # noqa: S404 — argv-list usage only, never shell=True
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


class FfmpegError(RuntimeError):
    """Raised when ``ffmpeg`` exits non-zero. Stderr is attached as
    ``stderr`` for callers that want to log / classify."""

    def __init__(self, message: str, *, argv: Sequence[str], stderr: str) -> None:
        super().__init__(message)
        self.argv = list(argv)
        self.stderr = stderr


class FfmpegTimeoutError(FfmpegError):
    """Raised when the ffmpeg subprocess exceeds ``timeout_s``."""


@dataclass(frozen=True, slots=True)
class FfmpegResult:
    """What :func:`run_ffmpeg_atomic` returns on success."""

    argv: list[str]
    elapsed_ms: int
    stderr: str
    output_path: Path


def _temp_path_for(output_path: Path) -> Path:
    """Sibling temp path that ``os.replace``'s atomically onto
    ``output_path`` on success. Suffix is ``.tmp<orig_suffix>`` so
    callers that probe by extension still see ``.mp4``-ish names —
    and so ffmpeg can auto-detect the muxer by extension
    (``foo.mp4.tmp`` fails muxer probing)."""
    suffix = output_path.suffix or ""
    return output_path.with_name(output_path.stem + ".tmp" + suffix)


def _write_stderr_artifact(debug_dir: Path, stderr_text: str) -> None:
    """Best-effort dump of stderr to ``debug_dir/ffmpeg-stderr.txt``.
    Silently swallows OSError so we never lose the original ffmpeg
    failure inside the artifact-write code path."""
    try:
        debug_dir.mkdir(parents=True, exist_ok=True)
        (debug_dir / "ffmpeg-stderr.txt").write_text(
            stderr_text, encoding="utf-8", errors="replace"
        )
    except OSError as exc:
        logger.warning("failed to write ffmpeg stderr to %s: %s", debug_dir, exc)


def _delete_temp(temp: Path) -> None:
    """Idempotent unlink — used in both timeout and failure cleanup."""
    try:
        temp.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("failed to remove temp output %s: %s", temp, exc)


def run_ffmpeg_atomic(
    argv_prefix: Sequence[str],
    *,
    output_path: Path,
    timeout_s: float,
    debug_dir: Path,
    keep_artifacts: bool = False,
) -> FfmpegResult:
    """Run ffmpeg atomically; never leave a partial ``output_path``.

    ``argv_prefix`` is the ffmpeg invocation *without* the final
    positional output argument — this function appends the temp output
    path itself so callers can't accidentally skip the atomic-rename
    contract. Encoding flags must already be inside ``argv_prefix``.

    :param argv_prefix: ffmpeg argv list (input flags + ``-i ...`` +
        encoder flags). Must begin with ``"ffmpeg"`` or an absolute
        path to the ffmpeg binary.
    :param output_path: final desired path. Parent dir is created.
    :param timeout_s: hard timeout for the whole subprocess.
    :param debug_dir: where ``ffmpeg-stderr.txt`` lands on failure (and
        on success when ``keep_artifacts`` is True).
    :param keep_artifacts: persist stderr even on success
        (``EDITOR_KEEP_UNIQ_ARTIFACTS=true``).

    :raises FfmpegTimeoutError: subprocess exceeded ``timeout_s``.
    :raises FfmpegError: ffmpeg exited non-zero.
    """
    if not argv_prefix:
        raise FfmpegError("empty ffmpeg argv", argv=[], stderr="")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = _temp_path_for(output_path)
    argv: list[str] = list(argv_prefix) + [str(temp)]

    start = time.monotonic()
    try:
        completed = subprocess.run(  # noqa: S603 — argv list, no shell
            argv,
            check=True,
            capture_output=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed_ms = int((time.monotonic() - start) * 1000)
        stderr = (exc.stderr or b"").decode(errors="replace")
        _write_stderr_artifact(debug_dir, stderr or "<timeout>")
        _delete_temp(temp)
        raise FfmpegTimeoutError(
            f"ffmpeg timed out after {timeout_s}s (elapsed {elapsed_ms}ms)",
            argv=argv,
            stderr=stderr,
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or b"").decode(errors="replace")
        _write_stderr_artifact(debug_dir, stderr)
        _delete_temp(temp)
        raise FfmpegError(
            f"ffmpeg exited {exc.returncode}", argv=argv, stderr=stderr
        ) from exc
    except FileNotFoundError as exc:
        _delete_temp(temp)
        raise FfmpegError(
            f"ffmpeg binary not found: {argv[0]}", argv=argv, stderr=str(exc)
        ) from exc

    elapsed_ms = int((time.monotonic() - start) * 1000)
    stderr = completed.stderr.decode(errors="replace")
    if keep_artifacts:
        _write_stderr_artifact(debug_dir, stderr)

    try:
        os.replace(temp, output_path)
    except OSError as exc:
        _delete_temp(temp)
        _write_stderr_artifact(debug_dir, stderr or str(exc))
        raise FfmpegError(
            f"failed to move {temp} -> {output_path}: {exc}",
            argv=argv,
            stderr=stderr,
        ) from exc

    return FfmpegResult(
        argv=argv, elapsed_ms=elapsed_ms, stderr=stderr, output_path=output_path
    )
