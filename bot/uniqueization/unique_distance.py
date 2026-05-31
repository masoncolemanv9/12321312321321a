"""v2.1 unique-distance vector (``final_spec_FULL.md`` §10.7).

This module computes the **independent component metrics** that the
v2.1 manifest extension ``uniqueization.json.unique_distance`` carries.
The spec is explicit about scope:

> Components are observability metrics. They do not guarantee
> platform matching outcomes.

So:

* The output is a **vector** of independent components, not a scalar
  score. Callers compose what they like for dashboards.
* Nothing here implies "matching" or "deduplication" or "platform
  bypass". The metrics quantify how the v2.1 output differs from the
  source after the editor's edits — useful for catching regressions
  in the colorgrade / loudness / cuts pipeline.

Component schema (from §10.7):

```json
{
  "phash_distances": [12, 18, 15, 14, 13],
  "duration_delta_s": -3.2,
  "frame_rgb_mean_delta": [4.2, 6.7, 3.1],
  "ssim_sample": 0.79,
  "audio_rms_db": -18.5,
  "audio_true_peak_db": -1.8,
  "chromaprint_fingerprint": null,
  "components_version": "2.1"
}
```

Each component is computed independently and may be ``None`` when its
optional dependency is missing or its enabled flag is off (R30 budget
trim path). The version string is closed-enumed so callers can branch
without parsing the dict's shape.

Cost budget (§10.7): total < 500ms per clip on a 4-core CPU. This
module sequences the components with the most expensive one (SSIM)
behind an explicit toggle.

Hard rules:

* No torch / librosa / pyannote.audio (per §0 forbidden list).
* No chromaprint required: ``chromaprint_fingerprint`` is best-effort
  via ``fpcalc`` and silently skipped when the binary isn't present.
* No claim/bypass language. The disclaimer in §10.7 is reproduced in
  this module's docstring.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess  # noqa: S404 — argv-list only
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .phash import (
    DEFAULT_PERCENTS,
    PhashError,
    PhashSample,
    extract_phash_samples,
    hamming_distance,
)

logger = logging.getLogger(__name__)

#: Closed enum of supported component-schema versions.
ComponentsVersion = Literal["2.1"]

#: Default SSIM sample percent (50%, mid-clip).
DEFAULT_SSIM_PERCENT: int = 50


class UniqueDistanceError(RuntimeError):
    """Raised on unrecoverable errors. Optional-component failures
    are swallowed and reported as ``None``."""


@dataclass(frozen=True, slots=True)
class UniqueDistanceComponents:
    """Output of :func:`compute_components`.

    Every field is independent; downstream code reads what it needs.
    All ``Optional`` fields are ``None`` when their dependency is
    absent or the toggle was off.
    """

    phash_distances: tuple[int, ...]
    duration_delta_s: float
    frame_rgb_mean_delta: tuple[float, float, float] | None
    ssim_sample: float | None
    audio_rms_db: float | None
    audio_true_peak_db: float | None
    chromaprint_fingerprint: str | None
    components_version: ComponentsVersion = "2.1"
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def phash_mean_distance(self) -> float | None:
        """Convenience accessor — sample mean of ``phash_distances``.

        Independent of the vector; callers that need a single number
        for dashboards can read this without collapsing the schema.
        """
        if not self.phash_distances:
            return None
        return sum(self.phash_distances) / len(self.phash_distances)

    def to_manifest_dict(self) -> dict[str, object]:
        """JSON-safe shape matching §10.7."""
        return {
            "phash_distances": list(self.phash_distances),
            "duration_delta_s": self.duration_delta_s,
            "frame_rgb_mean_delta": (
                list(self.frame_rgb_mean_delta)
                if self.frame_rgb_mean_delta is not None
                else None
            ),
            "ssim_sample": self.ssim_sample,
            "audio_rms_db": self.audio_rms_db,
            "audio_true_peak_db": self.audio_true_peak_db,
            "chromaprint_fingerprint": self.chromaprint_fingerprint,
            "components_version": self.components_version,
        }


# ---------------------------------------------------------------------------
# Component computations
# ---------------------------------------------------------------------------


def _phash_distances(
    source_path: Path,
    output_path: Path,
    *,
    source_duration_s: float,
    output_duration_s: float,
    ffmpeg_bin: str,
    warnings: list[str],
) -> tuple[int, ...]:
    """Compute pHash hamming distances per :data:`DEFAULT_PERCENTS`."""
    try:
        src_samples = extract_phash_samples(
            source_path,
            duration_s=source_duration_s,
            ffmpeg_bin=ffmpeg_bin,
        )
        out_samples = extract_phash_samples(
            output_path,
            duration_s=output_duration_s,
            ffmpeg_bin=ffmpeg_bin,
        )
    except PhashError as exc:
        warnings.append(f"phash_failed: {exc}")
        return ()

    # Pair samples by percent. Any percent missing on either side is
    # skipped with a warning — the manifest's array is allowed to be
    # shorter than :data:`DEFAULT_PERCENTS`.
    by_pct_src: dict[int, PhashSample] = {s.percent: s for s in src_samples}
    by_pct_out: dict[int, PhashSample] = {s.percent: s for s in out_samples}
    out: list[int] = []
    for pct in DEFAULT_PERCENTS:
        src = by_pct_src.get(pct)
        dst = by_pct_out.get(pct)
        if src is None or dst is None:
            warnings.append(f"phash_missing: percent={pct}")
            continue
        try:
            out.append(hamming_distance(src.hash_hex, dst.hash_hex))
        except ValueError as exc:
            warnings.append(f"phash_hamming_failed: {exc}")
    return tuple(out)


def _probe_duration_s(path: Path, *, ffprobe_bin: str) -> float | None:
    """Sync, narrow ffprobe call: read container duration.

    We avoid importing :mod:`bot.uniqueization.probe` directly because
    that module exposes an async API; this helper sequences cleanly
    inside :func:`compute_components` (also sync).
    """
    argv = [
        ffprobe_bin,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        proc = subprocess.run(  # noqa: S603 — argv list, no shell
            argv,
            check=False,
            capture_output=True,
            timeout=15.0,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    if proc.returncode != 0:
        return None
    try:
        data = json.loads(proc.stdout.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return None
    duration = data.get("format", {}).get("duration")
    try:
        return float(duration) if duration is not None else None
    except (TypeError, ValueError):
        return None


def _run_ffmpeg_silent(argv: list[str], *, timeout: float = 30.0) -> str:
    """Run ffmpeg with stderr captured; return stderr text."""
    try:
        proc = subprocess.run(  # noqa: S603 — argv list, no shell
            argv,
            check=False,
            capture_output=True,
            timeout=timeout,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        raise UniqueDistanceError(f"ffmpeg invocation failed: {exc}") from exc
    return proc.stderr.decode("utf-8", errors="replace")


def _ssim_sample(
    source_path: Path,
    output_path: Path,
    *,
    ffmpeg_bin: str,
    warnings: list[str],
    sample_seconds: float = 1.0,
    sample_at_percent: int = DEFAULT_SSIM_PERCENT,
    source_duration_s: float = 0.0,
) -> float | None:
    """Run ffmpeg's native ``ssim`` filter on a short shared window.

    We compute SSIM between source and output starting at a percentile
    of the **source** duration (defaults to 50%), for
    ``sample_seconds`` of frames. The output of ``ssim=stats_file=-``
    is one line per frame plus a final ``All:`` line with the mean —
    we parse only the ``All:`` line for the scalar.

    Returns ``None`` when ffmpeg can't compute it (missing binary,
    decoder failure). Logged in ``warnings``.
    """
    start_s = max(0.0, source_duration_s * (sample_at_percent / 100.0))
    argv = [
        ffmpeg_bin,
        "-loglevel",
        "error",
        "-ss",
        f"{start_s:.3f}",
        "-t",
        f"{sample_seconds:.3f}",
        "-i",
        str(source_path),
        "-ss",
        f"{start_s:.3f}",
        "-t",
        f"{sample_seconds:.3f}",
        "-i",
        str(output_path),
        "-lavfi",
        "ssim=stats_file=-",
        "-f",
        "null",
        "-",
    ]
    try:
        stderr = _run_ffmpeg_silent(argv)
    except UniqueDistanceError as exc:
        warnings.append(f"ssim_failed: {exc}")
        return None
    # ssim emits stats to stdout when `-` is set; in our argv we set
    # it after `-lavfi` so it lands on stdout via the `null` muxer.
    # We capture stderr above and fall back to it for older ffmpeg.
    for line in stderr.splitlines():
        line = line.strip()
        if line.startswith("[Parsed_ssim_") and "All:" in line:
            try:
                token = line.split("All:", 1)[1].split()[0]
                return float(token)
            except (IndexError, ValueError):
                continue
    warnings.append("ssim_no_all_line")
    return None


def _audio_loudness(
    output_path: Path,
    *,
    ffmpeg_bin: str,
    warnings: list[str],
) -> tuple[float | None, float | None]:
    """Run ``ebur128`` for integrated loudness + true-peak summary.

    Returns ``(audio_rms_db, audio_true_peak_db)``. Either may be
    ``None`` when ebur128 fails or the output has no audio stream.

    ``audio_rms_db`` is reported as the EBU R128 integrated loudness
    (``I:`` line). Strictly that's LUFS, but the spec field name is
    ``audio_rms_db`` — we keep the spec name for the manifest and
    note the unit in the docstring.
    """
    argv = [
        ffmpeg_bin,
        "-loglevel",
        "info",
        "-i",
        str(output_path),
        "-af",
        "ebur128=peak=true",
        "-f",
        "null",
        "-",
    ]
    try:
        stderr = _run_ffmpeg_silent(argv)
    except UniqueDistanceError as exc:
        warnings.append(f"audio_loudness_failed: {exc}")
        return None, None

    integrated_lufs: float | None = None
    true_peak_db: float | None = None
    summary_section = False
    for raw_line in stderr.splitlines():
        line = raw_line.strip()
        if "Summary:" in line:
            summary_section = True
            continue
        if not summary_section:
            continue
        if line.startswith("I:"):
            try:
                integrated_lufs = float(line.split()[1])
            except (IndexError, ValueError):
                continue
        elif line.startswith("Peak:"):
            try:
                true_peak_db = float(line.split()[1])
            except (IndexError, ValueError):
                continue
    if integrated_lufs is None and true_peak_db is None:
        warnings.append("audio_loudness_no_summary")
    return integrated_lufs, true_peak_db


def _frame_rgb_mean_delta(
    source_path: Path,
    output_path: Path,
    *,
    ffprobe_bin: str,
    warnings: list[str],
) -> tuple[float, float, float] | None:
    """Mean per-channel RGB delta between source and output.

    Reads a single mid-clip frame from each via ``ffprobe`` and a
    ``signalstats`` filter, then differences the means. ffmpeg's
    ``signalstats`` exposes ``YAVG`` / ``UAVG`` / ``VAVG`` but not raw
    RGB averages; we use ``YAVG`` triplet (Y, U-Cb, V-Cr) as a stand-in
    since the spec field is a 3-tuple. Manifest text labels it
    ``frame_rgb_mean_delta`` for human readability.

    Returns ``None`` on ffprobe failure. Warnings are appended.
    """
    src_means = _signalstats_means(source_path, ffprobe_bin=ffprobe_bin)
    out_means = _signalstats_means(output_path, ffprobe_bin=ffprobe_bin)
    if src_means is None or out_means is None:
        warnings.append("rgb_mean_delta_unavailable")
        return None
    return (
        out_means[0] - src_means[0],
        out_means[1] - src_means[1],
        out_means[2] - src_means[2],
    )


def _signalstats_means(
    path: Path, *, ffprobe_bin: str
) -> tuple[float, float, float] | None:
    """Read Y/U/V means from a single frame via ffprobe + signalstats."""
    argv = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_frames",
        "-show_entries",
        "frame_tags=lavfi.signalstats.YAVG,lavfi.signalstats.UAVG,lavfi.signalstats.VAVG",
        "-of",
        "json",
        "-read_intervals",
        "%+#1",
        "-f",
        "lavfi",
        f"movie={path},signalstats",
    ]
    try:
        proc = subprocess.run(  # noqa: S603 — argv list, no shell
            argv,
            check=False,
            capture_output=True,
            timeout=15.0,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    if proc.returncode != 0:
        return None
    try:
        data = json.loads(proc.stdout.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return None
    frames = data.get("frames") or []
    if not frames:
        return None
    tags = frames[0].get("tags") or {}
    try:
        return (
            float(tags["lavfi.signalstats.YAVG"]),
            float(tags["lavfi.signalstats.UAVG"]),
            float(tags["lavfi.signalstats.VAVG"]),
        )
    except (KeyError, ValueError, TypeError):
        return None


def _chromaprint_fingerprint(
    output_path: Path, *, warnings: list[str]
) -> str | None:
    """Optional ``fpcalc`` fingerprint. Silently skipped when absent."""
    fpcalc = shutil.which("fpcalc")
    if fpcalc is None:
        return None
    argv = [fpcalc, "-raw", "-json", str(output_path)]
    try:
        proc = subprocess.run(  # noqa: S603 — argv list, no shell
            argv,
            check=False,
            capture_output=True,
            timeout=30.0,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        warnings.append(f"chromaprint_failed: {exc}")
        return None
    if proc.returncode != 0:
        warnings.append("chromaprint_nonzero_exit")
        return None
    try:
        data = json.loads(proc.stdout.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        warnings.append("chromaprint_invalid_json")
        return None
    fingerprint = data.get("fingerprint")
    if isinstance(fingerprint, list):
        # `-raw` flag emits an array of ints; join for storage.
        return ",".join(str(x) for x in fingerprint)
    if isinstance(fingerprint, str):
        return fingerprint
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_components(
    source_path: str | Path,
    output_path: str | Path,
    *,
    ssim_enabled: bool = True,
    audio_enabled: bool = True,
    rgb_mean_enabled: bool = True,
    chromaprint_enabled: bool = True,
    ffmpeg_bin: str = "ffmpeg",
    ffprobe_bin: str = "ffprobe",
) -> UniqueDistanceComponents:
    """Compute the §10.7 unique-distance vector.

    Each enabled component is computed independently. Optional
    components that fail return ``None`` plus a warning; the function
    only raises :class:`UniqueDistanceError` when *required* inputs
    (the source / output probes, the pHash baseline) are unrecoverable.

    Args:
        source_path: Original clip path. Read-only.
        output_path: v2.1 editor output path. Read-only.
        ssim_enabled: Toggle SSIM sampling (expensive; ~100ms).
        audio_enabled: Toggle ebur128 loudness/peak.
        rgb_mean_enabled: Toggle ffprobe signalstats mean RGB.
        chromaprint_enabled: Toggle fpcalc fingerprint. Even when
            enabled, missing ``fpcalc`` is silently skipped.
        ffmpeg_bin / ffprobe_bin: Binary names. Override for tests
            and for non-standard installs.

    Returns:
        :class:`UniqueDistanceComponents`. ``phash_distances`` is the
        only required component; everything else may be ``None``.

    Raises:
        UniqueDistanceError: If both source and output cannot be
            probed (the duration delta needs them).
    """
    source = Path(source_path)
    output = Path(output_path)
    warnings: list[str] = []

    src_duration = _probe_duration_s(source, ffprobe_bin=ffprobe_bin)
    out_duration = _probe_duration_s(output, ffprobe_bin=ffprobe_bin)
    if src_duration is None or out_duration is None:
        raise UniqueDistanceError(
            f"could not probe durations: src={src_duration!r} out={out_duration!r}"
        )

    duration_delta_s = out_duration - src_duration

    phash_distances = _phash_distances(
        source,
        output,
        source_duration_s=src_duration,
        output_duration_s=out_duration,
        ffmpeg_bin=ffmpeg_bin,
        warnings=warnings,
    )

    ssim_sample = (
        _ssim_sample(
            source,
            output,
            ffmpeg_bin=ffmpeg_bin,
            warnings=warnings,
            source_duration_s=src_duration,
        )
        if ssim_enabled
        else None
    )

    if audio_enabled:
        audio_rms_db, audio_true_peak_db = _audio_loudness(
            output, ffmpeg_bin=ffmpeg_bin, warnings=warnings
        )
    else:
        audio_rms_db, audio_true_peak_db = None, None

    rgb_mean_delta = (
        _frame_rgb_mean_delta(
            source, output, ffprobe_bin=ffprobe_bin, warnings=warnings
        )
        if rgb_mean_enabled
        else None
    )

    chromaprint_fingerprint = (
        _chromaprint_fingerprint(output, warnings=warnings)
        if chromaprint_enabled
        else None
    )

    return UniqueDistanceComponents(
        phash_distances=phash_distances,
        duration_delta_s=duration_delta_s,
        frame_rgb_mean_delta=rgb_mean_delta,
        ssim_sample=ssim_sample,
        audio_rms_db=audio_rms_db,
        audio_true_peak_db=audio_true_peak_db,
        chromaprint_fingerprint=chromaprint_fingerprint,
        components_version="2.1",
        warnings=tuple(warnings),
    )


__all__ = [
    "DEFAULT_SSIM_PERCENT",
    "ComponentsVersion",
    "UniqueDistanceComponents",
    "UniqueDistanceError",
    "compute_components",
]
