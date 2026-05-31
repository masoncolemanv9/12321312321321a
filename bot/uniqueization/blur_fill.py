"""v2.1 blur-fill composer (``final_spec_FULL.md`` §8.11).

For sources that aren't already 9:16 portrait, the v2.1 editor renders
the main subject scaled into a portrait target and uses a blurred
oversized copy of the same frame as a backdrop. This module produces
the **ffmpeg filter_complex segment** that performs the composition;
it does not invoke ffmpeg itself (Part 7's worker integration handles
that).

Pure-function module. Inputs are source/target resolutions and a mode
enum; the output is a :class:`BlurFillFilter` carrying the filter
string, the input/output stream labels, and a closed-enum reason when
the filter is intentionally a no-op.

Mode enum (§8.11):

* ``off``           — no backdrop (native 9:16 source).
* ``light_blur``    — boxblur=20:cr=20 backdrop (medium profile default).
* ``heavy_blur_dark`` — same blur + ``eq=brightness=-0.20`` darkening
  (heavy profile, ``emotional_hold`` beats).

Skip rules (per §8.11):

* Native portrait 9:16 source (aspect ratio within 1% of 9:16) → no
  blur backdrop, skip with reason ``native_portrait``.
* Panoramic source wider than 16:9 (e.g. 21:9) → skip with reason
  ``aspect_too_wide``.
* Vertical source taller than 9:16 → skip with reason
  ``aspect_too_tall``.

The skip reasons are stable identifiers; Part 7 writes them to the
manifest's ``stages.blur_fill.skip_reason`` field.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

#: Closed mode enum mirrored in the manifest.
BlurFillMode = Literal["off", "light_blur", "heavy_blur_dark"]

#: Closed enum of skip reasons. ``None`` is reserved for "no skip" so
#: a non-empty filter string is paired with ``skip_reason=None``.
BlurFillSkipReason = Literal[
    "native_portrait",
    "aspect_too_wide",
    "aspect_too_tall",
    "mode_off",
]

#: Source aspect ratio bounds in which blur-fill is supported. §8.11
#: caps to ``4:3..16:9`` landscape; we mirror that with a small
#: tolerance to absorb integer-pixel rounding.
MIN_SUPPORTED_ASPECT: float = 4.0 / 3.0 / 1.05  # ~1.27 (slightly under 4:3)
MAX_SUPPORTED_ASPECT: float = 16.0 / 9.0 * 1.02  # ~1.81 (just past 16:9)
NATIVE_PORTRAIT_ASPECT: float = 9.0 / 16.0  # 0.5625
PORTRAIT_TOLERANCE: float = 0.01  # ±1% counts as native portrait


@dataclass(frozen=True, slots=True)
class BlurFillFilter:
    """Output of :func:`build_blur_fill_filter`.

    ``filter_complex`` is the segment to splice into the worker's
    filtergraph (Part 7). When ``skip_reason`` is not ``None``, the
    string is empty and Part 7 should route the input straight to the
    next stage (Part 2's filtergraph builder already handles the
    no-blur path).

    ``input_label`` and ``output_label`` name the streams the filter
    consumes / emits — Part 7 splices them in. Default labels mirror
    the §8.11.1 example.
    """

    filter_complex: str
    mode: BlurFillMode
    input_label: str
    output_label: str
    skip_reason: BlurFillSkipReason | None
    source_aspect: float
    target_aspect: float


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _aspect(resolution: tuple[int, int]) -> float:
    w, h = resolution
    if w <= 0 or h <= 0:
        raise ValueError(f"resolution must be positive, got {resolution}")
    return w / h


def _classify_source(
    aspect: float, *, native_portrait: float = NATIVE_PORTRAIT_ASPECT
) -> BlurFillSkipReason | None:
    """Return a skip reason if the source aspect isn't blur-fill-eligible."""
    # Native portrait → no backdrop needed.
    if abs(aspect - native_portrait) <= PORTRAIT_TOLERANCE:
        return "native_portrait"
    # Vertical taller than 9:16 (e.g. 9:18) — same family as native
    # portrait, just narrower. The §8.11 wording covers this implicitly
    # by anchoring on the 9:16 target.
    if aspect < native_portrait - PORTRAIT_TOLERANCE:
        return "aspect_too_tall"
    # Landscape wider than 16:9 — panoramic.
    if aspect > MAX_SUPPORTED_ASPECT:
        return "aspect_too_wide"
    return None


def build_blur_fill_filter(
    source_resolution: tuple[int, int],
    target_resolution: tuple[int, int] = (1080, 1920),
    *,
    mode: BlurFillMode = "light_blur",
    input_label: str = "0:v",
    output_label: str = "composite",
    blur_radius: int = 20,
    chroma_radius: int = 20,
    dark_eq_brightness: float = -0.20,
) -> BlurFillFilter:
    """Build the filter_complex segment for §8.11.1 blur-fill.

    Args:
        source_resolution: ``(width, height)`` of the source clip in
            pixels.
        target_resolution: Target render resolution. Defaults to
            ``(1080, 1920)`` (portrait 9:16).
        mode: One of the :data:`BlurFillMode` values. ``off`` returns
            an empty filter with ``skip_reason=mode_off`` — Part 7
            routes around it.
        input_label: ffmpeg stream label feeding into this segment
            (default ``0:v``).
        output_label: ffmpeg stream label the composite emits
            (default ``composite``).
        blur_radius / chroma_radius: ``boxblur`` parameters
            (``boxblur=<r>:cr=<cr>``). Defaults match the §8.11.1
            example. Validated as ``>0``.
        dark_eq_brightness: Brightness delta applied in
            ``heavy_blur_dark`` mode. Default ``-0.20`` per spec.

    Returns:
        :class:`BlurFillFilter`. Non-empty ``filter_complex`` when
        the source qualifies; empty string + ``skip_reason`` otherwise.
    """
    if blur_radius <= 0 or chroma_radius <= 0:
        raise ValueError(
            f"blur radii must be > 0, got {blur_radius}/{chroma_radius}"
        )

    source_aspect = _aspect(source_resolution)
    target_aspect = _aspect(target_resolution)

    if mode == "off":
        return BlurFillFilter(
            filter_complex="",
            mode=mode,
            input_label=input_label,
            output_label=output_label,
            skip_reason="mode_off",
            source_aspect=source_aspect,
            target_aspect=target_aspect,
        )

    skip = _classify_source(source_aspect)
    if skip is not None:
        return BlurFillFilter(
            filter_complex="",
            mode=mode,
            input_label=input_label,
            output_label=output_label,
            skip_reason=skip,
            source_aspect=source_aspect,
            target_aspect=target_aspect,
        )

    tw, th = target_resolution
    # §8.11.1 expression. The intermediate label `[bg_dark]` is only
    # produced in heavy mode; in light mode the dark step is skipped
    # and the overlay reads `[bg_blur]` directly.
    parts: list[str] = []
    parts.append(f"[{input_label}]split=2[main][bg]")
    parts.append(
        f"[bg]scale={tw}:{th}:force_original_aspect_ratio=increase,"
        f"boxblur={blur_radius}:cr={chroma_radius}[bg_blur]"
    )
    overlay_src = "bg_blur"
    if mode == "heavy_blur_dark":
        parts.append(
            f"[bg_blur]eq=brightness={dark_eq_brightness:g}[bg_dark]"
        )
        overlay_src = "bg_dark"
    parts.append(f"[main]scale={tw}:-1[main_scaled]")
    parts.append(
        f"[{overlay_src}][main_scaled]overlay=(W-w)/2:(H-h)/2"
        f"[{output_label}]"
    )

    return BlurFillFilter(
        filter_complex=";".join(parts),
        mode=mode,
        input_label=input_label,
        output_label=output_label,
        skip_reason=None,
        source_aspect=source_aspect,
        target_aspect=target_aspect,
    )


def is_landscape_or_square(
    source_resolution: tuple[int, int],
) -> bool:
    """Helper mirroring §8.11.1's ``is_landscape_or_square`` trigger.

    Returns ``True`` for sources where the v2.1 pipeline would invoke
    blur-fill; ``False`` for native portrait. Panoramic sources
    return ``True`` even though :func:`build_blur_fill_filter` will
    skip them — the caller still needs to know to try, so it can log
    the ``aspect_too_wide`` skip reason.
    """
    return _aspect(source_resolution) >= NATIVE_PORTRAIT_ASPECT + PORTRAIT_TOLERANCE


__all__ = [
    "MAX_SUPPORTED_ASPECT",
    "MIN_SUPPORTED_ASPECT",
    "NATIVE_PORTRAIT_ASPECT",
    "PORTRAIT_TOLERANCE",
    "BlurFillFilter",
    "BlurFillMode",
    "BlurFillSkipReason",
    "build_blur_fill_filter",
    "is_landscape_or_square",
]
