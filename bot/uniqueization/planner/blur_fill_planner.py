"""Blur-fill planner (final_spec §8.11).

Per-beat ``blur_fill`` enum selection. The actual ffmpeg filter is
applied by ``bot/uniqueization/blur_fill.py`` (Part 6); this module
only decides *whether* and *which* mode each beat should use.

Closed enum (§8.11):

* ``off`` — native 9:16, no blur backdrop.
* ``light_blur`` — standard backdrop blur (v2.1 default).
* ``heavy_blur_dark`` — blur + darken 10-25 % for
  ``emotional_hold`` beats. Only on medium / heavy profile.

Source-aspect handling:

* Native portrait (≈ 9:16, ratio ≤ 0.6) → ``off``.
* Landscape / square (16:9 .. 4:3) → ``light_blur`` default,
  ``heavy_blur_dark`` for emotional_hold on medium / heavy profile.
* Ultra-wide (>= 21:9, ratio ≥ 2.0) → ``off`` with manifest
  warning ``blur_fill_skipped_panoramic`` (§8.11 last paragraph).

Pure function. Deterministic.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from .beat_sheet import Beat
from .locks import StyleLock

__all__ = (
    "BLUR_FILL_ENUM",
    "BLUR_RATIO_PORTRAIT_MAX",
    "BLUR_RATIO_PANORAMIC_MIN",
    "BlurFillSegment",
    "BlurFillPlan",
    "plan_blur_fill",
)


# §8.11 closed enum.
BLUR_FILL_ENUM: tuple[str, ...] = ("off", "light_blur", "heavy_blur_dark")


# Source aspect thresholds.
BLUR_RATIO_PORTRAIT_MAX: float = 0.60  # width/height ratio
BLUR_RATIO_PANORAMIC_MIN: float = 2.0  # 21:9 ≈ 2.33


# §8.11 defaults by purpose.
_BASE_MODE_BY_PURPOSE: Mapping[str, str] = {
    "hook": "light_blur",
    "reveal": "light_blur",
    "tension": "light_blur",
    "emotional_hold": "heavy_blur_dark",
    "action": "light_blur",
    "dialogue": "light_blur",
    "reaction": "light_blur",
    "transition": "light_blur",
    "resolution": "light_blur",
}


@dataclass(frozen=True, slots=True)
class BlurFillSegment:
    """One §8.11 per-beat blur-fill decision."""

    beat_id: str
    start_s: float
    end_s: float
    mode: str
    darken_pct: float = 0.0
    reason: str = ""


@dataclass(frozen=True, slots=True)
class BlurFillPlan:
    """Container for :func:`plan_blur_fill` output."""

    segments: tuple[BlurFillSegment, ...] = field(default_factory=tuple)
    source_aspect_ratio: float = 0.5625  # 9:16 default
    profile: str = "medium"
    lock_id: str | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)


def _heavy_darken_pct(profile: str) -> float:
    """§8.11 — darken 10 – 25 % for ``heavy_blur_dark``. Medium / heavy only."""
    if profile == "heavy":
        return 0.25
    if profile == "medium":
        return 0.15
    return 0.0


def _resolve_mode(
    purpose: str,
    *,
    profile: str,
    aspect_ratio: float,
) -> tuple[str, str]:
    """Returns (mode, reason)."""
    if aspect_ratio <= BLUR_RATIO_PORTRAIT_MAX:
        return "off", f"native_portrait_ratio={aspect_ratio:.3f}"
    if aspect_ratio >= BLUR_RATIO_PANORAMIC_MIN:
        return "off", f"panoramic_unsupported_ratio={aspect_ratio:.3f}"
    base = _BASE_MODE_BY_PURPOSE.get(purpose, "light_blur")
    if base == "heavy_blur_dark" and profile == "light":
        # Light profile can never use heavy_blur_dark (§8.11).
        return "light_blur", "light_profile_demotes_heavy_to_light"
    return base, f"purpose={purpose};profile={profile}"


def plan_blur_fill(
    beat_sheet: tuple[Beat, ...],
    source_aspect_ratio: float = 0.5625,
    profile: str = "medium",
    lock: StyleLock | None = None,
) -> BlurFillPlan:
    """§8.11 blur-fill mode selection per beat."""
    lock_id = lock.lock_id if lock is not None else None
    warnings: list[str] = []

    if source_aspect_ratio <= 0:
        warnings.append(f"invalid_aspect_ratio_{source_aspect_ratio}")
        source_aspect_ratio = 0.5625

    if source_aspect_ratio >= BLUR_RATIO_PANORAMIC_MIN:
        warnings.append("blur_fill_skipped_panoramic")

    if not beat_sheet:
        return BlurFillPlan(
            source_aspect_ratio=source_aspect_ratio,
            profile=profile,
            lock_id=lock_id,
            warnings=tuple(warnings),
        )

    segments: list[BlurFillSegment] = []
    for beat in beat_sheet:
        mode, reason = _resolve_mode(
            beat.purpose,
            profile=profile,
            aspect_ratio=source_aspect_ratio,
        )
        if mode not in BLUR_FILL_ENUM:
            warnings.append(f"{beat.beat_id}:bad_mode_{mode}")
            mode = "light_blur"
        darken = (
            _heavy_darken_pct(profile)
            if mode == "heavy_blur_dark"
            else 0.0
        )
        segments.append(
            BlurFillSegment(
                beat_id=beat.beat_id,
                start_s=beat.start_s,
                end_s=beat.end_s,
                mode=mode,
                darken_pct=darken,
                reason=reason,
            )
        )

    return BlurFillPlan(
        segments=tuple(segments),
        source_aspect_ratio=source_aspect_ratio,
        profile=profile,
        lock_id=lock_id,
        warnings=tuple(warnings),
    )
