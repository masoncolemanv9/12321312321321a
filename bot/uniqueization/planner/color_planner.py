"""Color-style planner (final_spec §8.10).

Per-beat ``color_intent`` enum selection + adjacent-beat cross-fade
durations. Output is *parametric*: brightness / contrast / saturation
/ hue deltas relative to the v2.0 base. **No bundled LUTs** (§0
red-flag #7); LUT path is referenced via env / per-job override at
the worker layer, never embedded here.

Pure function. Deterministic.

Closed enum:

* ``neutral``
* ``warm_emotional``
* ``cold_tense``
* ``desaturated_action``
* ``high_contrast_punch``

Cross-fade duration: 200 – 400 ms when adjacent beats' intents
differ. The planner emits piecewise-linear interpolation between
adjacent segment params; the worker (Part 10) renders the ramp.

Profile-specific brightness/contrast/saturation/hue deltas are
bounded by §8.10:

* brightness delta ≤ ±10 %
* contrast delta ≤ ±15 %
* saturation delta ≤ ±25 %

Heavy profile keeps the SAME bounds — aggression comes from
intent diversity, not magnitude.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from .beat_sheet import Beat
from .locks import StyleLock

__all__ = (
    "COLOR_INTENT_ENUM",
    "CROSS_FADE_MIN_MS",
    "CROSS_FADE_MAX_MS",
    "CROSS_FADE_DEFAULT_MS",
    "PROFILE_BASE_COLORGRADE",
    "INTENT_DELTAS",
    "BRIGHTNESS_DELTA_CAP",
    "CONTRAST_DELTA_CAP",
    "SATURATION_DELTA_CAP",
    "ColorIntent",
    "ColorSegment",
    "ColorTransition",
    "ColorPlan",
    "plan_color",
)


COLOR_INTENT_ENUM: tuple[str, ...] = (
    "neutral",
    "warm_emotional",
    "cold_tense",
    "desaturated_action",
    "high_contrast_punch",
)


# §8.10 cross-fade bounds.
CROSS_FADE_MIN_MS: float = 200.0
CROSS_FADE_MAX_MS: float = 400.0
CROSS_FADE_DEFAULT_MS: float = 300.0


# §8.10 v2.0 base colorgrade table.
@dataclass(frozen=True, slots=True)
class ColorGradeParams:
    brightness: float
    contrast: float
    saturation: float
    hue_deg: float


PROFILE_BASE_COLORGRADE: Mapping[str, ColorGradeParams] = {
    "light": ColorGradeParams(brightness=0.01, contrast=1.03, saturation=1.04, hue_deg=1.0),
    "medium": ColorGradeParams(brightness=0.02, contrast=1.06, saturation=1.08, hue_deg=3.0),
    "heavy": ColorGradeParams(brightness=0.03, contrast=1.10, saturation=1.12, hue_deg=5.0),
}


# §8.10 caps (percent relative to base brightness/contrast/saturation).
BRIGHTNESS_DELTA_CAP: float = 0.10
CONTRAST_DELTA_CAP: float = 0.15
SATURATION_DELTA_CAP: float = 0.25
HUE_DELTA_CAP_DEG: float = 8.0


# §8.10 intent deltas. Picked to stay within caps when applied on top
# of any profile base.
INTENT_DELTAS: Mapping[str, Mapping[str, float]] = {
    "neutral": {
        "brightness_delta": 0.0,
        "contrast_delta": 1.0,
        "saturation_delta": 1.0,
        "hue_delta_deg": 0.0,
    },
    "warm_emotional": {
        "brightness_delta": 0.04,
        "contrast_delta": 1.02,
        "saturation_delta": 1.06,
        "hue_delta_deg": 4.0,
    },
    "cold_tense": {
        "brightness_delta": -0.04,
        "contrast_delta": 1.05,
        "saturation_delta": 0.92,
        "hue_delta_deg": -5.0,
    },
    "desaturated_action": {
        "brightness_delta": 0.0,
        "contrast_delta": 1.08,
        "saturation_delta": 0.85,
        "hue_delta_deg": 0.0,
    },
    "high_contrast_punch": {
        "brightness_delta": 0.0,
        "contrast_delta": 1.12,
        "saturation_delta": 1.05,
        "hue_delta_deg": 0.0,
    },
}


# Purpose × intent mapping. Lock overrides are §12.6 territory.
_BASE_INTENT_BY_PURPOSE: Mapping[str, str] = {
    "hook": "high_contrast_punch",
    "reveal": "high_contrast_punch",
    "tension": "cold_tense",
    "emotional_hold": "warm_emotional",
    "action": "desaturated_action",
    "dialogue": "neutral",
    "reaction": "warm_emotional",
    "transition": "neutral",
    "resolution": "warm_emotional",
}


@dataclass(frozen=True, slots=True)
class ColorIntent:
    """A resolved intent with its computed params (post-base)."""

    name: str
    brightness: float
    contrast: float
    saturation: float
    hue_deg: float


@dataclass(frozen=True, slots=True)
class ColorSegment:
    """One §8.10 per-beat color decision (no cross-fade ramp here)."""

    beat_id: str
    start_s: float
    end_s: float
    intent: ColorIntent
    reason: str = ""


@dataclass(frozen=True, slots=True)
class ColorTransition:
    """§8.10 cross-fade between adjacent beats with differing intents."""

    from_beat_id: str
    to_beat_id: str
    start_s: float  # start of fade (== end of source segment − duration/2)
    end_s: float  # end of fade (== start of target segment + duration/2)
    duration_ms: float


@dataclass(frozen=True, slots=True)
class ColorPlan:
    """Container for :func:`plan_color` output."""

    segments: tuple[ColorSegment, ...] = field(default_factory=tuple)
    transitions: tuple[ColorTransition, ...] = field(default_factory=tuple)
    profile: str = "medium"
    lock_id: str | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)


# ---- helpers -------------------------------------------------------


def _clamp(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _compute_intent_params(
    intent_name: str, base: ColorGradeParams
) -> ColorIntent:
    """Apply caps on top of base + delta."""
    deltas = INTENT_DELTAS.get(intent_name, INTENT_DELTAS["neutral"])
    brightness = _clamp(
        base.brightness + float(deltas["brightness_delta"]),
        base.brightness - BRIGHTNESS_DELTA_CAP,
        base.brightness + BRIGHTNESS_DELTA_CAP,
    )
    # contrast/saturation deltas are MULTIPLICATIVE factors.
    contrast = _clamp(
        base.contrast * float(deltas["contrast_delta"]),
        base.contrast * (1.0 - CONTRAST_DELTA_CAP),
        base.contrast * (1.0 + CONTRAST_DELTA_CAP),
    )
    saturation = _clamp(
        base.saturation * float(deltas["saturation_delta"]),
        base.saturation * (1.0 - SATURATION_DELTA_CAP),
        base.saturation * (1.0 + SATURATION_DELTA_CAP),
    )
    hue = _clamp(
        base.hue_deg + float(deltas["hue_delta_deg"]),
        base.hue_deg - HUE_DELTA_CAP_DEG,
        base.hue_deg + HUE_DELTA_CAP_DEG,
    )
    return ColorIntent(
        name=intent_name,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue_deg=hue,
    )


def _resolve_intent_name(
    purpose: str, *, lock: StyleLock | None
) -> str:
    base = _BASE_INTENT_BY_PURPOSE.get(purpose, "neutral")
    if lock is not None and isinstance(
        lock.color_intent_mapping_overrides, Mapping
    ):
        override = lock.color_intent_mapping_overrides.get(purpose)
        if isinstance(override, str) and override in COLOR_INTENT_ENUM:
            return override
    return base


def _resolve_cross_fade_ms(
    cur_dur_s: float, next_dur_s: float
) -> float:
    """Pick a cross-fade duration within §8.10 bounds.

    Shorter when adjacent beats are short (don't eat the beat).
    """
    avail_ms = min(cur_dur_s, next_dur_s) * 1000.0 * 0.5
    if avail_ms <= CROSS_FADE_MIN_MS:
        return CROSS_FADE_MIN_MS
    return min(CROSS_FADE_DEFAULT_MS, avail_ms, CROSS_FADE_MAX_MS)


# ---- planner -------------------------------------------------------


def plan_color(
    beat_sheet: tuple[Beat, ...],
    profile: str = "medium",
    lock: StyleLock | None = None,
) -> ColorPlan:
    """§8.10 per-beat color decisions + adjacent cross-fades."""
    base = PROFILE_BASE_COLORGRADE.get(
        profile, PROFILE_BASE_COLORGRADE["medium"]
    )
    lock_id = lock.lock_id if lock is not None else None
    if not beat_sheet:
        return ColorPlan(profile=profile, lock_id=lock_id)

    warnings: list[str] = []
    segments: list[ColorSegment] = []
    for beat in beat_sheet:
        intent_name = _resolve_intent_name(beat.purpose, lock=lock)
        if intent_name not in COLOR_INTENT_ENUM:
            warnings.append(
                f"{beat.beat_id}:unknown_intent_{intent_name}_fallback_neutral"
            )
            intent_name = "neutral"
        intent = _compute_intent_params(intent_name, base)
        segments.append(
            ColorSegment(
                beat_id=beat.beat_id,
                start_s=beat.start_s,
                end_s=beat.end_s,
                intent=intent,
                reason=f"purpose={beat.purpose}_intent={intent_name}",
            )
        )

    transitions: list[ColorTransition] = []
    for i in range(1, len(segments)):
        prev = segments[i - 1]
        cur = segments[i]
        if prev.intent.name == cur.intent.name:
            continue
        prev_dur = prev.end_s - prev.start_s
        cur_dur = cur.end_s - cur.start_s
        duration_ms = _resolve_cross_fade_ms(prev_dur, cur_dur)
        half_s = duration_ms / 2.0 / 1000.0
        boundary = cur.start_s
        transitions.append(
            ColorTransition(
                from_beat_id=prev.beat_id,
                to_beat_id=cur.beat_id,
                start_s=max(prev.start_s, boundary - half_s),
                end_s=min(cur.end_s, boundary + half_s),
                duration_ms=duration_ms,
            )
        )

    return ColorPlan(
        segments=tuple(segments),
        transitions=tuple(transitions),
        profile=profile,
        lock_id=lock_id,
        warnings=tuple(warnings),
    )
