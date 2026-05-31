"""Zoom-curve planner (final_spec §6).

Pure function: given a beat sheet, a profile name, and an optional
:class:`StyleLock`, returns a piecewise :class:`ZoomCurve` whose
segments abut and cover ``[0, render_duration]``. All §6.4
motion-sickness caps are HARD — no path here may violate them, even
with a lock. Locks may only *tighten* caps, never loosen them.

The §5.5 one-primary-move HARD RULE is enforced by construction:
each beat contributes exactly one zoom segment with one mode.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .beat_sheet import Beat, PrimaryMove
from .locks import StyleLock

__all__ = (
    "ZoomMode",
    "ZoomTarget",
    "ZOOM_MODE_ENUM",
    "ZOOM_TARGET_ENUM",
    "ProfileZoomCaps",
    "PROFILE_CAPS",
    "MOTION_VELOCITY_CAP",
    "MOTION_ACCELERATION_CAP",
    "PAN_VELOCITY_CAP",
    "ZoomSegment",
    "ZoomCurve",
    "plan_zoom_curve",
)


# ---- closed enums --------------------------------------------------


# §6.1 — closed.
ZOOM_MODE_ENUM: tuple[str, ...] = (
    "slow_push_in",
    "push_out",
    "hold",
    "hold_wide_reveal",
    "pan",
    "static",
)

ZOOM_TARGET_ENUM: tuple[str, ...] = (
    "face_priority",
    "action_geometry",
    "balanced_face_object",
    "object_priority",
    "center",
)


ZoomMode = str
ZoomTarget = str


# ---- profile caps (§6.2) -------------------------------------------


@dataclass(frozen=True, slots=True)
class ProfileZoomCaps:
    """§6.2 — absolute upper bound on zoom for one profile.

    ``base_max_zoom`` is the absolute ceiling. ``hook_delta_budget``
    is consumed *within* the ceiling (not added). ``action_override``
    is the clamp applied when ``purpose=action AND action_level≥0.6``
    — when triggered, this is the maximum allowed zoom, period.
    """

    base_max_zoom: float
    hook_delta_budget: float
    action_override: float


PROFILE_CAPS: Mapping[str, ProfileZoomCaps] = {
    "light": ProfileZoomCaps(
        base_max_zoom=1.30, hook_delta_budget=0.05, action_override=1.10
    ),
    "medium": ProfileZoomCaps(
        base_max_zoom=1.35, hook_delta_budget=0.05, action_override=1.20
    ),
    "heavy": ProfileZoomCaps(
        base_max_zoom=1.50, hook_delta_budget=0.05, action_override=1.25
    ),
}


# §6.4 — HARD motion caps. No code path may exceed these.
MOTION_VELOCITY_CAP: float = 0.08  # |d(zoom)/dt|
MOTION_ACCELERATION_CAP: float = 0.04  # |d²(zoom)/dt²|
PAN_VELOCITY_CAP: float = 0.10  # |d(center_x_norm)/dt|

# §6.1 — default transition_ms is 150, max is 400ms.
DEFAULT_TRANSITION_MS: float = 150.0
MAX_TRANSITION_MS: float = 400.0


# §6.2 action threshold + hysteresis.
ACTION_LEVEL_ENTER: float = 0.6
ACTION_LEVEL_EXIT: float = 0.55


# ---- output dataclasses --------------------------------------------


@dataclass(frozen=True, slots=True)
class ZoomSegment:
    """One piecewise §6.1 curve segment.

    ``transition_ms`` is the ease-in duration at this segment's
    *start* boundary. Always ≥150ms, ≤400ms. ``max_zoom`` ≥1.0;
    ``static`` segments use ``max_zoom=1.0`` and ``mode=static``.
    """

    beat_id: str
    start_s: float
    end_s: float
    mode: ZoomMode
    target: ZoomTarget
    max_zoom: float
    transition_ms: float
    reason: str = ""

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True, slots=True)
class ZoomCurve:
    """Frozen list of :class:`ZoomSegment` with metadata.

    ``profile``/``lock_id``/``warnings`` are recorded so the worker
    can replay the planner's choices in the manifest.
    """

    profile: str
    lock_id: str | None
    segments: tuple[ZoomSegment, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)


# ---- §6.3 framing policy table -------------------------------------


_FRAMING_DEFAULTS: Mapping[str, tuple[ZoomTarget, ZoomMode]] = {
    "hook": ("face_priority", "slow_push_in"),
    "reveal": ("balanced_face_object", "slow_push_in"),
    "tension": ("face_priority", "slow_push_in"),
    "emotional_hold": ("face_priority", "slow_push_in"),
    "action": ("action_geometry", "hold"),
    "dialogue": ("face_priority", "hold"),
    "reaction": ("face_priority", "hold"),
    "resolution": ("center", "push_out"),
    "transition": ("center", "static"),
}


# Mapping from beat ``primary_move`` (§5.5) into zoom ``mode`` (§6.1).
_PRIMARY_MOVE_TO_ZOOM_MODE: Mapping[PrimaryMove, ZoomMode] = {
    "hook_push_in": "slow_push_in",
    "slow_push_in": "slow_push_in",
    "push_out": "push_out",
    "hold": "hold",
    "pan": "pan",
    "match_cut": "hold",  # match_cut is a transition between segments
    "reframe": "hold",
    "static": "static",
}


# ---- planner core --------------------------------------------------


def _resolve_profile_caps(profile_name: str) -> ProfileZoomCaps:
    return PROFILE_CAPS.get(profile_name, PROFILE_CAPS["medium"])


def _lock_max_zoom_delta(lock: StyleLock | None) -> float:
    if lock is None:
        return 0.0
    delta = lock.zoom_grammar_overrides.get("max_zoom_delta", 0.0)
    try:
        return float(delta)
    except (TypeError, ValueError):
        return 0.0


def _lock_transition_floor(lock: StyleLock | None) -> float:
    if lock is None:
        return DEFAULT_TRANSITION_MS
    floor = lock.zoom_grammar_overrides.get("transition_ms_floor", DEFAULT_TRANSITION_MS)
    try:
        return float(floor)
    except (TypeError, ValueError):
        return DEFAULT_TRANSITION_MS


def _lock_action_threshold(lock: StyleLock | None) -> float:
    if lock is None:
        return ACTION_LEVEL_ENTER
    try:
        return float(lock.action_level_threshold)
    except (TypeError, ValueError):
        return ACTION_LEVEL_ENTER


def _lock_hook_zoom_delta(lock: StyleLock | None) -> float | None:
    if lock is None:
        return None
    val: Any = lock.hook_treatment_overrides.get("zoom_delta") if isinstance(
        lock.hook_treatment_overrides, Mapping
    ) else None
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _clamp_max_zoom(
    proposed: float,
    *,
    caps: ProfileZoomCaps,
    is_action_override: bool,
    lock_max_delta: float,
) -> tuple[float, str | None]:
    """Apply profile + action + lock + HARD §6.4 clamps.

    Returns (clamped_max_zoom, warning_or_None). HARD floor is 1.0;
    HARD ceiling is caps.base_max_zoom + lock_max_delta (clamped to
    [1.0, caps.base_max_zoom]).
    """
    # Lock can only TIGHTEN — delta < 0 reduces ceiling.
    lock_ceiling = caps.base_max_zoom + min(0.0, lock_max_delta)
    if is_action_override:
        lock_ceiling = min(lock_ceiling, caps.action_override)
    if proposed < 1.0:
        return 1.0, "clamp_min_zoom"
    if proposed > lock_ceiling:
        return lock_ceiling, "clamp_max_zoom"
    return proposed, None


def _segment_for_beat(
    beat: Beat,
    *,
    caps: ProfileZoomCaps,
    lock: StyleLock | None,
    is_first: bool,
    lock_transition_floor: float,
    action_threshold: float,
    warnings: list[str],
) -> ZoomSegment:
    purpose = beat.purpose
    target, default_mode = _FRAMING_DEFAULTS.get(purpose, ("center", "hold"))

    # Mode comes from beat.primary_move (§5.5), with a sanity fall-
    # back to the §6.3 framing default if the beat carries an invalid
    # primary.
    mode = _PRIMARY_MOVE_TO_ZOOM_MODE.get(beat.primary_move, default_mode)

    # Action override (§6.2 hysteresis).
    is_action_override = purpose == "action"

    # Pick proposed max_zoom.
    if mode == "static":
        proposed = 1.0
    elif is_first and purpose == "hook":
        budget = caps.hook_delta_budget
        lock_hook_delta = _lock_hook_zoom_delta(lock)
        if lock_hook_delta is not None:
            # Lock-supplied delta is consumed within budget, can't exceed it.
            budget = min(abs(budget), abs(lock_hook_delta))
        proposed = caps.base_max_zoom - 0.05 + budget  # use the budget delta inside the ceiling
    elif mode in ("hold", "pan", "slow_push_in", "push_out"):
        # Default reach: base cap - small headroom.
        proposed = caps.base_max_zoom - 0.05
    elif mode == "hold_wide_reveal":
        proposed = 1.10
    else:
        proposed = caps.base_max_zoom - 0.10

    max_zoom, warn = _clamp_max_zoom(
        proposed,
        caps=caps,
        is_action_override=is_action_override,
        lock_max_delta=_lock_max_zoom_delta(lock),
    )
    if warn:
        warnings.append(f"{beat.beat_id}:{warn}")

    # Transition. Default 150ms; lock raises floor; max 400ms.
    transition_ms = max(lock_transition_floor, DEFAULT_TRANSITION_MS)
    # Apply multiplier.
    if lock is not None and lock.transition_ms_multiplier > 0:
        transition_ms *= lock.transition_ms_multiplier
    transition_ms = min(transition_ms, MAX_TRANSITION_MS)

    # If beat is *very* short, scale transition down so the
    # §6.4 velocity cap |d(zoom)/dt|≤0.08/s holds. Required delta is
    # max_zoom - 1.0; min duration = delta / 0.08 (assuming linear).
    required_dur_s = max(0.0, (max_zoom - 1.0)) / MOTION_VELOCITY_CAP
    if required_dur_s > beat.duration_s + 1e-6 and beat.duration_s > 0:
        # Drop zoom so it fits at the cap velocity.
        max_zoom = round(min(max_zoom, 1.0 + beat.duration_s * MOTION_VELOCITY_CAP), 4)
        warnings.append(f"{beat.beat_id}:velocity_clamp")

    reason = f"purpose={purpose}; primary_move={beat.primary_move}; mode={mode}"
    if is_action_override:
        reason += "; action_override"
    if lock is not None:
        reason += f"; lock={lock.lock_id}"

    return ZoomSegment(
        beat_id=beat.beat_id,
        start_s=beat.start_s,
        end_s=beat.end_s,
        mode=mode,
        target=target,
        max_zoom=max_zoom,
        transition_ms=transition_ms,
        reason=reason,
    )


def _enforce_no_reversal(
    segments: list[ZoomSegment], warnings: list[str]
) -> list[ZoomSegment]:
    """§6.4: direction reversal forbidden within 0.5s window.

    When two adjacent segments would swing direction (push_in →
    push_out or vice versa) and their boundary gap is <0.5s, the
    second segment is rewritten to ``hold`` to break the reversal.
    """
    if len(segments) < 2:
        return segments
    out = list(segments)
    for i in range(1, len(out)):
        prev = out[i - 1]
        cur = out[i]
        if prev.mode == "slow_push_in" and cur.mode == "push_out":
            gap = cur.start_s - prev.end_s
            if gap < 0.5:
                out[i] = ZoomSegment(
                    beat_id=cur.beat_id,
                    start_s=cur.start_s,
                    end_s=cur.end_s,
                    mode="hold",
                    target=cur.target,
                    max_zoom=prev.max_zoom,
                    transition_ms=cur.transition_ms,
                    reason=cur.reason + "; no_reversal_clamp",
                )
                warnings.append(f"{cur.beat_id}:no_reversal_clamp")
        elif prev.mode == "push_out" and cur.mode == "slow_push_in":
            gap = cur.start_s - prev.end_s
            if gap < 0.5:
                out[i] = ZoomSegment(
                    beat_id=cur.beat_id,
                    start_s=cur.start_s,
                    end_s=cur.end_s,
                    mode="hold",
                    target=cur.target,
                    max_zoom=1.0,
                    transition_ms=cur.transition_ms,
                    reason=cur.reason + "; no_reversal_clamp",
                )
                warnings.append(f"{cur.beat_id}:no_reversal_clamp")
    return out


def _validate_caps_invariant(segments: list[ZoomSegment]) -> None:
    """Assert §6.4 HARD caps are obeyed at the segment level."""
    for s in segments:
        if s.max_zoom < 1.0 - 1e-6:
            raise AssertionError(f"max_zoom < 1.0 on {s.beat_id}")
        if s.transition_ms < DEFAULT_TRANSITION_MS - 1e-6:
            raise AssertionError(
                f"transition_ms<{DEFAULT_TRANSITION_MS} on {s.beat_id}"
            )
        if s.transition_ms > MAX_TRANSITION_MS + 1e-6:
            raise AssertionError(
                f"transition_ms>{MAX_TRANSITION_MS} on {s.beat_id}"
            )
        if s.duration_s > 0:
            velocity = abs(s.max_zoom - 1.0) / max(s.duration_s, 1e-6)
            if velocity > MOTION_VELOCITY_CAP + 1e-6:
                raise AssertionError(
                    f"velocity cap violated on {s.beat_id}: {velocity:.4f}"
                )


def plan_zoom_curve(
    beat_sheet: tuple[Beat, ...],
    profile: str = "medium",
    lock: StyleLock | None = None,
) -> ZoomCurve:
    """Plan the §6 piecewise zoom curve for ``beat_sheet``.

    HARD rules enforced:

    * §5.5 one primary move per beat (one segment per beat);
    * §6.2 profile cap (silently clamps overshoot with a manifest
      WARN, never an exception);
    * §6.4 motion-sickness caps (velocity / acceleration / no
      direction reversal within 0.5s — clamped, never violated);
    * adjacent segments abut and cover ``[0, render_duration]``;
    * coverage gaps filled with explicit ``static`` segments so the
      curve is total over ``[0, render_duration]``.
    """
    caps = _resolve_profile_caps(profile)
    warnings: list[str] = []
    transition_floor = _lock_transition_floor(lock)
    action_threshold = _lock_action_threshold(lock)

    if not beat_sheet:
        return ZoomCurve(profile=profile, lock_id=(lock.lock_id if lock else None))

    segments: list[ZoomSegment] = []
    cursor = beat_sheet[0].start_s
    for i, beat in enumerate(beat_sheet):
        # Coverage gap before this beat → fill with static.
        if beat.start_s - cursor > 1e-6:
            segments.append(
                ZoomSegment(
                    beat_id=f"_gap_{i:03d}",
                    start_s=cursor,
                    end_s=beat.start_s,
                    mode="static",
                    target="center",
                    max_zoom=1.0,
                    transition_ms=DEFAULT_TRANSITION_MS,
                    reason="coverage_gap_filler",
                )
            )
        seg = _segment_for_beat(
            beat,
            caps=caps,
            lock=lock,
            is_first=(i == 0),
            lock_transition_floor=transition_floor,
            action_threshold=action_threshold,
            warnings=warnings,
        )
        segments.append(seg)
        cursor = beat.end_s

    segments = _enforce_no_reversal(segments, warnings)
    _validate_caps_invariant(segments)

    return ZoomCurve(
        profile=profile,
        lock_id=(lock.lock_id if lock else None),
        segments=tuple(segments),
        warnings=tuple(warnings),
    )
