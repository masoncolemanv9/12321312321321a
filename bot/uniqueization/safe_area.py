"""Subtitle safe-area validator — Part 5 of the Editor Agent split.

At render-time, every subtitle event must avoid overlapping faces and
the channel logo (``final_spec_FULL.md`` §8.2). Analyzer-shipped face
boxes plus the YuNet detections already in the manifest's
``stages.face_sampling`` (Part 2) are joined into a single bbox set
before this validator decides whether to keep, move, dim, or drop the
event.

Fallback chain (§8.2 step 3):

1. **Preferred Y** (profile: light 0.78, medium 0.80, heavy 0.82).
2. **Alternative Y positions** ``[0.85, 0.72, 0.65]`` in declaration
   order.
3. **Opacity reduction** to 50% at the preferred Y.
4. **Drop event** with manifest reason ``subtitle_dropped``.

Constraints (pure-function module):

* No I/O, no subprocess. Inputs are bboxes + a candidate Y; the
  output is a :class:`SafeAreaDecision` describing what to do.
* All bbox coordinates are **normalized 0..1** to match Part 2's
  :class:`bot.uniqueization.face_yunet.FaceSample.bbox_pct` shape.
* The opacity floor is a hard 0.5; any caller passing
  ``min_opacity_floor`` below that is clamped, never honoured —
  ratio "0.5 then drop" is a spec invariant (§8.2 last bullet).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

#: Closed enum of profile names. Mirrors the names exposed by
#: :mod:`bot.uniqueization.profiles` (light/medium/heavy). Using a
#: string Literal here avoids a runtime import of the full
#: ``UniqProfile`` dataclass for callers that only need the name.
ProfileName = Literal["light", "medium", "heavy"]

#: Closed enum of why a placement was changed/dropped — recorded in
#: the manifest's ``stages.subtitles.events[].placement_fallback_reason``
#: field (§8.2). Stable identifiers for dashboards.
PlacementFallbackReason = Literal[
    "preferred_clear",
    "moved_to_alternative_y",
    "opacity_reduced",
    "subtitle_dropped",
]

#: Per-profile preferred Y (fraction of frame height, top-down).
#: §8.2 wording: "Preferred Y (profile: light 0.78, medium 0.80, heavy 0.82)".
PREFERRED_Y_BY_PROFILE: dict[ProfileName, float] = {
    "light": 0.78,
    "medium": 0.80,
    "heavy": 0.82,
}

#: Alternative Y positions tried in declaration order (§8.2).
ALTERNATIVE_Y_POSITIONS: tuple[float, ...] = (0.85, 0.72, 0.65)

#: Opacity floor for the reduce-opacity step. Below this we drop. The
#: spec wording fixes the value at 50%; tests pin the equality.
OPACITY_FLOOR: float = 0.5

#: Overlap fraction at-or-below which a position is considered
#: "clear" enough to use at full opacity. §8.2 says "if clear"
#: without a precise number; 10% is the lightest tolerance that
#: still lets thin face bounding boxes graze the cue without
#: triggering the alt-Y dance.
OVERLAP_CLEAR_THRESHOLD: float = 0.10

#: Overlap fraction above which the cue must be dropped (§8.2 last
#: bullet: "if overlap >50%").
OVERLAP_DROP_THRESHOLD: float = 0.5

#: Default normalized subtitle box width/height — used when callers
#: don't override. A 0.84 × 0.10 box at the middle of the frame width
#: matches the §9.10 light profile's 4-line caption layout closely
#: enough to validate against. Real callers (Part 7) pass measured
#: extents.
DEFAULT_SUBTITLE_BOX_W: float = 0.84
DEFAULT_SUBTITLE_BOX_H: float = 0.10


@dataclass(frozen=True, slots=True)
class Bbox:
    """Normalized 0..1 bbox in (x, y, w, h) form, top-left origin."""

    x: float
    y: float
    w: float
    h: float

    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.h

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    @classmethod
    def from_tuple(
        cls, t: tuple[float, float, float, float]
    ) -> Bbox:
        return cls(x=t[0], y=t[1], w=t[2], h=t[3])


@dataclass(frozen=True, slots=True)
class SafeAreaDecision:
    """Output of :func:`validate_subtitle_position`."""

    chosen_y: float
    opacity: float
    drop: bool
    reason: PlacementFallbackReason
    candidates_tried: tuple[float, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _intersection_area(a: Bbox, b: Bbox) -> float:
    ix1 = max(a.x, b.x)
    iy1 = max(a.y, b.y)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    return iw * ih


def _subtitle_box_at(
    y_top: float,
    *,
    box_w: float,
    box_h: float,
) -> Bbox:
    """Build the subtitle's render bbox centred on the frame, at ``y_top``.

    ``y_top`` is the top edge of the cue (matches ASS ``\\an2`` bottom-
    align convention if callers pass the **center** Y minus half-height,
    but this module is symmetric in either interpretation — what
    matters is that the same convention is used for both the cue and
    occlusion test).
    """
    x = max(0.0, (1.0 - box_w) / 2.0)
    y = max(0.0, min(1.0 - box_h, y_top))
    return Bbox(x=x, y=y, w=box_w, h=box_h)


def _overlap_fraction(
    box: Bbox, obstacles: Sequence[Bbox]
) -> float:
    """Sum of (obstacle ∩ box) areas, normalized by box area.

    Multi-obstacle overlap can exceed 1.0 in pathological cases when
    obstacles themselves overlap — we clamp to 1.0 since the caller
    only uses the value against fixed thresholds.
    """
    if box.area <= 0:
        return 1.0
    total = 0.0
    for obs in obstacles:
        total += _intersection_area(box, obs)
    return min(1.0, total / box.area)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_subtitle_position(
    face_bboxes: Sequence[Bbox] = (),
    logo_bbox: Bbox | None = None,
    *,
    profile: ProfileName = "medium",
    candidate_y: float | None = None,
    alternative_ys: Sequence[float] = ALTERNATIVE_Y_POSITIONS,
    subtitle_box_w: float = DEFAULT_SUBTITLE_BOX_W,
    subtitle_box_h: float = DEFAULT_SUBTITLE_BOX_H,
    overlap_clear_threshold: float = OVERLAP_CLEAR_THRESHOLD,
    overlap_drop_threshold: float = OVERLAP_DROP_THRESHOLD,
) -> SafeAreaDecision:
    """Decide where (or whether) to place a subtitle cue.

    Resolution order (§8.2):

    1. Try the preferred Y (``candidate_y`` or the profile default).
       If overlap with face/logo bboxes is at or below
       ``overlap_clear_threshold``, return ``preferred_clear``.
    2. Try each ``alternative_ys`` in order. First Y whose overlap
       is at or below the clear threshold wins with reason
       ``moved_to_alternative_y``.
    3. Reduce opacity to :data:`OPACITY_FLOOR` at the preferred Y.
       Used only when the preferred-Y overlap is still at or below
       ``overlap_drop_threshold`` (i.e. the cue isn't fully occluded,
       just sitting on a busy area). Above the drop threshold we
       drop — dimming a fully-covered cue doesn't help.
    4. Drop with ``subtitle_dropped`` when even the dimming step
       can't salvage the placement.

    Args:
        face_bboxes: Face boxes from analyzer + YuNet, normalized.
        logo_bbox: The logo bbox if rendered.
        profile: Profile for the preferred-Y default.
        candidate_y: Explicit preferred Y overriding the profile
            default. Useful for hook-overlay (Y=0.20) and for tests.
        alternative_ys: Alternate Y list (top-down). Defaults to
            §8.2's ``[0.85, 0.72, 0.65]``.
        subtitle_box_w/h: Normalized subtitle box dimensions used
            for occlusion testing.
        overlap_clear_threshold: Fraction at-or-below which a
            position is "clear" (full-opacity placement allowed).
            Defaults to :data:`OVERLAP_CLEAR_THRESHOLD` (10%).
        overlap_drop_threshold: Fraction above which the cue must be
            dropped. Defaults to :data:`OVERLAP_DROP_THRESHOLD` (50%,
            per §8.2).

    Returns:
        :class:`SafeAreaDecision` carrying ``chosen_y``, ``opacity``,
        ``drop`` flag, ``reason`` from the closed enum, and the list
        of Ys tried (handy for the manifest's debug payload).
    """

    obstacles: list[Bbox] = list(face_bboxes)
    if logo_bbox is not None:
        obstacles.append(logo_bbox)

    if candidate_y is None:
        if profile not in PREFERRED_Y_BY_PROFILE:
            raise ValueError(
                f"unknown profile {profile!r}; expected one of "
                f"{sorted(PREFERRED_Y_BY_PROFILE)}"
            )
        candidate_y = PREFERRED_Y_BY_PROFILE[profile]

    tried: list[float] = []

    def _try(y: float) -> float:
        tried.append(y)
        box = _subtitle_box_at(
            y, box_w=subtitle_box_w, box_h=subtitle_box_h
        )
        return _overlap_fraction(box, obstacles)

    # Step 1: preferred Y.
    preferred_overlap = _try(candidate_y)
    if preferred_overlap <= overlap_clear_threshold:
        return SafeAreaDecision(
            chosen_y=candidate_y,
            opacity=1.0,
            drop=False,
            reason="preferred_clear",
            candidates_tried=tuple(tried),
        )

    # Step 2: alternative Ys.
    for alt_y in alternative_ys:
        # Skip duplicates of the preferred (numerical equality);
        # comparing floats by abs tolerance covers the case where a
        # caller passes the same Y in both lists.
        if any(abs(alt_y - t) < 1e-9 for t in tried):
            continue
        alt_overlap = _try(alt_y)
        if alt_overlap <= overlap_clear_threshold:
            return SafeAreaDecision(
                chosen_y=alt_y,
                opacity=1.0,
                drop=False,
                reason="moved_to_alternative_y",
                candidates_tried=tuple(tried),
            )

    # Step 3: opacity reduction at the preferred Y — valid only when
    # the preferred-Y overlap is at or below the drop threshold.
    # This catches the "sitting on a busy area but not buried" case.
    if preferred_overlap <= overlap_drop_threshold:
        return SafeAreaDecision(
            chosen_y=candidate_y,
            opacity=OPACITY_FLOOR,
            drop=False,
            reason="opacity_reduced",
            candidates_tried=tuple(tried),
        )

    # Step 4: drop.
    return SafeAreaDecision(
        chosen_y=candidate_y,
        opacity=0.0,
        drop=True,
        reason="subtitle_dropped",
        candidates_tried=tuple(tried),
    )


__all__ = [
    "ALTERNATIVE_Y_POSITIONS",
    "DEFAULT_SUBTITLE_BOX_H",
    "DEFAULT_SUBTITLE_BOX_W",
    "OPACITY_FLOOR",
    "OVERLAP_CLEAR_THRESHOLD",
    "OVERLAP_DROP_THRESHOLD",
    "PREFERRED_Y_BY_PROFILE",
    "Bbox",
    "PlacementFallbackReason",
    "ProfileName",
    "SafeAreaDecision",
    "validate_subtitle_position",
]
