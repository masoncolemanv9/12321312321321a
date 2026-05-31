"""Cut application — Part 4 of the Editor Agent split.

``apply_cuts()`` takes a source duration and a list of source-time cut
intervals (delivered by the analyzer in the v2.1 payload, see
``final_spec_FULL.md`` §10.2) and returns the **kept segments** in
source-time plus the ffmpeg ``filter_complex`` label pairs the v2.1
worker (Part 7) will wire into the fused encode.

Design constraints (pure-function module):

* **No I/O, no subprocess.** This module computes intervals only. The
  ffmpeg invocation lives in Part 1's :mod:`bot.uniqueization.runner`,
  and the filtergraph assembly lives in Part 2's
  :mod:`bot.uniqueization.filtergraph`. Part 7 wires the two together.
* **Deterministic.** Same inputs → same outputs, byte-for-byte. The
  v2.1 payload validator in Part 7 cross-checks the kept-segments we
  derive here against the analyzer-shipped ``kept_segments[]`` and
  fails on mismatch (§10.3 ``validation_error_kept_segments``).
* **Safety constraints from §7.8 / §7.9.** While the analyzer is the
  *authoritative* enforcer of cut safety (min-duration, scene buffer,
  emotional silence, mid-word), the editor performs a **defensive
  pass** before applying: any cut that violates a documented safety
  rule is *skipped* (not silently honoured), and the rejection is
  recorded in :class:`AppliedCuts.skipped` for the manifest. The
  worker uses this list to surface a warning rather than aborting —
  abort semantics live one layer up.

Spec references:
    * §7.8 — cuts safety constraints (analyzer-side enforcement,
      defensively re-checked here).
    * §7.9 — cuts × scene boundaries (cuts subtracted FIRST, then
      everything else remaps via :mod:`.timeline`).
    * §7.10 — v2.1 cuts pipeline order.
    * §10.2 / §10.3 — payload shape and field semantics.
    * §10.4 — module additions list (cuts.py).
    * §10.5 — formal v2.1 stage execution order.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------
# Safety thresholds (§7.8). Mirrors ``ANALYZER_CUT_*`` env defaults so the
# editor's defensive check accepts what an in-spec analyzer ships.
# These are NOT configurable from the editor side — they describe the
# analyzer contract. Per-job overrides live in the analyzer.
# ---------------------------------------------------------------------------

#: Minimum duration for ``filler_word`` cuts (seconds). §7.8.
MIN_FILLER_CUT_S: float = 0.25

#: Minimum duration for ``silence_pause`` cuts (seconds). §7.8.
MIN_SILENCE_CUT_S: float = 0.5

#: Minimum kept-segment duration (seconds). §7.8.
MIN_KEPT_SEGMENT_S: float = 0.5

#: Scene-boundary buffer (seconds) — no cuts within this distance of a
#: scene boundary. §7.8.
SCENE_BOUNDARY_BUFFER_S: float = 0.15

CutKind = Literal["filler_word", "silence_pause", "dead_air"]

#: Closed enum of skip reasons for the manifest. Stable identifiers so
#: downstream tooling (analyzer-v2 tuning, dashboards) can rely on them.
SkipReason = Literal[
    "below_min_duration",
    "negative_or_zero_duration",
    "out_of_source_bounds",
    "overlaps_scene_boundary",
    "inside_emotional_silence",
    "would_leave_segment_below_min",
    "overlaps_previous_cut",
]


class CutsError(ValueError):
    """Raised when the cuts input is structurally invalid.

    Use sparingly: the worker treats this as a fatal payload error
    (§9.13 abort-class). Most issues should produce a *skipped* entry
    instead so the render proceeds with degraded but observable
    behaviour.
    """


@dataclass(frozen=True, slots=True)
class Cut:
    """One source-time interval the analyzer wants removed."""

    start_s: float
    end_s: float
    kind: CutKind
    reason: str = ""
    confidence: float = 1.0

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True, slots=True)
class KeptSegment:
    """Source-time slice of media that survives all cuts.

    ``render_start_s`` / ``render_end_s`` are intentionally *not*
    populated here — they belong to :mod:`.timeline`, which computes
    the source→render mapping after cuts are applied. Keeping the two
    concerns split keeps each module's contract local.
    """

    source_start_s: float
    source_end_s: float

    @property
    def duration_s(self) -> float:
        return self.source_end_s - self.source_start_s


@dataclass(frozen=True, slots=True)
class SkippedCut:
    """A cut the editor refused to honour, with a stable reason.

    Recorded in the v2.1 manifest under
    ``stages.cuts.extra.skipped_cuts[]`` (§7.8 last bullet:
    "Manifest records ALL skipped cuts with reason for
    debugging/tuning"). The original cut is preserved verbatim so
    downstream tooling can replay it.
    """

    cut: Cut
    reason: SkipReason
    detail: str = ""


@dataclass(frozen=True, slots=True)
class AppliedCuts:
    """Return type of :func:`apply_cuts`.

    Three concerns travel together so callers don't have to recombine
    them:

    * ``kept_segments`` — the source-time slices that survive, in
      monotonically increasing order, non-overlapping. Feeds
      :func:`bot.uniqueization.timeline.derive_timeline_map`.
    * ``filtergraph_labels`` — per-segment ``[v0]``/``[a0]`` etc. label
      pairs the Part 7 worker concatenates into the fused
      ``filter_complex``. The exact ffmpeg ``trim``/``atrim``/``concat``
      assembly lives in Part 2's filtergraph builder; this module only
      hands back the *labels* so the assembly step doesn't have to
      re-walk the segment list.
    * ``skipped`` — defensive-skip log for the manifest.
    """

    kept_segments: tuple[KeptSegment, ...]
    filtergraph_labels: tuple[SegmentLabels, ...]
    skipped: tuple[SkippedCut, ...] = ()


@dataclass(frozen=True, slots=True)
class SegmentLabels:
    """Filtergraph label pair for one kept segment.

    ``video_label`` / ``audio_label`` are bare ffmpeg labels (without
    the surrounding square brackets) so callers can format them
    however they need to. ``index`` is the zero-based position of the
    segment in the kept list — useful when downstream code needs to
    emit ``[v0][a0][v1][a1]...concat=...``.
    """

    index: int
    video_label: str
    audio_label: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _min_duration_for_kind(kind: CutKind) -> float:
    if kind == "filler_word":
        return MIN_FILLER_CUT_S
    if kind == "silence_pause":
        return MIN_SILENCE_CUT_S
    # dead_air has no documented minimum in §7.8 — fall back to silence's
    # tighter threshold to keep behaviour conservative.
    return MIN_SILENCE_CUT_S


def _interval_intersects(
    a_start: float, a_end: float, b_start: float, b_end: float
) -> bool:
    """Open-interval intersection (touching endpoints don't intersect)."""
    return a_start < b_end and b_start < a_end


def _interval_contains(
    outer_start: float, outer_end: float, inner_start: float, inner_end: float
) -> bool:
    return outer_start <= inner_start and inner_end <= outer_end


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_cuts(
    source_duration_s: float,
    cuts: Sequence[Cut],
    *,
    scene_boundaries_s: Sequence[float] = (),
    emotional_silence_segments_s: Sequence[tuple[float, float]] = (),
    min_kept_segment_s: float = MIN_KEPT_SEGMENT_S,
    scene_buffer_s: float = SCENE_BOUNDARY_BUFFER_S,
) -> AppliedCuts:
    """Apply analyzer cuts to a source clip; return kept segments.

    The algorithm (§7.9 — "Editor subtracts cuts FIRST"):

    1. Validate ``source_duration_s`` is positive.
    2. Sort cuts by ``start_s`` (stable). Defensive: analyzer ships
       sorted, but we don't trust the wire format.
    3. For each cut, run the safety gauntlet (§7.8). Failures go to
       ``skipped`` with a stable reason.
    4. Walk the timeline, emitting ``KeptSegment`` for every gap
       between honoured cuts (and the head/tail of the clip).
    5. Reject kept segments shorter than ``min_kept_segment_s``: if
       removing them is unsafe, merge with neighbour by relaxing the
       triggering cut instead. The merge logic is intentionally simple
       — relaxing means *un-honouring* the cut that produced the
       too-short segment.
    6. Assign deterministic filtergraph labels (``v0``/``a0``,
       ``v1``/``a1``, …).

    Args:
        source_duration_s: Length of the source clip, in seconds.
            Must be ``> 0``.
        cuts: The analyzer-shipped cuts (source-time intervals).
        scene_boundaries_s: Optional list of scene boundary timestamps
            (source-time). When provided, cuts that fall within
            ``scene_buffer_s`` of any boundary are skipped (§7.8).
            Pass an empty sequence to disable the check.
        emotional_silence_segments_s: Optional list of
            ``(start_s, end_s)`` pairs marking emotional-signal silence
            (analyzer flag, see §7.8 — "no cuts inside emotional
            silence"). Cuts overlapping any such segment are skipped.
        min_kept_segment_s: Override for the kept-segment floor.
            Defaults to §7.8's 500ms.
        scene_buffer_s: Override for the scene-boundary buffer.

    Returns:
        :class:`AppliedCuts` carrying ``kept_segments`` (monotonic,
        non-overlapping), per-segment filtergraph labels, and a
        ``skipped`` audit list.

    Raises:
        CutsError: ``source_duration_s`` is non-positive or a cut is
            structurally malformed (NaN, etc.). Per-cut safety
            failures **do not** raise — they are recorded as skipped.
    """

    if source_duration_s <= 0:
        raise CutsError(
            f"source_duration_s must be positive, got {source_duration_s!r}"
        )

    # 1. Sort defensively. Stable sort preserves analyzer order for
    # equal start times — matters for the duplicate/overlap check below.
    sorted_cuts: list[Cut] = sorted(cuts, key=lambda c: c.start_s)

    accepted: list[Cut] = []
    skipped: list[SkippedCut] = []

    for cut in sorted_cuts:
        # Structural checks — keep tight (these are spec invariants
        # the analyzer should never break).
        if cut.start_s != cut.start_s or cut.end_s != cut.end_s:  # NaN
            raise CutsError(f"cut endpoints contain NaN: {cut!r}")
        if cut.duration_s <= 0:
            skipped.append(
                SkippedCut(cut=cut, reason="negative_or_zero_duration")
            )
            continue
        if cut.start_s < 0 or cut.end_s > source_duration_s:
            skipped.append(
                SkippedCut(
                    cut=cut,
                    reason="out_of_source_bounds",
                    detail=(
                        f"cut [{cut.start_s},{cut.end_s}] outside "
                        f"[0,{source_duration_s}]"
                    ),
                )
            )
            continue

        # Min-duration check by kind (§7.8 first two bullets).
        min_dur = _min_duration_for_kind(cut.kind)
        if cut.duration_s + 1e-9 < min_dur:
            skipped.append(
                SkippedCut(
                    cut=cut,
                    reason="below_min_duration",
                    detail=(
                        f"{cut.kind} cut of {cut.duration_s:.3f}s is below "
                        f"min {min_dur:.3f}s"
                    ),
                )
            )
            continue

        # Scene-boundary buffer (§7.8 fourth bullet).
        if scene_boundaries_s and _violates_scene_buffer(
            cut, scene_boundaries_s, scene_buffer_s
        ):
            skipped.append(
                SkippedCut(
                    cut=cut,
                    reason="overlaps_scene_boundary",
                    detail=(
                        f"cut within {scene_buffer_s:.3f}s of a "
                        f"scene boundary"
                    ),
                )
            )
            continue

        # Emotional silence (§7.8 list bullet "no cuts in emotional
        # silence" — wording from Part 4 prompt).
        if emotional_silence_segments_s and _overlaps_any(
            cut.start_s, cut.end_s, emotional_silence_segments_s
        ):
            skipped.append(
                SkippedCut(cut=cut, reason="inside_emotional_silence")
            )
            continue

        # Overlap with previously-accepted cut. Analyzer is supposed
        # to merge these upstream; the defensive check keeps us honest
        # if it doesn't (avoids producing zero-length kept segments).
        if accepted and cut.start_s < accepted[-1].end_s:
            skipped.append(
                SkippedCut(
                    cut=cut,
                    reason="overlaps_previous_cut",
                    detail=(
                        f"cut starts at {cut.start_s:.3f}s, previous "
                        f"cut ends at {accepted[-1].end_s:.3f}s"
                    ),
                )
            )
            continue

        accepted.append(cut)

    # 4. Walk the timeline and emit kept segments.
    kept = _emit_kept_segments(source_duration_s, accepted)

    # 5. Enforce min_kept_segment by un-honouring offending cuts. Walk
    # the kept list; for each too-short segment, find the cut that
    # produced it (left or right boundary) and move it back to
    # ``skipped`` with reason ``would_leave_segment_below_min``.
    kept, accepted, demoted = _enforce_min_kept(
        source_duration_s, accepted, kept, min_kept_segment_s
    )
    skipped.extend(demoted)

    # 6. Materialise filtergraph labels.
    labels = tuple(
        SegmentLabels(index=i, video_label=f"v{i}", audio_label=f"a{i}")
        for i in range(len(kept))
    )

    return AppliedCuts(
        kept_segments=tuple(kept),
        filtergraph_labels=labels,
        skipped=tuple(skipped),
    )


# ---------------------------------------------------------------------------
# Step helpers (split out for testability)
# ---------------------------------------------------------------------------


def _violates_scene_buffer(
    cut: Cut, boundaries_s: Sequence[float], buffer_s: float
) -> bool:
    for b in boundaries_s:
        if abs(cut.start_s - b) < buffer_s or abs(cut.end_s - b) < buffer_s:
            return True
        # Cut straddling a boundary is also a violation.
        if cut.start_s < b < cut.end_s:
            return True
    return False


def _overlaps_any(
    start_s: float,
    end_s: float,
    segments: Iterable[tuple[float, float]],
) -> bool:
    for seg_start, seg_end in segments:
        if _interval_intersects(start_s, end_s, seg_start, seg_end):
            return True
    return False


def _emit_kept_segments(
    source_duration_s: float, accepted: Sequence[Cut]
) -> list[KeptSegment]:
    kept: list[KeptSegment] = []
    cursor = 0.0
    for cut in accepted:
        if cut.start_s > cursor:
            kept.append(
                KeptSegment(source_start_s=cursor, source_end_s=cut.start_s)
            )
        cursor = max(cursor, cut.end_s)
    if cursor < source_duration_s:
        kept.append(
            KeptSegment(
                source_start_s=cursor,
                source_end_s=source_duration_s,
            )
        )
    return kept


def _enforce_min_kept(
    source_duration_s: float,
    accepted: list[Cut],
    kept: list[KeptSegment],
    min_kept_s: float,
) -> tuple[list[KeptSegment], list[Cut], list[SkippedCut]]:
    """Drop cuts that produce too-short kept segments.

    Strategy: iterate kept segments; if any is below ``min_kept_s``,
    blame the adjacent cut (preferring the smaller-confidence one) and
    re-emit. Re-runs until stable. Bounded iteration count by
    ``len(accepted) + 1`` so a misbehaving input can't loop forever.
    """

    demoted: list[SkippedCut] = []
    if not accepted:
        return kept, accepted, demoted

    for _ in range(len(accepted) + 1):
        offender_idx = next(
            (
                i
                for i, seg in enumerate(kept)
                if seg.duration_s + 1e-9 < min_kept_s
            ),
            None,
        )
        if offender_idx is None:
            return kept, accepted, demoted

        # Find the cut(s) bounding the offending kept segment. The
        # left bound is the (offender_idx-1)-th honoured cut's end;
        # the right bound is the (offender_idx)-th honoured cut's
        # start (or source end). Drop one — pick the lower-confidence
        # cut, ties broken by left first.
        left_cut_idx = offender_idx - 1 if offender_idx > 0 else None
        right_cut_idx = offender_idx if offender_idx < len(accepted) else None
        candidates = [i for i in (left_cut_idx, right_cut_idx) if i is not None]
        if not candidates:
            # No cut to demote — segment is the whole clip and
            # source_duration_s itself is below the floor. Surface the
            # situation but don't infinite-loop.
            return kept, accepted, demoted
        chosen = min(candidates, key=lambda i: accepted[i].confidence)
        demoted.append(
            SkippedCut(
                cut=accepted[chosen],
                reason="would_leave_segment_below_min",
                detail=(
                    f"kept segment "
                    f"[{kept[offender_idx].source_start_s:.3f},"
                    f"{kept[offender_idx].source_end_s:.3f}] is "
                    f"{kept[offender_idx].duration_s:.3f}s, below "
                    f"{min_kept_s:.3f}s floor"
                ),
            )
        )
        accepted = accepted[:chosen] + accepted[chosen + 1 :]
        kept = _emit_kept_segments(source_duration_s, accepted)

    return kept, accepted, demoted


# ---------------------------------------------------------------------------
# Validation helpers (used by Part 7 cross-check; exposed for tests).
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ShippedKeptSegment:
    """The analyzer-shipped kept_segment shape (matches §10.2)."""

    source_start_s: float
    source_end_s: float
    render_start_s: float = field(default=0.0)
    render_end_s: float = field(default=0.0)


def kept_segments_match(
    derived: Sequence[KeptSegment],
    shipped: Sequence[ShippedKeptSegment],
    *,
    abs_tol_s: float = 1e-3,
) -> bool:
    """Compare editor-derived vs analyzer-shipped kept_segments.

    Used by Part 7 to enforce §10.3 ``validation_error_kept_segments``.
    Comparison is on **source-time only** — render-time is the
    timeline module's job, validated separately. Tolerance is 1ms
    (analyzer's float precision floor).
    """
    if len(derived) != len(shipped):
        return False
    for d, s in zip(derived, shipped, strict=True):
        if abs(d.source_start_s - s.source_start_s) > abs_tol_s:
            return False
        if abs(d.source_end_s - s.source_end_s) > abs_tol_s:
            return False
    return True


__all__ = [
    "MIN_FILLER_CUT_S",
    "MIN_KEPT_SEGMENT_S",
    "MIN_SILENCE_CUT_S",
    "SCENE_BOUNDARY_BUFFER_S",
    "AppliedCuts",
    "Cut",
    "CutKind",
    "CutsError",
    "KeptSegment",
    "SegmentLabels",
    "ShippedKeptSegment",
    "SkipReason",
    "SkippedCut",
    "apply_cuts",
    "kept_segments_match",
]
