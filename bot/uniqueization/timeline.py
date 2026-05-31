"""Source-time ↔ render-time mapping for v2.1 cuts.

After :mod:`.cuts` decides which source-time intervals survive, every
other v2.1 module needs to ask the inverse question: *given a
source-time event (a whisper word, a scene boundary, a mirror cluster
interval), where does it live in render-time?*

That's what :class:`TimelineMap` answers. Piecewise-linear v1 (§10.2 /
§10.3) — no time-warping yet; every kept segment maps with a slope of
1.0 to a render interval. Future v2 maps may introduce speed-up /
slow-down per segment; the public API hides that detail.

Constraints (pure-function module, §10.4):

* No I/O, no subprocess.
* Deterministic. Same kept_segments → same TimelineMap → same remaps.
* Monotonicity-validated: both axes must be strictly increasing
  across segments. Non-monotonic input raises :class:`TimelineError`
  (caught one layer up and turned into the
  ``validation_error_timeline_map`` payload error from §10.3).
* Cross-validation with analyzer-shipped ``timeline_map`` is the
  Part 7 worker's job; this module supplies the helper
  :func:`timeline_map_match` used by that check.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

from .cuts import KeptSegment


class TimelineError(ValueError):
    """Raised when the timeline map is structurally inconsistent.

    Worker callers convert this to ``validation_error_timeline_map``
    (§10.3 abort-class). Tests use it as the canonical assertion
    surface for "this should never validate".
    """


@dataclass(frozen=True, slots=True)
class TimelineSegment:
    """One piecewise-linear segment: ``source[a..b] -> render[c..d]``.

    The v1 contract requires ``b - a == d - c`` (slope-1). Stored
    redundantly so consumers don't have to re-derive durations.
    """

    source_start_s: float
    source_end_s: float
    render_start_s: float
    render_end_s: float

    @property
    def duration_s(self) -> float:
        return self.source_end_s - self.source_start_s


@dataclass(frozen=True, slots=True)
class TimelineMap:
    """Piecewise-linear v1 timeline map.

    Use :func:`derive_timeline_map` to build one from kept_segments.
    Direct construction is reserved for tests; the public path should
    always go through the validating builder.
    """

    segments: tuple[TimelineSegment, ...]
    version: int = 1

    @property
    def render_duration_s(self) -> float:
        return self.segments[-1].render_end_s if self.segments else 0.0

    @property
    def source_duration_s(self) -> float:
        return self.segments[-1].source_end_s if self.segments else 0.0

    def find_segment_for_source_time(
        self, source_s: float
    ) -> TimelineSegment | None:
        """Return the segment containing ``source_s``, or ``None``.

        Boundary semantics: a source time exactly at a segment's
        ``source_end_s`` belongs to the *next* segment if one exists,
        matching how ffmpeg's ``between(t,a,b)`` interprets ``b`` as
        exclusive. The very last segment owns its own end (so the
        clip-end timestamp doesn't fall off the map).
        """
        if not self.segments:
            return None
        for i, seg in enumerate(self.segments):
            is_last = i == len(self.segments) - 1
            if seg.source_start_s <= source_s < seg.source_end_s:
                return seg
            if is_last and source_s == seg.source_end_s:
                return seg
        return None


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def derive_timeline_map(
    kept_segments: Sequence[KeptSegment],
) -> TimelineMap:
    """Build a piecewise-linear v1 timeline map from kept_segments.

    The map is monotone-checked: input segments must already be
    sorted by ``source_start_s``, non-overlapping, and have positive
    durations. The mapping itself is slope-1 — render-time advances
    by exactly the source-time elapsed within each segment.

    Returns an empty :class:`TimelineMap` when ``kept_segments`` is
    empty (degenerate case: every source-second was cut). Callers
    interpret that as "abort — no render content" (§9.13).
    """
    if not kept_segments:
        return TimelineMap(segments=())

    out: list[TimelineSegment] = []
    render_cursor = 0.0
    for i, seg in enumerate(kept_segments):
        if seg.source_end_s <= seg.source_start_s:
            raise TimelineError(
                f"kept_segments[{i}] has non-positive duration: "
                f"[{seg.source_start_s},{seg.source_end_s}]"
            )
        if out and seg.source_start_s < out[-1].source_end_s:
            raise TimelineError(
                f"kept_segments[{i}] starts at {seg.source_start_s}, "
                f"overlapping previous end {out[-1].source_end_s}"
            )
        duration = seg.source_end_s - seg.source_start_s
        out.append(
            TimelineSegment(
                source_start_s=seg.source_start_s,
                source_end_s=seg.source_end_s,
                render_start_s=render_cursor,
                render_end_s=render_cursor + duration,
            )
        )
        render_cursor += duration

    return TimelineMap(segments=tuple(out))


# ---------------------------------------------------------------------------
# Remapping
# ---------------------------------------------------------------------------


def _remap_point(
    timeline_map: TimelineMap, source_s: float
) -> float | None:
    """Map a single source-time point to render-time.

    Returns ``None`` for points that fall inside a *cut* — i.e.
    between two kept segments. The two interval/word remapping
    helpers below decide what to do with such points.
    """
    seg = timeline_map.find_segment_for_source_time(source_s)
    if seg is None:
        return None
    offset = source_s - seg.source_start_s
    return seg.render_start_s + offset


def remap_intervals(
    intervals: Sequence[tuple[float, float]],
    timeline_map: TimelineMap,
) -> list[tuple[float, float]]:
    """Remap source-time intervals to render-time.

    Each input ``(src_start, src_end)`` may overlap zero, one, or many
    kept segments. The function emits one render-time interval **per
    crossed segment**, clipped to the segment's boundary on either
    side. Intervals that fall entirely inside cuts are dropped (no
    render-time presence). This matches §10.6's cluster-mirror
    expectations: a cluster spanning a cut produces two separate
    ``between(t,...)`` ranges, not one straddling range.
    """
    if not timeline_map.segments:
        return []
    out: list[tuple[float, float]] = []
    for src_start, src_end in intervals:
        if src_end <= src_start:
            continue
        for seg in timeline_map.segments:
            overlap_start = max(src_start, seg.source_start_s)
            overlap_end = min(src_end, seg.source_end_s)
            if overlap_end <= overlap_start:
                continue
            rs = seg.render_start_s + (overlap_start - seg.source_start_s)
            re = seg.render_start_s + (overlap_end - seg.source_start_s)
            out.append((rs, re))
    return out


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class TimedWord(Generic[T]):
    """Generic source-time word that can carry whatever payload.

    The v2.1 worker (Part 7) constructs these from the whisper
    transcript before calling :func:`remap_words`; the subtitle
    renderer (Part 5) consumes the remapped output. Keeping the
    payload generic means this module doesn't have to know about
    whisper's word schema yet.
    """

    source_start_s: float
    source_end_s: float
    payload: T


@dataclass(frozen=True, slots=True)
class RenderWord(Generic[T]):
    """One whisper word remapped to render-time.

    Words that straddle a cut are *clipped* to the surviving slice
    (we don't drop characters mid-word). Words that fall entirely
    inside a cut are filtered out by :func:`remap_words`.
    """

    source_start_s: float
    source_end_s: float
    render_start_s: float
    render_end_s: float
    payload: T


def remap_words(
    words: Sequence[TimedWord[T]],
    timeline_map: TimelineMap,
) -> list[RenderWord[T]]:
    """Remap whisper words to render-time, dropping cut-only words.

    For words that straddle a cut boundary: the output preserves the
    word but trims its render-time to the surviving portion. This is
    the gentlest choice — alternatives (drop the word entirely, or
    glue together two render fragments) all produce worse subtitles.
    Trimming a 200ms word to 50ms is still readable; dropping it
    leaves a hole.
    """
    if not timeline_map.segments:
        return []
    out: list[RenderWord[T]] = []
    for word in words:
        if word.source_end_s <= word.source_start_s:
            continue
        # First render-time fragment wins. If a word straddles
        # multiple cuts we keep the *first* surviving segment only —
        # multi-segment words are rare and rendering them twice would
        # produce duplicated subtitles.
        for seg in timeline_map.segments:
            overlap_start = max(word.source_start_s, seg.source_start_s)
            overlap_end = min(word.source_end_s, seg.source_end_s)
            if overlap_end <= overlap_start:
                continue
            rs = seg.render_start_s + (overlap_start - seg.source_start_s)
            re = seg.render_start_s + (overlap_end - seg.source_start_s)
            out.append(
                RenderWord(
                    source_start_s=word.source_start_s,
                    source_end_s=word.source_end_s,
                    render_start_s=rs,
                    render_end_s=re,
                    payload=word.payload,
                )
            )
            break
    return out


# ---------------------------------------------------------------------------
# Cross-validation hook (Part 7 calls this to enforce §10.3).
# ---------------------------------------------------------------------------


def timeline_map_match(
    derived: TimelineMap,
    shipped: TimelineMap,
    *,
    abs_tol_s: float = 1e-3,
) -> bool:
    """Compare editor-derived vs analyzer-shipped timeline maps.

    Used by Part 7 to enforce §10.3 ``validation_error_timeline_map``.
    Compares segment count, version, and every endpoint within 1ms
    tolerance.
    """
    if derived.version != shipped.version:
        return False
    if len(derived.segments) != len(shipped.segments):
        return False
    for d, s in zip(derived.segments, shipped.segments, strict=True):
        if abs(d.source_start_s - s.source_start_s) > abs_tol_s:
            return False
        if abs(d.source_end_s - s.source_end_s) > abs_tol_s:
            return False
        if abs(d.render_start_s - s.render_start_s) > abs_tol_s:
            return False
        if abs(d.render_end_s - s.render_end_s) > abs_tol_s:
            return False
    return True


def identity_map(source_duration_s: float) -> TimelineMap:
    """v2.0 fallback: single segment, source==render.

    Returned by Part 7 when the payload has no ``timeline_map`` /
    ``kept_segments`` fields (v2.0 path or analyzer-v2 disabled).
    Keeps remapping helpers usable without a special "no map"
    branch in the caller.
    """
    if source_duration_s <= 0:
        raise TimelineError(
            f"identity_map needs positive source_duration_s, got {source_duration_s!r}"
        )
    return TimelineMap(
        segments=(
            TimelineSegment(
                source_start_s=0.0,
                source_end_s=source_duration_s,
                render_start_s=0.0,
                render_end_s=source_duration_s,
            ),
        )
    )


__all__ = [
    "RenderWord",
    "TimedWord",
    "TimelineError",
    "TimelineMap",
    "TimelineSegment",
    "derive_timeline_map",
    "identity_map",
    "remap_intervals",
    "remap_words",
    "timeline_map_match",
]
