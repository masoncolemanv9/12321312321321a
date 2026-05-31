"""Beat-sheet planner (final_spec §5).

A *beat* is the primary v6 planning unit (§5.1). Each beat carries:

* time bounds (``start_s`` / ``end_s``) in **render** time;
* a closed-enum ``purpose`` (§5.3) — hook / reveal / tension /
  emotional_hold / action / dialogue / reaction / resolution /
  transition;
* an ``emotional_intent`` free-text string when analyzer ships
  ``emotional_intent_per_beat`` (§3.3) — empty otherwise;
* exactly one ``primary_move`` (§5.5 HARD RULE);
* zero or more ``supports`` (orthogonal channels — subtitle / audio /
  color).

Boundary algorithm (§5.2): pyscenedetect scenes → transcript-density
subdivisions → emotion-change boundaries → merge micro-beats <1s.
Deterministic: same ``director_brief`` → same beat list.

This module is pure: it consumes a parsed brief and returns a frozen
list of :class:`Beat`. The worker (Part 10) is responsible for
turning it into an ``edit_plan.beat_sheet[]`` entry.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .locks import StyleLock

__all__ = (
    "BeatPurpose",
    "PrimaryMove",
    "PURPOSE_ENUM",
    "PRIMARY_MOVE_ENUM",
    "TRANSCRIPT_DENSITY_THRESHOLD",
    "Beat",
    "build_beat_sheet",
    "default_primary_move_for_purpose",
)


# ---- closed enums --------------------------------------------------


# §5.3 — closed; extension requires v6.x amendment.
PURPOSE_ENUM: tuple[str, ...] = (
    "hook",
    "reveal",
    "tension",
    "emotional_hold",
    "action",
    "dialogue",
    "reaction",
    "resolution",
    "transition",
)

# §5.5 — closed; one per beat (HARD RULE).
PRIMARY_MOVE_ENUM: tuple[str, ...] = (
    "hook_push_in",
    "slow_push_in",
    "push_out",
    "hold",
    "pan",
    "match_cut",
    "reframe",
    "static",
)


BeatPurpose = str  # one of PURPOSE_ENUM
PrimaryMove = str  # one of PRIMARY_MOVE_ENUM


# §5.2 transcript-density subdivision threshold (chars-per-second).
# Empirically tuned: dialogue ≈14 cps, monologue ≈18 cps. Boundary
# only when density >threshold within a scene.
TRANSCRIPT_DENSITY_THRESHOLD: float = 17.0

# §5.2 merge floor: anything below 1s collapses into its longer
# neighbor.
MERGE_FLOOR_S: float = 1.0

# §5.2 emotion-change confidence floor.
EMOTION_DELTA_CONFIDENCE: float = 0.6


@dataclass(frozen=True, slots=True)
class Beat:
    """One §5.1 planning unit. All times are *render* seconds.

    ``supports`` lives in a *different perceptual channel* from
    ``primary_move`` so the §5.5 one-primary-move HARD RULE is
    enforced by construction: ``primary_move`` is a single scalar;
    the planner never returns multiple primary moves per beat.
    """

    beat_id: str
    start_s: float
    end_s: float
    purpose: BeatPurpose
    primary_move: PrimaryMove
    emotional_intent: str = ""
    supports: tuple[str, ...] = ()
    note: str = ""
    reason: str = ""
    # Provenance: which boundary-source produced this beat. Useful
    # for downstream debug. Closed enum.
    source: str = "scene"  # "scene" | "transcript" | "emotion"

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


# ---- §5.4 creative-goal classification (used to bias purpose) ------


_HOOK_TEXT_SIGNALS: tuple[str, ...] = (
    "betrayal",
    "shock",
    "twist",
    "reveal",
    "secret",
    "shocking",
)


def _is_first_n_seconds(scene_start_s: float, threshold: float = 2.4) -> bool:
    """A scene starting in the first ``threshold`` seconds is a hook candidate."""
    return scene_start_s <= threshold


def _classify_purpose(
    frame_render_start_s: float,
    frame_render_end_s: float,
    *,
    is_first: bool,
    has_action: bool,
    has_emotion: str,
    reveal_window: tuple[float, float] | None,
    is_last: bool,
    action_level: float = 0.0,
) -> BeatPurpose:
    """Pick a §5.3 purpose based on rule cascade.

    All time inputs are render-time. Order matters: HARD rules first
    (action_level + reveal_window), then hook (first segment), then
    emotion-driven, then dialogue default. ``transition`` is reserved
    for explicit analyzer hints or pre-classified short scenes (<1s)
    we never reach because of §5.2 merge floor.
    """
    # 1. Action override (§6.2 hysteresis-friendly).
    if has_action or action_level >= 0.6:
        return "action"

    # 2. Reveal window (render-time).
    if reveal_window is not None and not (
        frame_render_end_s <= reveal_window[0]
        or frame_render_start_s >= reveal_window[1]
    ):
        return "reveal"

    # 3. Hook — only the first beat.
    if is_first:
        return "hook"

    # 4. Resolution — last beat.
    if is_last:
        return "resolution"

    # 5. Emotion-driven.
    if has_emotion in ("anger", "sadness", "fear", "crying"):
        return "emotional_hold"
    if has_emotion in ("shock", "surprise"):
        return "reaction"

    # 6. Dialogue default.
    return "dialogue"


def default_primary_move_for_purpose(
    purpose: BeatPurpose,
    *,
    is_first_beat: bool = False,
) -> PrimaryMove:
    """Pick the §5.5 single ``primary_move`` for a beat purpose.

    The §6.3 framing-policies table assigns multiple modes per
    purpose — the planner picks one. ``is_first_beat=True`` upgrades
    a hook purpose to ``hook_push_in``.
    """
    if purpose == "hook":
        return "hook_push_in" if is_first_beat else "slow_push_in"
    if purpose == "reveal":
        return "slow_push_in"
    if purpose == "tension":
        return "slow_push_in"
    if purpose == "emotional_hold":
        return "slow_push_in"
    if purpose == "action":
        return "hold"
    if purpose == "dialogue":
        return "hold"
    if purpose == "reaction":
        return "hold"
    if purpose == "resolution":
        return "push_out"
    if purpose == "transition":
        return "static"
    # Defensive fallback.
    return "static"


# ---- boundary-derivation pipeline ----------------------------------


@dataclass(frozen=True, slots=True)
class _BoundaryFrame:
    start_s: float
    end_s: float
    source: str  # "scene" | "transcript" | "emotion"


def _scene_boundaries(
    scenes: Iterable[Mapping[str, Any]], window_start: float, window_end: float
) -> list[_BoundaryFrame]:
    out: list[_BoundaryFrame] = []
    for s in scenes:
        if not isinstance(s, Mapping):
            continue
        try:
            ss = float(s["start_s"])
            se = float(s["end_s"])
        except (KeyError, TypeError, ValueError):
            continue
        ss = max(ss, window_start)
        se = min(se, window_end)
        if se - ss > 1e-6:
            out.append(_BoundaryFrame(start_s=ss, end_s=se, source="scene"))
    out.sort(key=lambda b: b.start_s)
    return out


def _transcript_subdivisions(
    parent: _BoundaryFrame,
    segments: list[Mapping[str, Any]],
) -> list[_BoundaryFrame]:
    """If transcript density inside ``parent`` exceeds the
    §5.2 threshold, subdivide by segment boundary."""
    inside = [
        s
        for s in segments
        if isinstance(s, Mapping)
        and float(s.get("start_s", -1)) >= parent.start_s - 1e-6
        and float(s.get("end_s", -1)) <= parent.end_s + 1e-6
    ]
    if not inside:
        return [parent]
    total_chars = sum(len(str(s.get("text", ""))) for s in inside)
    duration = max(parent.end_s - parent.start_s, 1e-6)
    cps = total_chars / duration
    if cps < TRANSCRIPT_DENSITY_THRESHOLD or len(inside) < 2:
        return [parent]
    out: list[_BoundaryFrame] = []
    cursor = parent.start_s
    for seg in inside[:-1]:
        boundary = float(seg["end_s"])
        if boundary - cursor < 1e-3:
            continue
        out.append(_BoundaryFrame(start_s=cursor, end_s=boundary, source="transcript"))
        cursor = boundary
    out.append(_BoundaryFrame(start_s=cursor, end_s=parent.end_s, source="transcript"))
    return out


def _emotion_subdivisions(
    parent: _BoundaryFrame,
    emotion_changes: list[Mapping[str, Any]],
) -> list[_BoundaryFrame]:
    inside = [
        e
        for e in emotion_changes
        if isinstance(e, Mapping)
        and float(e.get("start_s", -1)) > parent.start_s + 1e-3
        and float(e.get("start_s", -1)) < parent.end_s - 1e-3
        and float(e.get("emotion_delta_confidence", 0.0)) >= EMOTION_DELTA_CONFIDENCE
    ]
    if not inside:
        return [parent]
    out: list[_BoundaryFrame] = []
    cursor = parent.start_s
    for ev in inside:
        b = float(ev["start_s"])
        if b - cursor < 1e-3:
            continue
        out.append(_BoundaryFrame(start_s=cursor, end_s=b, source="emotion"))
        cursor = b
    out.append(_BoundaryFrame(start_s=cursor, end_s=parent.end_s, source="emotion"))
    return out


def _merge_micro(boundaries: list[_BoundaryFrame]) -> list[_BoundaryFrame]:
    """Collapse <1s frames into the longer neighbor (§5.2)."""
    if not boundaries:
        return []
    merged = list(boundaries)
    changed = True
    while changed and len(merged) > 1:
        changed = False
        for i, b in enumerate(merged):
            if b.end_s - b.start_s >= MERGE_FLOOR_S - 1e-6:
                continue
            # Choose longer neighbor.
            left = merged[i - 1] if i > 0 else None
            right = merged[i + 1] if i + 1 < len(merged) else None
            if left is None and right is None:
                break
            if left is None or (right is not None and (right.end_s - right.start_s) >= (left.end_s - left.start_s)):
                # merge with right
                assert right is not None
                merged[i] = _BoundaryFrame(
                    start_s=b.start_s, end_s=right.end_s, source=b.source
                )
                del merged[i + 1]
            else:
                # merge with left
                assert left is not None
                merged[i - 1] = _BoundaryFrame(
                    start_s=left.start_s, end_s=b.end_s, source=left.source
                )
                del merged[i]
            changed = True
            break
    return merged


def _shift_to_render_time(
    boundaries: list[_BoundaryFrame], window_start: float
) -> list[_BoundaryFrame]:
    """Translate source-time boundaries into render-time."""
    return [
        _BoundaryFrame(
            start_s=max(0.0, b.start_s - window_start),
            end_s=max(0.0, b.end_s - window_start),
            source=b.source,
        )
        for b in boundaries
        if b.end_s > b.start_s
    ]


def _hint_for_window(
    hints: list[Mapping[str, Any]], frame: _BoundaryFrame, window_start: float
) -> tuple[str, str]:
    """Return (emotion, note) from beats_hint overlap."""
    emo = ""
    note = ""
    for h in hints:
        if not isinstance(h, Mapping):
            continue
        try:
            hs = float(h.get("start_s", 0.0)) - window_start
            he = float(h.get("end_s", 0.0)) - window_start
        except (TypeError, ValueError):
            continue
        if hs >= frame.end_s - 1e-6 or he <= frame.start_s + 1e-6:
            continue
        emo = str(h.get("emotional_intent", "") or "")
        note = str(h.get("note", "") or "")
        if emo:
            break
    return emo, note


# ---- public entry point --------------------------------------------


def build_beat_sheet(
    director_brief: Mapping[str, Any],
    *,
    lock: StyleLock | None = None,
) -> tuple[Beat, ...]:
    """Build the §5 beat sheet for ``director_brief``.

    ``lock`` (style lock from :mod:`locks`) influences purpose
    assignment for the hook beat (e.g. ``cinematic_drama`` may favour
    slow_push_in over hook_push_in). It MAY NOT change boundary
    placement — boundaries are purely a function of the brief.

    Returns a frozen tuple of :class:`Beat`. Empty tuple on a brief
    that has zero scenes (validator should have caught this).
    """
    if not isinstance(director_brief, Mapping):
        return ()
    window = director_brief.get("window") or {}
    if not isinstance(window, Mapping):
        return ()
    try:
        w_start = float(window["start_s"])
        w_end = float(window["end_s"])
    except (KeyError, TypeError, ValueError):
        return ()

    scenes_raw = director_brief.get("scenes") or []
    if not isinstance(scenes_raw, list):
        return ()

    transcript_raw = director_brief.get("transcript_segments") or []
    transcript = [s for s in transcript_raw if isinstance(s, Mapping)]

    emotion_changes_raw = director_brief.get("emotion_changes") or []
    emotion_changes = [e for e in emotion_changes_raw if isinstance(e, Mapping)]

    beats_hint_raw = director_brief.get("beats_hint") or []
    beats_hint = [h for h in beats_hint_raw if isinstance(h, Mapping)]

    # 1. Scene boundaries (source-time).
    boundaries = _scene_boundaries(scenes_raw, w_start, w_end)
    if not boundaries:
        return ()

    # 2. Subdivide by transcript density.
    subdivided: list[_BoundaryFrame] = []
    for b in boundaries:
        subdivided.extend(_transcript_subdivisions(b, transcript))

    # 3. Subdivide by emotion change.
    refined: list[_BoundaryFrame] = []
    for b in subdivided:
        refined.extend(_emotion_subdivisions(b, emotion_changes))

    # 4. Merge micro-beats.
    refined = _merge_micro(refined)

    # 5. Shift to render time (subtract window.start_s).
    refined = _shift_to_render_time(refined, w_start)

    if not refined:
        return ()

    # 6. Classify purposes + assign primary moves.
    # Reveal-window in render-time.
    reveal_window: tuple[float, float] | None = None
    rl = director_brief.get("reorder_license")
    if isinstance(rl, Mapping):
        try:
            rs = float(rl["reveal_start_s"]) - w_start
            re_ = float(rl["reveal_end_s"]) - w_start
            if re_ > rs:
                reveal_window = (rs, re_)
        except (KeyError, TypeError, ValueError):
            reveal_window = None

    # Action map (scene → action_level), shifted to render time.
    scene_emotions: dict[float, str] = {}
    scene_action: dict[float, bool] = {}
    for s in scenes_raw:
        if not isinstance(s, Mapping):
            continue
        try:
            ss = float(s["start_s"]) - w_start
        except (KeyError, TypeError, ValueError):
            continue
        scene_emotions[ss] = str(s.get("emotion_label", "") or "")
        scene_action[ss] = float(s.get("action_level", 0.0) or 0.0) >= 0.6

    n = len(refined)
    beats: list[Beat] = []
    for i, frame in enumerate(refined):
        is_first = i == 0
        is_last = i == n - 1
        # Find the source-scene this frame falls inside (largest
        # overlap) to inherit action / emotion. Sorted by start, so
        # the closest preceding scene wins on ties.
        emo, note = _hint_for_window(beats_hint, frame, w_start)
        has_action = False
        scene_emotion = ""
        for ss_render, ss_action in scene_action.items():
            if ss_render <= frame.start_s + 1e-6 and ss_action:
                has_action = True
        for ss_render, emotion in scene_emotions.items():
            if ss_render <= frame.start_s + 1e-6 and emotion:
                scene_emotion = emotion
        purpose = _classify_purpose(
            frame.start_s,
            frame.end_s,
            is_first=is_first,
            has_action=has_action,
            has_emotion=emo or scene_emotion,
            reveal_window=reveal_window,
            is_last=is_last,
            action_level=0.7 if has_action else 0.0,
        )
        primary = default_primary_move_for_purpose(
            purpose, is_first_beat=is_first
        )

        # Lock interaction (only for hook beat — keeps the
        # one-primary-move HARD RULE intact).
        if lock is not None and purpose == "hook":
            override = lock.hook_treatment_overrides
            forced = override.get("primary_move") if isinstance(override, Mapping) else None
            if isinstance(forced, str) and forced in PRIMARY_MOVE_ENUM:
                primary = forced

        supports: list[str] = []
        if purpose in ("hook", "reveal"):
            supports.append("brightness_pulse")
        if purpose in ("hook", "reveal", "reaction", "dialogue"):
            supports.append("karaoke_emphasis")

        reason_bits = [f"source={frame.source}", f"is_first={is_first}", f"is_last={is_last}"]
        if has_action:
            reason_bits.append("action_level>=0.6")
        if scene_emotion:
            reason_bits.append(f"scene_emotion={scene_emotion}")
        if reveal_window is not None and purpose == "reveal":
            reason_bits.append("reveal_window_overlap")
        reason = "; ".join(reason_bits)

        beat = Beat(
            beat_id=f"beat_{i:03d}",
            start_s=frame.start_s,
            end_s=frame.end_s,
            purpose=purpose,
            primary_move=primary,
            emotional_intent=emo or scene_emotion,
            supports=tuple(supports),
            note=note,
            reason=reason,
            source=frame.source,
        )
        beats.append(beat)

    return tuple(beats)
