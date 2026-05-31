"""Cut-planner (final_spec §7.1 – §7.5, §7.10).

Two-stage decision pipeline:

1. Analyzer ships ``director_brief.cuts[]`` with candidate intervals
   (``kind ∈ {filler, dead_air, repeat}``).
2. Editor's :func:`plan_cuts_final` writes one :class:`CutDecision`
   per candidate. Every candidate gets ``apply: bool`` + ``reason``
   (§7.1 last paragraph mandates a reason for the manifest).

Hard preservation rules (§7.3):

* **Emotional silence** — analyzer marks
  ``silence_segments[].emotional_signal=true`` → any cut overlapping
  that interval gets ``apply=false`` with reason
  ``emotional_signal_preserved`` (§7.3.1).
* **Reaction window after reveal/hook** — protect a profile-keyed
  window after the reveal/hook beat (§7.3.2).
* **Action continuity** — no cut inside any
  ``purpose=action AND action_level>=action_threshold`` beat
  (§7.3.3). HARD RULE.

Profile / lock dialogue tightening (§7.2): controls per-kind
thresholds. Lock can override profile thresholds.

Hook reorder (§7.4) is tightly constrained — the planner records
the *eligibility* of reorder but does not execute the reorder
itself (that's the worker's job). When the analyzer ships
``reorder_license`` and the four preconditions are met, the planner
marks the decision result with ``reorder_eligible=True``.
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .beat_sheet import Beat
from .locks import StyleLock
from .zoom_planner import ACTION_LEVEL_ENTER

__all__ = (
    "CUT_KIND_ENUM",
    "CutDecision",
    "CutPlan",
    "PROFILE_CUT_THRESHOLDS",
    "REACTION_WINDOW_BY_PROFILE",
    "plan_cuts_final",
)


CUT_KIND_ENUM: tuple[str, ...] = ("filler", "dead_air", "repeat")


# ---- profile thresholds (§7.2) -------------------------------------


PROFILE_CUT_THRESHOLDS: Mapping[str, Mapping[str, float]] = {
    "light": {
        "filler_confidence_min": 0.80,
        "dead_air_threshold_s": 0.8,
    },
    "medium": {
        "filler_confidence_min": 0.70,
        "dead_air_threshold_s": 0.5,
    },
    "heavy": {
        "filler_confidence_min": 0.60,
        "dead_air_threshold_s": 0.3,
    },
}


# §7.3.2 reaction-window length after reveal/hook beat.
REACTION_WINDOW_BY_PROFILE: Mapping[str, float] = {
    "light": 1.5,
    "medium": 2.0,
    "heavy": 3.0,
}


# ---- dataclasses ---------------------------------------------------


@dataclass(frozen=True, slots=True)
class CutDecision:
    """One §4 ``cuts_final[]`` entry.

    ``candidate_id`` echoes the analyzer-provided id when present,
    else a deterministic ``cut_NNN`` synthesised from list order.
    Source-time bounds. The worker translates them into render-time
    via :mod:`timeline` after picking which to apply.
    """

    candidate_id: str
    start_s: float
    end_s: float
    kind: str
    apply: bool
    reason: str
    confidence: float = 1.0

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True, slots=True)
class CutPlan:
    """Container for :func:`plan_cuts_final` output."""

    decisions: tuple[CutDecision, ...] = field(default_factory=tuple)
    reorder_eligible: bool = False
    reorder_reason: str = ""
    warnings: tuple[str, ...] = field(default_factory=tuple)


# ---- threshold resolution ------------------------------------------


def _resolve_thresholds(
    profile: str, lock: StyleLock | None
) -> tuple[float, float]:
    base = PROFILE_CUT_THRESHOLDS.get(profile, PROFILE_CUT_THRESHOLDS["medium"])
    filler_min = float(base["filler_confidence_min"])
    dead_air_min = float(base["dead_air_threshold_s"])
    if lock is not None and isinstance(lock.cut_threshold_overrides, Mapping):
        ov = lock.cut_threshold_overrides
        if "filler_confidence_min" in ov:
            with contextlib.suppress(TypeError, ValueError):
                filler_min = float(ov["filler_confidence_min"])
        if "dead_air_threshold_s" in ov:
            with contextlib.suppress(TypeError, ValueError):
                dead_air_min = float(ov["dead_air_threshold_s"])
    return filler_min, dead_air_min


# ---- HARD preservation helpers -------------------------------------


def _emotional_silences(brief: Mapping[str, Any]) -> tuple[tuple[float, float], ...]:
    """All silence intervals with ``emotional_signal=true`` (source time)."""
    sil = brief.get("silence_segments") or []
    if not isinstance(sil, list):
        return ()
    out: list[tuple[float, float]] = []
    for s in sil:
        if not isinstance(s, Mapping):
            continue
        if not bool(s.get("emotional_signal", False)):
            continue
        try:
            out.append((float(s["start_s"]), float(s["end_s"])))
        except (KeyError, TypeError, ValueError):
            continue
    return tuple(out)


def _action_windows_source_time(
    brief: Mapping[str, Any],
    beat_sheet: tuple[Beat, ...],
    action_threshold: float,
) -> tuple[tuple[float, float], ...]:
    """Source-time intervals where action continuity (§7.3.3) is HARD.

    Two channels feed this:

    * ``director_brief.scenes[]`` with ``action_level >= threshold``;
    * any :class:`Beat` with ``purpose=action`` (render time).
      Render-time intervals are shifted back to source time via the
      brief's ``window.start_s`` offset.
    """
    out: list[tuple[float, float]] = []

    scenes = brief.get("scenes") or []
    if isinstance(scenes, list):
        for s in scenes:
            if not isinstance(s, Mapping):
                continue
            try:
                level = float(s.get("action_level", 0.0) or 0.0)
                if level < action_threshold:
                    continue
                out.append((float(s["start_s"]), float(s["end_s"])))
            except (KeyError, TypeError, ValueError):
                continue

    window = brief.get("window") or {}
    if isinstance(window, Mapping):
        try:
            w_start = float(window.get("start_s", 0.0))
        except (TypeError, ValueError):
            w_start = 0.0
    else:
        w_start = 0.0
    for b in beat_sheet:
        if b.purpose == "action":
            out.append((b.start_s + w_start, b.end_s + w_start))

    return tuple(out)


def _reaction_windows_source_time(
    brief: Mapping[str, Any],
    beat_sheet: tuple[Beat, ...],
    profile: str,
) -> tuple[tuple[float, float], ...]:
    """Source-time intervals where reaction protection (§7.3.2) applies.

    Window length is profile-keyed: light=1.5s / medium=2.0s /
    heavy=3.0s. Starts at the *end* of any reveal or hook beat with
    sufficient ``reveal_confidence`` (when analyzer ships it; default
    1.0).
    """
    window_len = REACTION_WINDOW_BY_PROFILE.get(profile, 2.0)
    window = brief.get("window") or {}
    if isinstance(window, Mapping):
        try:
            w_start = float(window.get("start_s", 0.0))
        except (TypeError, ValueError):
            w_start = 0.0
    else:
        w_start = 0.0

    # Optional analyzer reveal_confidence keyed by beat-id.
    confidences: dict[str, float] = {}
    rcs = brief.get("reveal_confidences") or {}
    if isinstance(rcs, Mapping):
        for k, v in rcs.items():
            try:
                confidences[str(k)] = float(v)
            except (TypeError, ValueError):
                continue

    out: list[tuple[float, float]] = []
    # Sort beats so we can truncate window at the next reveal start.
    sorted_beats = sorted(beat_sheet, key=lambda b: b.start_s)
    next_reveal_start: dict[str, float | None] = {}
    for i, b in enumerate(sorted_beats):
        nxt: float | None = None
        for b2 in sorted_beats[i + 1 :]:
            if b2.purpose in ("reveal", "hook"):
                nxt = b2.start_s
                break
        next_reveal_start[b.beat_id] = nxt

    for b in sorted_beats:
        if b.purpose not in ("reveal", "hook"):
            continue
        conf = confidences.get(b.beat_id, 1.0)
        if conf < 0.7:
            continue
        proposed_end_render = b.end_s + window_len
        nxt = next_reveal_start.get(b.beat_id)
        if nxt is not None:
            proposed_end_render = min(proposed_end_render, nxt)
        # If reveal is the last beat, extend to the last beat's end.
        if proposed_end_render <= b.end_s:
            continue
        out.append((b.end_s + w_start, proposed_end_render + w_start))
    return tuple(out)


def _overlaps_any(
    start: float, end: float, ranges: tuple[tuple[float, float], ...]
) -> tuple[float, float] | None:
    """Return the *first* range that overlaps ``[start,end)``."""
    for r in ranges:
        if start < r[1] - 1e-6 and end > r[0] + 1e-6:
            return r
    return None


# ---- reorder eligibility (§7.4) ------------------------------------


def _reorder_eligible(
    brief: Mapping[str, Any], beat_sheet: tuple[Beat, ...]
) -> tuple[bool, str]:
    rl = brief.get("reorder_license")
    if not isinstance(rl, Mapping):
        return False, "no_reorder_license"
    try:
        rs = float(rl["reveal_start_s"])
        re = float(rl["reveal_end_s"])
    except (KeyError, TypeError, ValueError):
        return False, "reorder_license_malformed"
    if re <= rs:
        return False, "reorder_license_inverted"

    pre_skip = bool(rl.get("pre_reveal_skippable", False))
    if not pre_skip:
        return False, "pre_reveal_not_skippable"

    # Precondition 3: reveal + post-reveal context fits in clip.
    window = brief.get("window") or {}
    if not isinstance(window, Mapping):
        return False, "window_missing"
    try:
        w_start = float(window["start_s"])
        w_end = float(window["end_s"])
    except (KeyError, TypeError, ValueError):
        return False, "window_malformed"
    clip_dur = w_end - w_start
    if (re - rs) + (w_end - re) > clip_dur + 1e-6:
        return False, "reveal_plus_context_exceeds_clip"

    # Precondition 2: align to scene-cluster boundary OR word_end_timestamp.
    scenes = brief.get("scenes") or []
    boundary_match = False
    if isinstance(scenes, list):
        for s in scenes:
            if not isinstance(s, Mapping):
                continue
            try:
                ss = float(s["start_s"])
                se = float(s["end_s"])
            except (KeyError, TypeError, ValueError):
                continue
            if abs(rs - ss) < 0.25 or abs(re - se) < 0.25:
                boundary_match = True
                break
    if not boundary_match:
        words = brief.get("transcript_words") or []
        if isinstance(words, list):
            for w in words:
                if not isinstance(w, Mapping):
                    continue
                try:
                    we = float(w["end_s"])
                except (KeyError, TypeError, ValueError):
                    continue
                if abs(rs - we) < 0.15 or abs(re - we) < 0.15:
                    boundary_match = True
                    break
    if not boundary_match:
        return False, "reveal_not_aligned_to_boundary"

    return True, "all_preconditions_met"


# ---- planner -------------------------------------------------------


def plan_cuts_final(
    director_brief: Mapping[str, Any],
    beat_sheet: tuple[Beat, ...],
    profile: str = "medium",
    lock: StyleLock | None = None,
) -> CutPlan:
    """Translate analyzer cut candidates into editor decisions.

    Output preserves the analyzer's candidate order; every candidate
    receives one :class:`CutDecision`. ``apply=False`` decisions
    carry a closed-enum-friendly ``reason`` for the manifest.
    """
    cuts = director_brief.get("cuts") or []
    if not isinstance(cuts, list):
        return CutPlan()

    filler_min, dead_air_min = _resolve_thresholds(profile, lock)

    # Lock can lift the action threshold (action_sports).
    action_threshold = ACTION_LEVEL_ENTER
    if lock is not None:
        with contextlib.suppress(TypeError, ValueError):
            action_threshold = float(lock.action_level_threshold)

    silence_ranges = _emotional_silences(director_brief)
    action_ranges = _action_windows_source_time(
        director_brief, beat_sheet, action_threshold
    )
    reaction_ranges = _reaction_windows_source_time(
        director_brief, beat_sheet, profile
    )

    decisions: list[CutDecision] = []
    warnings: list[str] = []
    for i, c in enumerate(cuts):
        if not isinstance(c, Mapping):
            warnings.append(f"cut_{i}_skipped:not_object")
            continue
        try:
            s = float(c["start_s"])
            e = float(c["end_s"])
        except (KeyError, TypeError, ValueError):
            warnings.append(f"cut_{i}_skipped:bad_bounds")
            continue
        if e <= s:
            warnings.append(f"cut_{i}_skipped:inverted")
            continue

        kind_raw = c.get("kind", "filler")
        # §7.10 analyzer cut "kind" enum is filler|dead_air|repeat;
        # legacy editor uses "filler_word|silence_pause|dead_air".
        # Normalise here so editor v2.1 splice (Part 7) consumes a
        # consistent label.
        kind = str(kind_raw)
        if kind == "filler":
            kind_canon = "filler"
        elif kind in ("dead_air", "silence_pause"):
            kind_canon = "dead_air"
        elif kind == "repeat":
            kind_canon = "repeat"
        else:
            warnings.append(f"cut_{i}_skipped:unknown_kind:{kind}")
            continue

        cand_id = str(c.get("candidate_id") or f"cut_{i:03d}")
        confidence = float(c.get("confidence", 1.0) or 1.0)

        # 1. HARD: action continuity (§7.3.3).
        overlap = _overlaps_any(s, e, action_ranges)
        if overlap is not None:
            decisions.append(
                CutDecision(
                    candidate_id=cand_id,
                    start_s=s,
                    end_s=e,
                    kind=kind_canon,
                    apply=False,
                    reason="action_continuity",
                    confidence=confidence,
                )
            )
            continue

        # 2. HARD: emotional silence (§7.3.1).
        overlap = _overlaps_any(s, e, silence_ranges)
        if overlap is not None:
            decisions.append(
                CutDecision(
                    candidate_id=cand_id,
                    start_s=s,
                    end_s=e,
                    kind=kind_canon,
                    apply=False,
                    reason="emotional_signal_preserved",
                    confidence=confidence,
                )
            )
            continue

        # 3. HARD: reaction protection (§7.3.2).
        overlap = _overlaps_any(s, e, reaction_ranges)
        if overlap is not None:
            decisions.append(
                CutDecision(
                    candidate_id=cand_id,
                    start_s=s,
                    end_s=e,
                    kind=kind_canon,
                    apply=False,
                    reason="reaction_protection",
                    confidence=confidence,
                )
            )
            continue

        # 4. Profile / lock thresholds (§7.2).
        if kind_canon == "filler":
            if confidence < filler_min:
                decisions.append(
                    CutDecision(
                        candidate_id=cand_id,
                        start_s=s,
                        end_s=e,
                        kind=kind_canon,
                        apply=False,
                        reason=f"filler_confidence_below_threshold:{filler_min}",
                        confidence=confidence,
                    )
                )
                continue
            decisions.append(
                CutDecision(
                    candidate_id=cand_id,
                    start_s=s,
                    end_s=e,
                    kind=kind_canon,
                    apply=True,
                    reason="filler_word_apply",
                    confidence=confidence,
                )
            )
            continue
        if kind_canon == "dead_air":
            duration = e - s
            if duration < dead_air_min:
                decisions.append(
                    CutDecision(
                        candidate_id=cand_id,
                        start_s=s,
                        end_s=e,
                        kind=kind_canon,
                        apply=False,
                        reason=f"dead_air_below_threshold:{dead_air_min}s",
                        confidence=confidence,
                    )
                )
                continue
            decisions.append(
                CutDecision(
                    candidate_id=cand_id,
                    start_s=s,
                    end_s=e,
                    kind=kind_canon,
                    apply=True,
                    reason="dead_air_apply",
                    confidence=confidence,
                )
            )
            continue
        if kind_canon == "repeat":
            # Repeats are LLM-detected; require high confidence.
            if confidence < max(0.75, filler_min):
                decisions.append(
                    CutDecision(
                        candidate_id=cand_id,
                        start_s=s,
                        end_s=e,
                        kind=kind_canon,
                        apply=False,
                        reason="repeat_confidence_too_low",
                        confidence=confidence,
                    )
                )
                continue
            decisions.append(
                CutDecision(
                    candidate_id=cand_id,
                    start_s=s,
                    end_s=e,
                    kind=kind_canon,
                    apply=True,
                    reason="repeat_apply",
                    confidence=confidence,
                )
            )
            continue

    reorder_eligible, reorder_reason = _reorder_eligible(director_brief, beat_sheet)
    return CutPlan(
        decisions=tuple(decisions),
        reorder_eligible=reorder_eligible,
        reorder_reason=reorder_reason,
        warnings=tuple(warnings),
    )
