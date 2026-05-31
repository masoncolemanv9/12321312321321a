"""Subtitle planner (final_spec §8.1, §8.4).

Per-beat subtitle decisions:

* ``style_id`` — closed enum
  {``off`` | ``minimal`` | ``soft_emphasis`` | ``karaoke_emphasis``}
  picked from the §8.1 purpose × emotional-intent table. Profile
  light demotes ``karaoke_emphasis`` → ``soft_emphasis``; profile
  heavy promotes ``soft_emphasis`` → ``karaoke_emphasis``.
  ``minimal`` / ``off`` never change.
* ``emphasis_words`` — subset of analyzer ``emphasis_candidates[]``
  per §8.4 two-stage selection. Per-beat cap N=1/2/3 by duration
  buckets <3s / 3-6s / >6s. Order preserved.
* ``y_position_candidates`` — profile preferred + alt list per
  §8.2. The renderer (Part 5 ``subtitle_renderer.py``) resolves
  final position at render-time using face/logo bboxes.

Pure function. No I/O. **NO LLM** anywhere — selection runs
verbatim on analyzer-shipped candidates (§12.2: emphasis_words
LLM-allowed in analyzer, not in planner).

Style lock (§12.6) ``subtitle_style`` table overrides the §8.1
mapping per-purpose. Lock cannot escalate ``off`` to anything
visible (§8.1 last paragraph).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .beat_sheet import Beat
from .locks import StyleLock

__all__ = (
    "SUBTITLE_STYLE_ENUM",
    "SEMANTIC_ROLE_ENUM",
    "Y_POSITION_BY_PROFILE",
    "Y_FALLBACK_POSITIONS",
    "EmphasisWord",
    "SubtitleEvent",
    "SubtitlePlan",
    "plan_subtitle",
)


# §8.1 closed enum.
SUBTITLE_STYLE_ENUM: tuple[str, ...] = (
    "off",
    "minimal",
    "soft_emphasis",
    "karaoke_emphasis",
)

# §8.4 closed enum of analyzer semantic roles.
SEMANTIC_ROLE_ENUM: tuple[str, ...] = (
    "revelation",
    "emotion_peak",
    "punchline",
    "key_subject",
    "key_verb",
)


# §8.2 preferred Y by profile (normalised 0..1, top→bottom).
Y_POSITION_BY_PROFILE: Mapping[str, float] = {
    "light": 0.78,
    "medium": 0.80,
    "heavy": 0.82,
}

# §8.2 fallback chain (descending preference).
Y_FALLBACK_POSITIONS: tuple[float, ...] = (0.85, 0.72, 0.65)


# §8.1 base mapping by purpose. Emotional-intent qualifiers refine
# at runtime.
_BASE_STYLE_BY_PURPOSE: Mapping[str, str] = {
    "hook": "karaoke_emphasis",
    "reveal": "karaoke_emphasis",
    "tension": "soft_emphasis",
    "emotional_hold": "minimal",
    "action": "minimal",
    "dialogue": "karaoke_emphasis",
    "reaction": "minimal",
    "transition": "off",
    "resolution": "soft_emphasis",
}


# §8.4 purpose × semantic-role preferences. Beat picks the FIRST
# candidate whose role is in this set; if none, falls back to
# salience-ordered selection.
_ROLE_PREFERENCE_BY_PURPOSE: Mapping[str, tuple[str, ...]] = {
    "hook": ("revelation",),
    "reveal": ("revelation",),
    "tension": ("punchline", "key_verb"),
    "dialogue": ("punchline", "key_verb"),
    "emotional_hold": ("emotion_peak",),
    "reaction": ("emotion_peak",),
    "action": ("key_verb",),
    "transition": (),
    "resolution": ("punchline",),
}


@dataclass(frozen=True, slots=True)
class EmphasisWord:
    """One §8.4 selection. Inputs are analyzer-supplied."""

    word: str
    start_s: float
    end_s: float
    salience: float
    semantic_role: str


@dataclass(frozen=True, slots=True)
class SubtitleEvent:
    """One §8.1 + §8.4 per-beat subtitle decision."""

    beat_id: str
    start_s: float
    end_s: float
    style_id: str
    emphasis_words: tuple[EmphasisWord, ...]
    y_position_preferred: float
    y_position_fallbacks: tuple[float, ...]
    reason: str = ""


@dataclass(frozen=True, slots=True)
class SubtitlePlan:
    """Container for :func:`plan_subtitle` output."""

    events: tuple[SubtitleEvent, ...] = field(default_factory=tuple)
    profile: str = "medium"
    lock_id: str | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)


# ---- style selection -----------------------------------------------


def _emotional_intent_override(
    purpose: str, emotional_intent: str, base: str
) -> str:
    """Apply §8.1 emotional-intent qualifiers on top of the purpose default."""
    intent = (emotional_intent or "").lower()
    if purpose == "emotional_hold" and intent in ("sadness", "crying", "grief"):
        # row: emotional_hold + sadness/crying → minimal or off (we pick off)
        return "off"
    if purpose == "reveal" and intent == "shock":
        return "karaoke_emphasis"
    if purpose == "hook" and intent in ("shock", "reveal"):
        return "karaoke_emphasis"
    if purpose == "reaction" and intent == "shock":
        return "minimal"
    if purpose == "resolution" and intent in ("calm", "warm"):
        return "soft_emphasis"
    return base


def _profile_adjust(style_id: str, profile: str) -> str:
    """§8.1 last paragraph — profile demotes / promotes."""
    if style_id in ("off", "minimal"):
        return style_id
    if profile == "light" and style_id == "karaoke_emphasis":
        return "soft_emphasis"
    if profile == "heavy" and style_id == "soft_emphasis":
        return "karaoke_emphasis"
    return style_id


def _lock_override(
    purpose: str, current: str, lock: StyleLock | None
) -> str:
    """Apply §12.6 lock's ``subtitle_style`` table. Lock cannot
    escalate ``off`` to anything visible."""
    if lock is None:
        return current
    style_map = lock.subtitle_style
    if not isinstance(style_map, Mapping):
        return current
    override = style_map.get(purpose)
    if not isinstance(override, str) or override not in SUBTITLE_STYLE_ENUM:
        return current
    # §8.1 — preserve hard ``off``.
    if current == "off" and override != "off":
        return current
    return override


def _resolve_style(
    purpose: str,
    emotional_intent: str,
    *,
    profile: str,
    lock: StyleLock | None,
) -> str:
    base = _BASE_STYLE_BY_PURPOSE.get(purpose, "soft_emphasis")
    intent_adjusted = _emotional_intent_override(
        purpose, emotional_intent, base
    )
    profile_adjusted = _profile_adjust(intent_adjusted, profile)
    return _lock_override(purpose, profile_adjusted, lock)


# ---- emphasis-word selection ---------------------------------------


def _per_beat_cap(duration_s: float) -> int:
    if duration_s < 3.0:
        return 1
    if duration_s <= 6.0:
        return 2
    return 3


def _is_candidate_in_beat(
    cand: Mapping[str, Any], beat: Beat, *, window_start_s: float
) -> bool:
    """Candidate timestamps are source-time; beats are render-time."""
    try:
        c_start = float(cand["start_s"]) - window_start_s
        c_end = float(cand["end_s"]) - window_start_s
    except (KeyError, TypeError, ValueError):
        return False
    if c_end <= c_start:
        return False
    return not (c_end <= beat.start_s or c_start >= beat.end_s)


def _coerce_candidate(
    cand: Mapping[str, Any], window_start_s: float
) -> EmphasisWord | None:
    try:
        word = str(cand["word"]).strip()
        s = float(cand["start_s"]) - window_start_s
        e = float(cand["end_s"]) - window_start_s
        salience = float(cand.get("salience", 1.0))
        role = str(cand.get("semantic_role", ""))
    except (KeyError, TypeError, ValueError):
        return None
    if not word or e <= s:
        return None
    if role and role not in SEMANTIC_ROLE_ENUM:
        # Unknown role — keep word, blank role for downstream safety.
        role = ""
    return EmphasisWord(
        word=word,
        start_s=s,
        end_s=e,
        salience=salience,
        semantic_role=role,
    )


def _select_emphasis(
    beat: Beat,
    candidates: tuple[EmphasisWord, ...],
    *,
    cap: int,
) -> tuple[EmphasisWord, ...]:
    """§8.4 two-stage selection within a single beat."""
    if not candidates or cap <= 0:
        return ()
    role_prefs = _ROLE_PREFERENCE_BY_PURPOSE.get(beat.purpose, ())
    # Stage 1 — role-preferred. Preserve analyzer order.
    preferred = tuple(
        c for c in candidates if c.semantic_role in role_prefs
    )
    if len(preferred) >= cap:
        # Within the preferred set, pick highest-salience first;
        # ties broken by source-time order (stable sort).
        ranked = sorted(
            enumerate(preferred), key=lambda kv: (-kv[1].salience, kv[0])
        )
        return tuple(c for _, c in ranked[:cap])
    # Stage 2 — fall back to global salience ordering for remaining slots.
    chosen = list(preferred)
    chosen_words = {(c.word, round(c.start_s, 3)) for c in chosen}
    remaining = sorted(
        (
            (i, c)
            for i, c in enumerate(candidates)
            if (c.word, round(c.start_s, 3)) not in chosen_words
        ),
        key=lambda kv: (-kv[1].salience, kv[0]),
    )
    for _, c in remaining:
        if len(chosen) >= cap:
            break
        chosen.append(c)
    return tuple(chosen)


# ---- planner -------------------------------------------------------


def plan_subtitle(
    beat_sheet: tuple[Beat, ...],
    director_brief: Mapping[str, Any],
    profile: str = "medium",
    lock: StyleLock | None = None,
) -> SubtitlePlan:
    """Per-beat subtitle decisions.

    Each beat receives one :class:`SubtitleEvent` (or is skipped when
    ``style_id`` resolves to ``off``). Emphasis words are picked from
    analyzer ``emphasis_candidates[]`` honouring §8.4 caps and the
    beat's purpose × semantic-role preferences.
    """
    if not beat_sheet:
        return SubtitlePlan(
            profile=profile,
            lock_id=lock.lock_id if lock is not None else None,
        )

    window = director_brief.get("window") or {}
    if isinstance(window, Mapping):
        try:
            window_start_s = float(window.get("start_s", 0.0))
        except (TypeError, ValueError):
            window_start_s = 0.0
    else:
        window_start_s = 0.0

    # Normalise + coerce analyzer candidates once.
    raw_candidates = director_brief.get("emphasis_candidates") or []
    if not isinstance(raw_candidates, list):
        raw_candidates = []
    all_candidates: list[EmphasisWord] = []
    warnings: list[str] = []
    for i, c in enumerate(raw_candidates):
        if not isinstance(c, Mapping):
            warnings.append(f"emphasis_candidate_{i}_skipped:not_object")
            continue
        ew = _coerce_candidate(c, window_start_s)
        if ew is None:
            warnings.append(f"emphasis_candidate_{i}_skipped:bad_fields")
            continue
        all_candidates.append(ew)

    y_pref = Y_POSITION_BY_PROFILE.get(
        profile, Y_POSITION_BY_PROFILE["medium"]
    )

    events: list[SubtitleEvent] = []
    for beat in beat_sheet:
        style_id = _resolve_style(
            beat.purpose,
            beat.emotional_intent,
            profile=profile,
            lock=lock,
        )
        if style_id == "off":
            events.append(
                SubtitleEvent(
                    beat_id=beat.beat_id,
                    start_s=beat.start_s,
                    end_s=beat.end_s,
                    style_id="off",
                    emphasis_words=(),
                    y_position_preferred=y_pref,
                    y_position_fallbacks=Y_FALLBACK_POSITIONS,
                    reason=f"purpose={beat.purpose}_style=off",
                )
            )
            continue
        # Candidates within this beat (render-time bounds).
        in_beat = tuple(
            c
            for c in all_candidates
            if not (c.end_s <= beat.start_s or c.start_s >= beat.end_s)
        )
        cap = _per_beat_cap(beat.duration_s)
        emphasis = _select_emphasis(beat, in_beat, cap=cap)
        # ``minimal`` deliberately strips emphasis (§8.1).
        if style_id == "minimal":
            emphasis = ()
        reason_bits = [
            f"purpose={beat.purpose}",
            f"style={style_id}",
            f"cap={cap}",
        ]
        if beat.emotional_intent:
            reason_bits.append(f"intent={beat.emotional_intent}")
        events.append(
            SubtitleEvent(
                beat_id=beat.beat_id,
                start_s=beat.start_s,
                end_s=beat.end_s,
                style_id=style_id,
                emphasis_words=emphasis,
                y_position_preferred=y_pref,
                y_position_fallbacks=Y_FALLBACK_POSITIONS,
                reason="; ".join(reason_bits),
            )
        )

    return SubtitlePlan(
        events=tuple(events),
        profile=profile,
        lock_id=lock.lock_id if lock is not None else None,
        warnings=tuple(warnings),
    )
