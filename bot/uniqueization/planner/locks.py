"""Built-in creative style locks (final_spec_FULL §12.6).

A *style lock* is a frozen bundle of cross-channel overrides keyed by
``lock_id``. v6.0 ships three built-ins (``cinematic_drama``,
``fast_dialogue``, ``action_sports``) per §12.6 table; user-defined
locks are deferred to v6.x. The locks here are *pure data* — no I/O,
no side effects — so they remain importable from any process boundary
(planner, validator, tests).

Application order (§12.6 last paragraph):

* base profile (light / medium / heavy) defaults applied first;
* style lock overrides applied **on top** of the profile;
* per-job payload overrides applied **after** the lock.

This module exposes only frozen dataclasses + a lookup table. The
planners (beat / zoom / cut / mirror) consume the lock by reading
its fields, never by mutating it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

__all__ = (
    "ForbiddenEffect",
    "StyleLock",
    "STYLE_LOCKS",
    "get_lock",
    "BUILT_IN_LOCK_IDS",
)


# Closed enum of effects a lock can forbid. Adding a new value is a
# v6.x spec amendment (§12.6) — keep this in lock-step with the
# planners that read ``forbidden_effects``.
ForbiddenEffect = str  # "cross_fade_short" | "match_cut_zoom" | ...


@dataclass(frozen=True, slots=True)
class StyleLock:
    """One row of §12.6 lock table.

    Field semantics:

    * ``subtitle_style``: subtitle intensity floor / ceiling per
      beat purpose. The planner clamps its computed intensity
      against this band before emitting the cue.
    * ``zoom_grammar_overrides``: caps & deltas applied on top of
      §6.2 profile caps. ``max_zoom_delta`` MAY be negative (lock
      reduces aggression). Motion-sickness §6.4 caps remain HARD
      even under lock — the lock can only *tighten* them.
    * ``color_intent_mapping_overrides``: remaps a default
      ``color_intent`` for a given beat purpose. ``{}`` = use
      planner default.
    * ``hook_treatment_overrides``: knobs for the hook beat —
      ``zoom_delta`` (capped within profile + motion-sickness),
      ``brightness_pulse_delta``, ``subtitle_intensity``.
    * ``cut_threshold_overrides``: filler / dead-air thresholds
      from §7.2 that the lock tightens / loosens. ``None`` keeps
      the profile default.
    * ``action_level_threshold``: lock-specific override for the
      §6.2 action override threshold (default 0.6, hysteresis
      0.55→0.6).
    * ``transition_ms_multiplier``: scalar applied to every
      ``transition_ms`` the zoom planner emits. ``1.0`` = no
      change. Lock cannot push transition_ms below 150ms (§6.1
      default) — clamped by the zoom planner.
    * ``forbidden_effects``: closed set of effects the planner
      refuses to emit while the lock is active.
    """

    lock_id: str
    subtitle_style: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({})
    )
    zoom_grammar_overrides: Mapping[str, float] = field(
        default_factory=lambda: MappingProxyType({})
    )
    color_intent_mapping_overrides: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({})
    )
    hook_treatment_overrides: Mapping[str, object] = field(
        default_factory=lambda: MappingProxyType({})
    )
    cut_threshold_overrides: Mapping[str, float] = field(
        default_factory=lambda: MappingProxyType({})
    )
    action_level_threshold: float = 0.6
    transition_ms_multiplier: float = 1.0
    forbidden_effects: tuple[ForbiddenEffect, ...] = ()


# ---- built-in lock definitions (§12.6 table) -----------------------


_CINEMATIC_DRAMA = StyleLock(
    lock_id="cinematic_drama",
    # Emotional intensity, slow pacing, deep color, minimal cuts.
    # Keep subtitle restraint — never overpower emotional beats.
    subtitle_style=MappingProxyType(
        {
            "hook": "soft_emphasis",
            "reveal": "karaoke_emphasis",
            "tension": "soft_emphasis",
            "emotional_hold": "minimal",
            "action": "off",  # rarely action beats; if any, suppress text
            "dialogue": "soft_emphasis",
            "reaction": "minimal",
            "transition": "off",
            "resolution": "soft_emphasis",
        }
    ),
    # Slow zooms (reduce velocity); transitions doubled per spec.
    zoom_grammar_overrides=MappingProxyType(
        {
            "max_zoom_delta": -0.05,
            "transition_ms_floor": 250.0,  # raise default 150 -> 250
        }
    ),
    color_intent_mapping_overrides=MappingProxyType(
        {
            "hook": "warm_emotional",
            "reveal": "high_contrast_punch",
            "emotional_hold": "warm_emotional",
            "resolution": "warm_emotional",
        }
    ),
    hook_treatment_overrides=MappingProxyType(
        {
            "zoom_delta": 0.04,
            "brightness_pulse_delta": 0.04,
            "subtitle_intensity": "soft_emphasis",
        }
    ),
    cut_threshold_overrides=MappingProxyType(
        {
            "filler_confidence_min": 0.85,
            "dead_air_threshold_s": 1.0,  # rarely cut silence — drama lives on pauses
        }
    ),
    action_level_threshold=0.7,  # tougher to enter action override
    transition_ms_multiplier=2.0,  # §12.6 test rule: doubles transition_ms
    forbidden_effects=("dip_to_black_mid_clip",),
)


_FAST_DIALOGUE = StyleLock(
    lock_id="fast_dialogue",
    # Tight pacing, karaoke always, push-in on punchlines.
    subtitle_style=MappingProxyType(
        {
            "hook": "karaoke_emphasis",
            "reveal": "karaoke_emphasis",
            "tension": "karaoke_emphasis",
            "emotional_hold": "soft_emphasis",
            "action": "minimal",
            "dialogue": "karaoke_emphasis",
            "reaction": "karaoke_emphasis",
            "transition": "off",
            "resolution": "karaoke_emphasis",
        }
    ),
    zoom_grammar_overrides=MappingProxyType(
        {
            "max_zoom_delta": 0.0,
            "transition_ms_floor": 150.0,
        }
    ),
    color_intent_mapping_overrides=MappingProxyType(
        {
            "dialogue": "neutral_bright",
            "hook": "high_contrast_punch",
        }
    ),
    hook_treatment_overrides=MappingProxyType(
        {
            "zoom_delta": 0.05,
            "brightness_pulse_delta": 0.05,
            "subtitle_intensity": "karaoke_emphasis",
        }
    ),
    cut_threshold_overrides=MappingProxyType(
        {
            # §12.6 test rule: tightens cut thresholds.
            "filler_confidence_min": 0.55,
            "dead_air_threshold_s": 0.3,
        }
    ),
    action_level_threshold=0.6,
    transition_ms_multiplier=1.0,
    forbidden_effects=("cross_fade_short",),
)


_ACTION_SPORTS = StyleLock(
    lock_id="action_sports",
    # Wide action, fast cuts, bright color, minimal subtitle.
    subtitle_style=MappingProxyType(
        {
            "hook": "minimal",
            "reveal": "soft_emphasis",
            "tension": "minimal",
            "emotional_hold": "minimal",
            "action": "off",
            "dialogue": "soft_emphasis",
            "reaction": "minimal",
            "transition": "off",
            "resolution": "soft_emphasis",
        }
    ),
    zoom_grammar_overrides=MappingProxyType(
        {
            "max_zoom_delta": 0.0,
            "transition_ms_floor": 150.0,
        }
    ),
    color_intent_mapping_overrides=MappingProxyType(
        {
            "action": "high_contrast_punch",
            "hook": "high_contrast_punch",
            "reveal": "high_contrast_punch",
        }
    ),
    hook_treatment_overrides=MappingProxyType(
        {
            "zoom_delta": 0.03,
            "brightness_pulse_delta": 0.05,
            "subtitle_intensity": "minimal",
        }
    ),
    cut_threshold_overrides=MappingProxyType(
        {
            "filler_confidence_min": 0.65,
            "dead_air_threshold_s": 0.4,
        }
    ),
    # §12.6 test rule: raises action_level threshold.
    action_level_threshold=0.5,
    transition_ms_multiplier=1.0,
    forbidden_effects=("dip_to_black_mid_clip", "cross_fade_short"),
)


STYLE_LOCKS: Mapping[str, StyleLock] = MappingProxyType(
    {
        _CINEMATIC_DRAMA.lock_id: _CINEMATIC_DRAMA,
        _FAST_DIALOGUE.lock_id: _FAST_DIALOGUE,
        _ACTION_SPORTS.lock_id: _ACTION_SPORTS,
    }
)


BUILT_IN_LOCK_IDS: tuple[str, ...] = tuple(STYLE_LOCKS.keys())


def get_lock(lock_id: str | None) -> StyleLock | None:
    """Return the built-in lock for ``lock_id`` or ``None``.

    ``None`` / unknown ids return ``None`` — planners interpret this
    as "no lock, use profile defaults". This is intentionally
    non-fatal: validation (§12.4) is the place to reject an unknown
    ``style_lock_id``; planners must stay total functions so the
    fallback path (§12.5 ``creative_planner_status: degraded``) can
    still produce a plan.
    """
    if lock_id is None:
        return None
    return STYLE_LOCKS.get(lock_id)
