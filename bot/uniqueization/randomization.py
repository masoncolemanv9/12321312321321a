"""Per-job uniqueness randomization for the Editor Agent pipeline.

The user controls a single ``uniqueness_pct`` slider on the
"🎯 Уникальность" sub-screen of the Монтажёр addon. The slider has
five semantic anchors, all rooted in the **debate-vetted Parts 1-11
configuration** rather than in any value invented by this module:

* ``0`` — the editor runs exactly as it ran in Parts 1-11 (i.e. the
  debate's "best vetted result"). No variation is applied. Output is
  byte-equivalent to the pre-randomization baseline so every golden
  test continues to pass without changes.
* ``1 — 100`` (🟢 green) — per-job randomization stays **inside** the
  debate-vetted envelope defined by the three profile points
  ``light → medium → heavy``. The 3 profiles were approved by D1+D2
  across 5 debate rounds (``final_spec_FULL.md`` line 3) and their
  span is therefore the debate's allowed variation range.
* ``101 — 400`` (🔴 red) — per-job randomization extends up to **4×**
  the debate-vetted envelope. This is explicitly above what the
  debate approved; the UI marks it red.

Why this lives outside the planner modules
==========================================

The v6 sub-planners (``zoom_planner``, ``cut_planner``, …) are pure
deterministic functions — their golden tests pin every byte of the
produced ``edit_plan.json``. Routing per-job randomness through them
would require threading ``random.Random`` into every sub-planner and
re-baking every golden fixture.

Instead, we randomize a small set of **ffmpeg numerical parameters**
on the ``UniqConfig`` at the top of the worker, BEFORE the
deterministic planner pipeline runs. The planner sees a slightly
different ``cfg`` for each job; its own logic remains deterministic
given that input.

Debate-vetted ranges (single source of truth)
==============================================

The four randomized fields are the four numeric knobs that the three
debate-vetted profiles (``light`` / ``medium`` / ``heavy``,
``bot/uniqueization/profiles.py``) explicitly carry. The light value
and the heavy value bookend the debate-allowed envelope; the value
the worker sees after ``apply_profile`` is the per-job center.

::

    field                  light → heavy (debate envelope)
    ─────────────────────  ───────────────────────────────
    zoom                   0.00 → 0.45
    mirror_duration_s      1.50 → 2.00
    effects_opacity        0.10 → 0.20
    audio_fx_wet           0.05 → 0.15

Per-job formula
===============

For a slider value ``X`` (0 ≤ X ≤ ``MAX_UNIQUENESS_PCT``):

1. ``v = uniform(0.6 * X, X)`` — the actual variation % drawn for
   this job. This matches the user's "slider 5 → random in [3, 5]"
   example.
2. For each field, ``width = (hi - lo) * v / DEBATE_PCT`` — at
   ``v = DEBATE_PCT (= 100)`` the spread equals one full debate
   envelope; at ``v = 4 * DEBATE_PCT`` the spread is 4× the envelope.
3. ``offset = uniform(-width/2, +width/2)``.
4. ``value = clamp(center + offset, physical_lo, physical_hi)`` —
   the physical bounds are the hard ffmpeg limits beyond which the
   parameter would corrupt output, NOT the debate envelope; the
   debate envelope is the *recommended* range, the physical bound is
   the *safety* range.

The center is read from the post-``apply_profile`` ``UniqConfig`` so
the variation is always measured around whichever profile the
operator selected.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import UniqConfig

__all__ = (
    "DEBATE_PCT",
    "DEBATE_RANGES",
    "DebateRange",
    "MAX_UNIQUENESS_PCT",
    "RandomizationReport",
    "SLIDER_STEPS",
    "clamp_pct",
    "describe_zone",
    "jitter_config",
    "jitter_value",
    "make_rng",
    "make_seed",
)


# ``DEBATE_PCT`` is the slider value that maps to "exactly the debate
# envelope". Above it (up to ``MAX_UNIQUENESS_PCT``) the variation
# overshoots the debate. Both constants are integers so the UI can
# render them with a plain f-string.
DEBATE_PCT: int = 100

# The slider's hard maximum: 4 × ``DEBATE_PCT``. Per the user's
# specification — "максимум в 4 раза больше значений дебатёров". Any
# value above is clamped.
MAX_UNIQUENESS_PCT: int = 4 * DEBATE_PCT

# Buttons rendered on the slider sub-screen. 0 → off (default best),
# the green band is dense in the low single digits because the
# variation there is most useful for subtle per-video drift; the red
# band is sparse because each step there is already a large jump.
SLIDER_STEPS: tuple[int, ...] = (
    0,
    5,
    10,
    15,
    20,
    25,
    50,
    75,
    100,
    150,
    200,
    250,
    300,
    350,
    400,
)


@dataclass(frozen=True, slots=True)
class DebateRange:
    """One debate-vetted numeric parameter envelope.

    ``field`` MUST match a field name on :class:`UniqConfig` and
    therefore also on :class:`UniqProfile`. ``debate_lo`` /
    ``debate_hi`` are the bounds of the debate-vetted envelope (the
    span across the three profiles). ``physical_lo`` / ``physical_hi``
    are the hard ffmpeg/codec safety limits — at slider values in the
    red band the variation may exceed the debate envelope but it
    must never escape the physical bounds.

    ``debate_ref`` is the source citation: a section / line into
    ``final_spec_FULL.md`` or ``bot/uniqueization/profiles.py``.
    """

    field: str
    debate_lo: float
    debate_hi: float
    physical_lo: float
    physical_hi: float
    debate_ref: str

    @property
    def debate_width(self) -> float:
        """Width of the debate-vetted envelope."""
        return self.debate_hi - self.debate_lo


# The four debate-vetted ranges. EACH bound traces back to a line of
# code that pre-dates Part 12; nothing here is extrapolated. The
# ``light`` / ``heavy`` profile values are the bookends, ``medium``
# sits in the middle.
DEBATE_RANGES: tuple[DebateRange, ...] = (
    DebateRange(
        field="zoom",
        debate_lo=0.00,
        debate_hi=0.45,
        physical_lo=0.0,
        physical_hi=0.70,
        debate_ref=(
            "profiles.py PROFILE_DEFAULTS.zoom (light=0.00, medium=0.35, "
            "heavy=0.45); final_spec_FULL.md §6.2 zoom-reframe-agent."
        ),
    ),
    DebateRange(
        field="mirror_duration_s",
        debate_lo=1.50,
        debate_hi=2.00,
        physical_lo=0.5,
        physical_hi=3.0,
        debate_ref=(
            "profiles.py PROFILE_DEFAULTS.mirror_duration_s "
            "(light=1.5, medium=1.5, heavy=2.0); final_spec_FULL.md "
            "§7.6.5 v2.0 fallback 1.5s middle window."
        ),
    ),
    DebateRange(
        field="effects_opacity",
        debate_lo=0.10,
        debate_hi=0.20,
        physical_lo=0.0,
        physical_hi=0.40,
        debate_ref=(
            "profiles.py PROFILE_DEFAULTS.effects_opacity "
            "(light=0.10, medium=0.15, heavy=0.20); "
            "docs/DEBATE_TOPICS/editor-agent-v2.md §2 line 35."
        ),
    ),
    DebateRange(
        field="audio_fx_wet",
        debate_lo=0.05,
        debate_hi=0.15,
        physical_lo=0.0,
        physical_hi=0.40,
        debate_ref=(
            "profiles.py PROFILE_DEFAULTS.audio_fx_wet "
            "(light=0.05, medium=0.10, heavy=0.15); "
            "final_spec_FULL.md §8.9 wet defaults."
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class RandomizationReport:
    """What the worker writes to ``manifest.extra.uniqueness_randomization``.

    Carrying ``seed`` + ``picks`` is what lets us reproduce a given
    job's variation later (debug, A/B, audit).
    """

    seed: int
    uniqueness_pct: int
    variation_pct: float  # actual per-video v drawn from uniform(0.6X, X)
    zone_label: str
    picks: Mapping[str, Mapping[str, Any]]


def clamp_pct(value: object) -> int:
    """Coerce any input to the slider's allowed integer range.

    Used by both the env-var loader and the addon callback so they
    never agree on the wrong cap.
    """
    try:
        as_int = int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0
    return max(0, min(MAX_UNIQUENESS_PCT, as_int))


def describe_zone(pct: int) -> str:
    """Return a short Russian label describing the slider position.

    Zones are derived from the debate envelope itself: anything inside
    the envelope is green; anything above 1× envelope is red.
    """
    pct = clamp_pct(pct)
    if pct == 0:
        return "детерминированный (без вариации)"
    if pct <= DEBATE_PCT:
        return f"🟢 внутри debate-envelope (≤{DEBATE_PCT}%)"
    return (
        f"🔴 выше debate-envelope (×{pct / DEBATE_PCT:.1f}, "
        f"максимум ×{MAX_UNIQUENESS_PCT / DEBATE_PCT:.0f})"
    )


def make_seed(
    *,
    job_id: object = None,
    upload_ts: object = None,
    clip_index: int = 0,
    salt: str = "editor-uniqueness-v1",
) -> int:
    """Derive a 64-bit deterministic seed from per-job inputs.

    Same ``(job_id, upload_ts, clip_index)`` → same seed → reproducible
    variation. Different uploads → different seeds → independent
    variations.
    """
    payload = f"{salt}|{job_id}|{upload_ts}|{clip_index}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def make_rng(
    *,
    job_id: object = None,
    upload_ts: object = None,
    clip_index: int = 0,
    salt: str = "editor-uniqueness-v1",
) -> random.Random:
    """Convenience: build a seeded ``random.Random``."""
    return random.Random(
        make_seed(
            job_id=job_id,
            upload_ts=upload_ts,
            clip_index=clip_index,
            salt=salt,
        )
    )


def jitter_value(
    spec: DebateRange,
    center: float,
    rng: random.Random,
    variation_pct: float,
) -> float:
    """Pick a per-video value for one parameter.

    ``center`` is the post-``apply_profile`` value of the field on the
    operator's selected profile (NOT recomputed by this module — the
    profile choice remains the operator's, the slider only adds drift
    around it).

    Formula::

        spread = debate_width * (variation_pct / DEBATE_PCT)
        offset = uniform(-spread/2, +spread/2)
        value  = clamp(center + offset, physical_lo, physical_hi)

    At ``variation_pct = 0`` ⇒ returns ``center`` unchanged.
    At ``variation_pct = DEBATE_PCT`` (100) ⇒ ±½ debate width drift.
    At ``variation_pct = MAX_UNIQUENESS_PCT`` (400) ⇒ ±2 debate widths.
    """
    if variation_pct <= 0.0:
        return round(center, 6)
    spread = spec.debate_width * (variation_pct / DEBATE_PCT)
    offset = rng.uniform(-spread / 2.0, spread / 2.0)
    value = center + offset
    clamped = max(spec.physical_lo, min(spec.physical_hi, value))
    return round(clamped, 6)


def _draw_variation_pct(rng: random.Random, slider_pct: int) -> float:
    """Per-video draw matching the user's "slider 5 → [3, 5]" rule.

    Returns ``uniform(0.6 * slider_pct, slider_pct)`` so the actual
    variation amount is itself random within a 40 %-wide band below
    the slider value. This means consecutive jobs at the same slider
    setting won't always burn the full budget — the budget itself is
    randomized.
    """
    if slider_pct <= 0:
        return 0.0
    return rng.uniform(0.6 * slider_pct, float(slider_pct))


def jitter_config(
    cfg: UniqConfig,
    *,
    uniqueness_pct: int,
    rng: random.Random | None = None,
    seed: int | None = None,
) -> tuple[UniqConfig, RandomizationReport]:
    """Return a per-job jittered :class:`UniqConfig` + audit report.

    At ``uniqueness_pct == 0`` returns the input ``cfg`` unchanged
    (same object reference) — this guarantees pre-Part-12 byte
    equivalence for every job that does not opt into the slider.

    Raises ``ValueError`` if ``uniqueness_pct > 0`` and neither
    ``rng`` nor ``seed`` is supplied.
    """
    pct = clamp_pct(uniqueness_pct)
    if pct == 0:
        return cfg, RandomizationReport(
            seed=seed if seed is not None else 0,
            uniqueness_pct=0,
            variation_pct=0.0,
            zone_label=describe_zone(0),
            picks={},
        )

    if rng is None:
        if seed is None:
            raise ValueError(
                "jitter_config requires either rng or seed when "
                "uniqueness_pct > 0"
            )
        rng = random.Random(seed)

    variation_pct = _draw_variation_pct(rng, pct)

    origin = dict(cfg._origin)
    new_values: dict[str, Any] = {}
    picks: dict[str, dict[str, Any]] = {}

    for spec in DEBATE_RANGES:
        center = float(getattr(cfg, spec.field))
        value = jitter_value(spec, center, rng, variation_pct)
        new_values[spec.field] = value
        picks[spec.field] = {
            "value": value,
            "center": center,
            "debate_lo": spec.debate_lo,
            "debate_hi": spec.debate_hi,
            "physical_lo": spec.physical_lo,
            "physical_hi": spec.physical_hi,
            "debate_ref": spec.debate_ref,
        }
        origin[spec.field] = f"uniqueness:{pct}"

    jittered = replace(cfg, **new_values, _origin=origin)
    return jittered, RandomizationReport(
        seed=seed if seed is not None else 0,
        uniqueness_pct=pct,
        variation_pct=round(variation_pct, 6),
        zone_label=describe_zone(pct),
        picks=picks,
    )
