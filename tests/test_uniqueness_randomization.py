"""Tests for the per-job uniqueness randomization (Part 12 redesigned).

Covers ``bot.uniqueization.randomization`` and its integration with
``bot.uniqueization.config.UniqConfig`` + the editor_agent addon UI.

The module is intentionally narrow: four ffmpeg knobs varied inside
the debate-vetted envelope spanned by the three D1+D2-approved
profiles (``light → medium → heavy``). These tests pin both the math
(clamping, zone labels, seed reproducibility, the "slider=5 →
random ∈ [3, 5]" formula) and the no-op guarantee at
``uniqueness_pct=0`` that lets every pre-existing golden test keep
passing without changes.
"""

from __future__ import annotations

import random

import pytest

from bot.uniqueization.config import UniqConfig, apply_profile, load_config
from bot.uniqueization.profiles import resolve_profile
from bot.uniqueization.randomization import (
    DEBATE_PCT,
    DEBATE_RANGES,
    MAX_UNIQUENESS_PCT,
    SLIDER_STEPS,
    RandomizationReport,
    clamp_pct,
    describe_zone,
    jitter_config,
    jitter_value,
    make_rng,
    make_seed,
)


def _medium_cfg() -> UniqConfig:
    """Centered config — every randomized field has a non-trivial center."""
    cfg = load_config()
    return apply_profile(cfg, resolve_profile("medium"))


# ---- clamp_pct ----------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        (0, 0),
        (1, 1),
        (DEBATE_PCT, DEBATE_PCT),
        (MAX_UNIQUENESS_PCT, MAX_UNIQUENESS_PCT),
        (MAX_UNIQUENESS_PCT + 1, MAX_UNIQUENESS_PCT),
        (1000, MAX_UNIQUENESS_PCT),
        (-1, 0),
        (-100, 0),
        ("30", 30),
        ("700", MAX_UNIQUENESS_PCT),
        ("abc", 0),
        (None, 0),
        (29.9, 29),
    ],
)
def test_clamp_pct(raw: object, expected: int) -> None:
    assert clamp_pct(raw) == expected  # type: ignore[arg-type]


# ---- describe_zone ------------------------------------------------


def test_describe_zone_0_label() -> None:
    assert "детерминированный" in describe_zone(0)


@pytest.mark.parametrize("pct", [1, 5, 25, 50, 99, DEBATE_PCT])
def test_describe_zone_green_band(pct: int) -> None:
    assert "🟢" in describe_zone(pct)


@pytest.mark.parametrize("pct", [DEBATE_PCT + 1, 150, 250, MAX_UNIQUENESS_PCT])
def test_describe_zone_red_band(pct: int) -> None:
    assert "🔴" in describe_zone(pct)


def test_describe_zone_above_max_is_clamped() -> None:
    # Anything above MAX_UNIQUENESS_PCT still produces a red label.
    assert "🔴" in describe_zone(MAX_UNIQUENESS_PCT * 10)


# ---- make_seed / make_rng -----------------------------------------


def test_make_seed_is_deterministic() -> None:
    s1 = make_seed(job_id=42, clip_index=0)
    s2 = make_seed(job_id=42, clip_index=0)
    assert s1 == s2


def test_make_seed_distinguishes_job_id() -> None:
    seeds = {make_seed(job_id=i, clip_index=0) for i in range(100)}
    # Effectively all unique — 2^64 collision space, 100 samples.
    assert len(seeds) == 100


def test_make_seed_distinguishes_clip_index() -> None:
    a = make_seed(job_id=42, clip_index=0)
    b = make_seed(job_id=42, clip_index=1)
    assert a != b


def test_make_seed_distinguishes_upload_ts() -> None:
    a = make_seed(job_id=42, upload_ts=None, clip_index=0)
    b = make_seed(job_id=42, upload_ts="2025-01-01T00:00:00Z", clip_index=0)
    assert a != b


def test_make_rng_is_reproducible() -> None:
    r1 = make_rng(job_id=42, clip_index=0)
    r2 = make_rng(job_id=42, clip_index=0)
    for _ in range(10):
        assert r1.random() == r2.random()


# ---- DEBATE_RANGES integrity --------------------------------------


def test_debate_ranges_have_valid_bounds() -> None:
    """Every range must have lo ≤ hi (debate) and lo ≤ hi (physical)."""
    for spec in DEBATE_RANGES:
        assert spec.debate_lo <= spec.debate_hi
        assert spec.physical_lo <= spec.physical_hi
        # Debate envelope must fit inside the physical envelope
        # (the physical bound is a hard safety limit; the debate
        # envelope is the recommended subrange).
        assert spec.physical_lo <= spec.debate_lo
        assert spec.debate_hi <= spec.physical_hi
        assert spec.debate_width > 0


def test_debate_ranges_reference_real_uniqconfig_fields() -> None:
    cfg = load_config()
    for spec in DEBATE_RANGES:
        assert hasattr(cfg, spec.field), (
            f"DEBATE_RANGES references {spec.field!r}, "
            f"but UniqConfig has no such field"
        )


def test_debate_ranges_citations_reference_documented_sources() -> None:
    """Every debate_ref must cite a real, in-repo source.

    Guard against future regressions where someone adds a new range
    but forgets to back it with a documentation citation.
    """
    allowed_sources = (
        "profiles.py",
        "final_spec_FULL.md",
        "docs/DEBATE_TOPICS/editor-agent-v2.md",
    )
    for spec in DEBATE_RANGES:
        assert any(src in spec.debate_ref for src in allowed_sources), (
            f"{spec.field!r} debate_ref does not cite an in-repo source: "
            f"{spec.debate_ref!r}"
        )


# ---- jitter_value -------------------------------------------------


def test_jitter_value_at_pct_zero_returns_center() -> None:
    rng = random.Random(0)
    cfg = _medium_cfg()
    for spec in DEBATE_RANGES:
        center = float(getattr(cfg, spec.field))
        assert jitter_value(spec, center, rng, 0) == round(center, 6)


def test_jitter_value_at_debate_pct_stays_inside_physical_bounds() -> None:
    """At DEBATE_PCT the spread equals one debate envelope; values
    must stay within the physical safety bounds."""
    rng = random.Random(12345)
    cfg = _medium_cfg()
    for spec in DEBATE_RANGES:
        center = float(getattr(cfg, spec.field))
        samples = [
            jitter_value(spec, center, rng, DEBATE_PCT) for _ in range(5000)
        ]
        assert min(samples) >= spec.physical_lo - 1e-6
        assert max(samples) <= spec.physical_hi + 1e-6


def test_jitter_value_at_max_pct_clamps_to_physical_bounds() -> None:
    """At MAX_UNIQUENESS_PCT the would-be spread is 4× the debate
    envelope; clamping must keep values inside the physical bounds."""
    rng = random.Random(98765)
    cfg = _medium_cfg()
    for spec in DEBATE_RANGES:
        center = float(getattr(cfg, spec.field))
        samples = [
            jitter_value(spec, center, rng, MAX_UNIQUENESS_PCT)
            for _ in range(5000)
        ]
        assert min(samples) >= spec.physical_lo - 1e-6
        assert max(samples) <= spec.physical_hi + 1e-6


def test_jitter_value_intermediate_pct_is_narrower_than_max() -> None:
    rng_mid = random.Random(7)
    rng_max = random.Random(7)
    spec = DEBATE_RANGES[0]  # zoom
    cfg = _medium_cfg()
    center = float(getattr(cfg, spec.field))
    mid_samples = [jitter_value(spec, center, rng_mid, 25) for _ in range(2000)]
    max_samples = [
        jitter_value(spec, center, rng_max, MAX_UNIQUENESS_PCT)
        for _ in range(2000)
    ]
    mid_spread = max(mid_samples) - min(mid_samples)
    max_spread = max(max_samples) - min(max_samples)
    assert mid_spread < max_spread


# ---- jitter_config ------------------------------------------------


def test_jitter_config_at_pct_zero_is_no_op() -> None:
    """pct=0 returns the input config unchanged.

    This is what lets all pre-existing golden tests pass without any
    seeding instrumentation.
    """
    cfg = load_config()
    out, report = jitter_config(cfg, uniqueness_pct=0, seed=42)
    assert out is cfg
    assert report.uniqueness_pct == 0
    assert report.variation_pct == 0.0
    assert report.picks == {}


def test_jitter_config_inside_green_band_changes_at_least_one_knob() -> None:
    cfg = _medium_cfg()
    out, report = jitter_config(cfg, uniqueness_pct=DEBATE_PCT, seed=42)
    changed = [
        getattr(out, spec.field) != getattr(cfg, spec.field)
        for spec in DEBATE_RANGES
    ]
    assert any(changed)
    # All randomized fields must remain inside the physical bounds.
    for spec in DEBATE_RANGES:
        v = getattr(out, spec.field)
        assert spec.physical_lo <= v <= spec.physical_hi
    assert set(report.picks.keys()) == {spec.field for spec in DEBATE_RANGES}


def test_jitter_config_same_seed_same_output() -> None:
    cfg = _medium_cfg()
    a, ra = jitter_config(cfg, uniqueness_pct=DEBATE_PCT, seed=42)
    b, rb = jitter_config(cfg, uniqueness_pct=DEBATE_PCT, seed=42)
    for spec in DEBATE_RANGES:
        assert getattr(a, spec.field) == getattr(b, spec.field)
    assert ra.picks == rb.picks
    assert ra.variation_pct == rb.variation_pct


def test_jitter_config_different_seeds_produce_different_outputs() -> None:
    cfg = _medium_cfg()
    a, _ = jitter_config(cfg, uniqueness_pct=DEBATE_PCT, seed=1)
    b, _ = jitter_config(cfg, uniqueness_pct=DEBATE_PCT, seed=2)
    diffs = [
        getattr(a, spec.field) != getattr(b, spec.field)
        for spec in DEBATE_RANGES
    ]
    assert any(diffs)


def test_jitter_config_100_seeds_produce_diverse_outputs() -> None:
    """User's headline use case: 100 uploads of the same video → 100 different configs."""
    cfg = _medium_cfg()
    fingerprints: set[tuple[float, ...]] = set()
    for job_id in range(100):
        out, _ = jitter_config(
            cfg,
            uniqueness_pct=DEBATE_PCT,
            seed=make_seed(job_id=job_id, clip_index=0),
        )
        fingerprints.add(
            tuple(round(getattr(out, spec.field), 4) for spec in DEBATE_RANGES)
        )
    # All 100 fingerprints must be unique. Collision probability for
    # 100 draws from a ~10^16-sized space is negligible.
    assert len(fingerprints) == 100


def test_jitter_config_requires_seed_or_rng_when_pct_positive() -> None:
    cfg = load_config()
    with pytest.raises(ValueError):
        jitter_config(cfg, uniqueness_pct=10)


def test_jitter_config_origin_marked_uniqueness() -> None:
    cfg = _medium_cfg()
    out, _ = jitter_config(cfg, uniqueness_pct=30, seed=42)
    for spec in DEBATE_RANGES:
        assert out._origin[spec.field] == "uniqueness:30"


def test_jitter_config_preserves_other_fields() -> None:
    """Jitter must NOT touch any field outside DEBATE_RANGES."""
    cfg = _medium_cfg()
    out, _ = jitter_config(
        cfg, uniqueness_pct=MAX_UNIQUENESS_PCT, seed=42
    )
    randomized_fields = {spec.field for spec in DEBATE_RANGES}
    for f in UniqConfig.__dataclass_fields__:
        if f.startswith("_") or f in randomized_fields:
            continue
        assert getattr(out, f) == getattr(cfg, f), (
            f"field {f!r} was modified by jitter_config but is NOT in "
            f"DEBATE_RANGES"
        )


# ---- "slider=5 → random ∈ [3, 5]" — user's exact formula ----------


def test_variation_pct_at_slider_5_falls_in_3_to_5_band() -> None:
    """Per-video variation MUST land in [0.6 × slider, slider].

    From the user's own example: "если я выберу 5 процентов то рандом
    должен быть от 3% до 5%". That math is the entire promise of the
    slider, so we burn 2 000 seeds to verify the empirical distribution.
    """
    cfg = _medium_cfg()
    observed: list[float] = []
    for seed in range(2000):
        _, report = jitter_config(cfg, uniqueness_pct=5, seed=seed)
        observed.append(report.variation_pct)
    assert min(observed) >= 3.0 - 1e-9
    assert max(observed) <= 5.0 + 1e-9


def test_variation_pct_band_scales_with_slider() -> None:
    """Same property at the debate boundary: slider=100 → v ∈ [60, 100]."""
    cfg = _medium_cfg()
    observed: list[float] = []
    for seed in range(1000):
        _, report = jitter_config(cfg, uniqueness_pct=DEBATE_PCT, seed=seed)
        observed.append(report.variation_pct)
    assert min(observed) >= 0.6 * DEBATE_PCT - 1e-9
    assert max(observed) <= float(DEBATE_PCT) + 1e-9


# ---- config / env var integration ---------------------------------


def test_uniqueness_pct_env_var_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EDITOR_UNIQUENESS_PCT", "245")
    cfg = load_config()
    assert cfg.uniqueness_pct == 245


def test_uniqueness_pct_default_is_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EDITOR_UNIQUENESS_PCT", raising=False)
    cfg = load_config()
    assert cfg.uniqueness_pct == 0


def test_uniqueness_pct_per_job_override_wins() -> None:
    cfg = load_config(overrides={"uniqueness_pct": 30})
    assert cfg.uniqueness_pct == 30


def test_uniqueness_pct_env_var_clamps_above_max(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EDITOR_UNIQUENESS_PCT", str(MAX_UNIQUENESS_PCT + 50))
    cfg = load_config()
    assert cfg.uniqueness_pct == MAX_UNIQUENESS_PCT


# ---- RandomizationReport contents ---------------------------------


def test_report_picks_carry_debate_ref_and_envelope() -> None:
    cfg = _medium_cfg()
    _, report = jitter_config(
        cfg, uniqueness_pct=DEBATE_PCT, seed=42
    )
    for spec in DEBATE_RANGES:
        pick = report.picks[spec.field]
        assert pick["debate_ref"] == spec.debate_ref
        assert pick["debate_lo"] == spec.debate_lo
        assert pick["debate_hi"] == spec.debate_hi
        assert pick["physical_lo"] == spec.physical_lo
        assert pick["physical_hi"] == spec.physical_hi
        assert pick["center"] == float(getattr(cfg, spec.field))
        assert spec.physical_lo <= pick["value"] <= spec.physical_hi


def test_report_zone_label_matches_pct() -> None:
    cfg = _medium_cfg()
    cases = [
        (10, "🟢"),
        (50, "🟢"),
        (DEBATE_PCT, "🟢"),
        (DEBATE_PCT + 1, "🔴"),
        (250, "🔴"),
        (MAX_UNIQUENESS_PCT, "🔴"),
    ]
    for pct, marker in cases:
        _, report = jitter_config(cfg, uniqueness_pct=pct, seed=1)
        assert marker in report.zone_label, (
            f"pct={pct}: expected {marker} in {report.zone_label!r}"
        )


def test_report_is_immutable_dataclass() -> None:
    cfg = _medium_cfg()
    _, report = jitter_config(cfg, uniqueness_pct=30, seed=1)
    assert isinstance(report, RandomizationReport)
    with pytest.raises(AttributeError):
        report.uniqueness_pct = 0  # type: ignore[misc]


# ---- SLIDER_STEPS contract ----------------------------------------


def test_slider_steps_starts_at_zero() -> None:
    assert SLIDER_STEPS[0] == 0


def test_slider_steps_ends_at_max() -> None:
    assert SLIDER_STEPS[-1] == MAX_UNIQUENESS_PCT


def test_slider_steps_strictly_increasing() -> None:
    for a, b in zip(SLIDER_STEPS, SLIDER_STEPS[1:], strict=False):
        assert a < b


def test_slider_steps_contain_debate_pct() -> None:
    """The debate boundary MUST be a directly-clickable step."""
    assert DEBATE_PCT in SLIDER_STEPS


def test_slider_steps_have_finer_granularity_in_green_band() -> None:
    """The green band (≤ DEBATE_PCT) needs ≥ 5 steps for usable control."""
    green = [s for s in SLIDER_STEPS if 0 < s <= DEBATE_PCT]
    assert len(green) >= 5


# ---- UI integration: addon screen has the slider ------------------


def test_addon_handlers_export_uniqueness_helpers() -> None:
    from bot.addons.editor_agent import handlers

    assert hasattr(handlers, "current_uniqueness_pct")
    assert hasattr(handlers, "show_uniqueness_screen")
    assert hasattr(handlers, "_kb_uniqueness_screen")
    assert hasattr(handlers, "_uniqueness_screen_body")


def test_addon_uniqueness_body_contains_all_debate_ranges() -> None:
    """The slider screen must surface every range from DEBATE_RANGES."""
    from bot.addons.editor_agent import handlers

    body = handlers._uniqueness_screen_body()
    for spec in DEBATE_RANGES:
        assert spec.field in body
        assert spec.debate_ref in body


def test_addon_uniqueness_body_warns_above_debate_pct() -> None:
    """The screen must explicitly call out the red ("выходим из рамок") zone."""
    from bot.addons.editor_agent import handlers

    body = handlers._uniqueness_screen_body()
    assert "выходим из рамок дебатов" in body


def test_addon_uniqueness_body_documents_per_job_formula() -> None:
    """The "slider=5 → [3, 5]" rule must be visible to the operator."""
    from bot.addons.editor_agent import handlers

    body = handlers._uniqueness_screen_body()
    assert "0.6" in body
    assert "uniform" in body
