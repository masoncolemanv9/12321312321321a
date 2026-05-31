"""Top-level v6 creative-planner orchestrator (final_spec §1.2, §12.5).

This is the Part 10 capstone glue: it takes a parsed
``director_brief.json`` mapping and walks the §1.2 forward pipeline

    validation → beat_sheet → zoom → cuts → mirror → subtitle →
    audio → color → blur_fill → hook → manifest → reasoning →
    creative_intensity_score

emitting a single :class:`CreativePlanResult` whose ``edit_plan`` dict
matches the §4 sidecar schema (and validates against
``bot/uniqueization/assets/schemas/edit_plan.schema.json``).

The §1.3 failure-mode taxonomy is the only thing the worker cares
about externally:

* ``status="full"`` — every stage succeeded.
* ``status="degraded"`` — at least one DEGRADE finding from
  :mod:`.validation` OR at least one optional sub-planner raised
  (caught here, never bubbled). The plan is still usable.
* ``status="skipped"`` — caller passed ``force_skip=True``
  (typically ``EDITOR_V6_FORCE_SKIP=true``). No planning is run; the
  worker should fall back to the v2.0 / v2.1 path.
* ``status="error"`` — validation produced at least one HARD finding,
  OR a HARD-rule sub-planner raised an exception we cannot recover
  from (e.g. :class:`hook_planner.BudgetViolation`). The plan is
  unusable and the worker MUST abort to the v2.0 / v2.1 fallback.

Pure-function contract (§12.1):

* No I/O. Caller is responsible for loading the brief JSON, writing
  the resulting ``edit_plan`` dict to disk, and feeding the plan into
  the renderer adapter.
* No LLM in safety-critical decisions (§12.2). ``manifest_reasoning``
  is emitted here as a *deterministic* rule trace so the worker /
  publisher can render the manifest without any model call. Callers
  that want LLM-generated reasoning may rewrite that field
  *after* validation.
* Deterministic given identical input (within MAJOR/MINOR version of
  every sub-module).

The orchestrator never raises on optional-stage failures — it captures
them as warnings inside ``CreativePlanResult.diagnostics`` and falls
back to a sensible default (e.g. an empty :class:`SubtitlePlan`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .audio_planner import AudioPlan, plan_audio
from .beat_sheet import Beat, build_beat_sheet
from .blur_fill_planner import BlurFillPlan, plan_blur_fill
from .color_planner import ColorPlan, plan_color
from .cut_planner import CutPlan, plan_cuts_final
from .hook_planner import BudgetViolation, HookPlan, plan_hook
from .locks import StyleLock, get_lock
from .mirror_planner import MirrorPlan, plan_mirror_final
from .subtitle_planner import SubtitlePlan, plan_subtitle
from .validation import (
    Diagnostic,
    ValidationResult,
    validate_director_brief,
)
from .zoom_planner import ZoomCurve, plan_zoom_curve

__all__ = (
    "CREATIVE_PLAN_SCHEMA_VERSION",
    "CreativePlanResult",
    "CreativePlanError",
    "CREATIVE_INTENSITY_CEILINGS",
    "run_creative_planner",
    "compute_creative_intensity_score",
    "build_manifest_reasoning",
)


# §4 sidecar schema version this orchestrator emits.
CREATIVE_PLAN_SCHEMA_VERSION: str = "1.0.0"


# §4.1 + §15 profile ceilings. Above these the planner emits a
# DEGRADE warning but does NOT clamp the score (the warning surfaces
# in the manifest so the CI flag can fire).
CREATIVE_INTENSITY_CEILINGS: Mapping[str, float] = {
    "light": 0.40,
    "medium": 0.60,
    "heavy": 0.80,
}


# §1.3 status taxonomy. Exposed as a frozenset so callers can validate.
STATUS_ENUM: frozenset[str] = frozenset({"full", "degraded", "skipped", "error"})


# Map planner cut_planner.CUT_KIND_ENUM ("filler" | "dead_air" |
# "repeat") to ``bot.uniqueization.cuts.CutKind`` accepted by the v2.0
# / v2.1 worker ("filler_word" | "silence_pause" | "dead_air").
# Defined here so the renderer adapter and tests share one source.
_CUT_KIND_PLANNER_TO_WORKER: Mapping[str, str] = {
    "filler": "filler_word",
    "dead_air": "dead_air",
    "repeat": "filler_word",
}


class CreativePlanError(RuntimeError):
    """Raised when the orchestrator cannot construct a usable plan.

    The worker catches this and abort-or-falls-back per §1.3 / §9.13.
    """


@dataclass(frozen=True, slots=True)
class CreativePlanResult:
    """Outcome of :func:`run_creative_planner`.

    ``edit_plan`` is the §4 dict (already populated with safe
    defaults). On ``status="error"`` / ``status="skipped"`` callers
    SHOULD NOT pass ``edit_plan`` to the renderer — its content is
    incomplete by definition. ``diagnostics`` are the deterministic
    findings emitted by either :mod:`.validation` or the orchestrator
    itself (sub-planner downgrade warnings).
    """

    status: str
    edit_plan: dict[str, Any]
    validation: ValidationResult
    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)
    intensity_score: float = 0.0
    reasoning: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_usable(self) -> bool:
        """True when the renderer adapter can consume ``edit_plan``."""
        return self.status in {"full", "degraded"}


# ---- helpers ------------------------------------------------------


def _utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO 8601 form.

    Determinism note: callers that pin time for tests should pass
    ``produced_at_iso`` directly to :func:`run_creative_planner`.
    """
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_call(
    fn,
    *args,
    fallback,
    diags: list[Diagnostic],
    stage_name: str,
    hard: bool = False,
    **kwargs,
):
    """Run ``fn`` under exception capture.

    On exception (other than :class:`BudgetViolation` when ``hard``
    is True) returns ``fallback`` and appends a ``"degrade"``
    :class:`Diagnostic`. ``BudgetViolation`` is always a HARD finding
    (it indicates the §8.7 budget was forced).
    """
    try:
        return fn(*args, **kwargs)
    except BudgetViolation as exc:
        diags.append(
            Diagnostic(
                severity="hard",
                code=f"{stage_name}_budget_violation",
                message=f"{stage_name} raised BudgetViolation: {exc}",
                path=stage_name,
            )
        )
        if hard:
            raise CreativePlanError(str(exc)) from exc
        return fallback
    except Exception as exc:  # noqa: BLE001 — pure-function failure policy
        diags.append(
            Diagnostic(
                severity="degrade",
                code=f"{stage_name}_failed",
                message=f"{stage_name} raised {type(exc).__name__}: {exc}",
                path=stage_name,
            )
        )
        return fallback


def _resolve_lock(
    director_brief: Mapping[str, Any], lock_override: StyleLock | None
) -> tuple[StyleLock | None, str | None]:
    """Resolve a :class:`StyleLock` either from the caller or the brief."""
    if lock_override is not None:
        return lock_override, lock_override.lock_id
    raw = director_brief.get("style_lock_id")
    if isinstance(raw, str) and raw:
        lock = get_lock(raw)
        if lock is not None:
            return lock, lock.lock_id
    return None, None


def _beat_to_dict(
    beat: Beat,
    *,
    color_intent_by_id: Mapping[str, str],
    subtitle_intensity_by_id: Mapping[str, str],
    audio_policy_by_id: Mapping[str, str],
    blur_fill_by_id: Mapping[str, str],
) -> dict[str, Any]:
    """Serialise a :class:`Beat` into the §4 ``beat_sheet[]`` shape."""
    return {
        "beat_id": beat.beat_id,
        "start_s": beat.start_s,
        "end_s": beat.end_s,
        "purpose": beat.purpose,
        "primary_move": beat.primary_move,
        "emotional_intent": beat.emotional_intent,
        "supports": list(beat.supports),
        "note": beat.note or None,
        "color_intent": color_intent_by_id.get(beat.beat_id, ""),
        "subtitle_intensity": subtitle_intensity_by_id.get(beat.beat_id, ""),
        "audio_policy": audio_policy_by_id.get(beat.beat_id, ""),
        "blur_fill": blur_fill_by_id.get(beat.beat_id, ""),
        "reason": beat.reason,
    }


def _zoom_to_dict(curve: ZoomCurve) -> list[dict[str, Any]]:
    return [
        {
            "start_s": seg.start_s,
            "end_s": seg.end_s,
            "mode": seg.mode,
            "target": seg.target,
            "max_zoom": seg.max_zoom,
            "transition_ms": int(round(seg.transition_ms)),
            "reason": seg.reason,
        }
        for seg in curve.segments
    ]


def _cuts_to_dict(plan: CutPlan) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": d.candidate_id,
            "start_s": d.start_s,
            "end_s": d.end_s,
            "kind": d.kind,
            "apply": bool(d.apply),
            "reason": d.reason,
            "confidence": d.confidence,
        }
        for d in plan.decisions
    ]


def _mirror_to_dict(plan: MirrorPlan) -> list[dict[str, Any]]:
    return [
        {
            "cluster_id": d.cluster_id,
            "apply": bool(d.apply),
            "reason": d.reason,
            "scene_ids": list(d.scene_ids),
        }
        for d in plan.decisions
    ]


def _subtitle_to_dict(plan: SubtitlePlan) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    for ev in plan.events:
        events.append(
            {
                "beat_id": ev.beat_id,
                "render_start_s": ev.start_s,
                "render_end_s": ev.end_s,
                "style_id": ev.style_id,
                "y_position_preferred": ev.y_position_preferred,
                "y_position_fallbacks": list(ev.y_position_fallbacks),
                "emphasis_words": [
                    {
                        "word": w.word,
                        "start_s": w.start_s,
                        "end_s": w.end_s,
                        "salience": w.salience,
                        "semantic_role": w.semantic_role,
                    }
                    for w in ev.emphasis_words
                ],
                "reason": ev.reason,
            }
        )
    return {
        "events": events,
        "profile": plan.profile,
        "lock_id": plan.lock_id,
        "warnings": list(plan.warnings),
    }


def _audio_to_dict(plan: AudioPlan) -> dict[str, Any]:
    return {
        "loudness_target_lufs": plan.loudness_target_lufs,
        "true_peak_max_dbtp": plan.true_peak_max_dbtp,
        "profile": plan.profile,
        "lock_id": plan.lock_id,
        "beat_policies": [
            {
                "beat_id": p.beat_id,
                "start_s": p.start_s,
                "end_s": p.end_s,
                "policy": p.policy,
                "duck_target_db": p.duck_target_db,
                "silence_duration_s": p.silence_duration_s,
                "bump_events": [
                    {
                        "beat_id": b.beat_id,
                        "word": b.word,
                        "start_s": b.start_s,
                        "end_s": b.end_s,
                        "delta_db": b.delta_db,
                    }
                    for b in p.bump_events
                ],
                "reason": p.reason,
            }
            for p in plan.beat_policies
        ],
        "warnings": list(plan.warnings),
    }


def _color_to_dict(plan: ColorPlan) -> dict[str, Any]:
    return {
        "profile": plan.profile,
        "lock_id": plan.lock_id,
        "segments": [
            {
                "beat_id": s.beat_id,
                "start_s": s.start_s,
                "end_s": s.end_s,
                "intent": {
                    "name": s.intent.name,
                    "brightness": s.intent.brightness,
                    "contrast": s.intent.contrast,
                    "saturation": s.intent.saturation,
                    "hue_deg": s.intent.hue_deg,
                },
                "reason": s.reason,
            }
            for s in plan.segments
        ],
        "transitions": [
            {
                "from_beat_id": t.from_beat_id,
                "to_beat_id": t.to_beat_id,
                "start_s": t.start_s,
                "end_s": t.end_s,
                "duration_ms": t.duration_ms,
            }
            for t in plan.transitions
        ],
        "warnings": list(plan.warnings),
    }


def _blur_fill_to_dict(plan: BlurFillPlan) -> dict[str, Any]:
    return {
        "profile": plan.profile,
        "lock_id": plan.lock_id,
        "source_aspect_ratio": plan.source_aspect_ratio,
        "segments": [
            {
                "beat_id": s.beat_id,
                "start_s": s.start_s,
                "end_s": s.end_s,
                "mode": s.mode,
                "darken_pct": s.darken_pct,
                "reason": s.reason,
            }
            for s in plan.segments
        ],
        "warnings": list(plan.warnings),
    }


def _hook_to_dict(plan: HookPlan) -> dict[str, Any]:
    overlay = None
    if plan.overlay is not None:
        overlay = {
            "text": plan.overlay.text,
            "start_s": plan.overlay.start_s,
            "end_s": plan.overlay.end_s,
            "y_position": plan.overlay.y_position,
            "suppresses_dialogue_subtitle": plan.overlay.suppresses_dialogue_subtitle,
            "reason": plan.overlay.reason,
        }
    return {
        "primary_move": plan.primary_move,
        "supports": list(plan.supports),
        "overlay": overlay,
        "forbidden_effects": list(plan.forbidden_effects),
        "profile": plan.profile,
        "lock_id": plan.lock_id,
        "reason": plan.reason,
        "warnings": list(plan.warnings),
    }


def _hook_overlay_inline(plan: HookPlan) -> dict[str, Any] | None:
    """Spec §4 top-level ``hook_overlay`` block (inline, no nesting)."""
    if plan.overlay is None:
        return None
    return {
        "text": plan.overlay.text,
        "start_s": plan.overlay.start_s,
        "end_s": plan.overlay.end_s,
        "y_position": plan.overlay.y_position,
        "suppresses_dialogue_subtitle": plan.overlay.suppresses_dialogue_subtitle,
    }


def _timeline_map_from_cuts(
    beats: tuple[Beat, ...], cuts_plan: CutPlan, render_duration_s: float
) -> list[dict[str, Any]]:
    """Render-time identity map when no cuts apply, else map kept segments.

    For the §4 sidecar we record what the renderer will consume; the
    fine-grained timeline_map produced by :mod:`bot.uniqueization.timeline`
    is recomputed inside the worker (Part 2 logic) so we do not
    duplicate that math here. The sidecar version is descriptive.
    """
    applied = [d for d in cuts_plan.decisions if d.apply]
    if not applied:
        return [
            {
                "render_start_s": 0.0,
                "render_end_s": render_duration_s,
                "source_start_s": 0.0,
                "source_end_s": render_duration_s,
                "reorder_source": None,
            }
        ]
    segments: list[dict[str, Any]] = []
    source_cursor = 0.0
    render_cursor = 0.0
    applied_sorted = sorted(applied, key=lambda d: d.start_s)
    for cut in applied_sorted:
        if cut.start_s > source_cursor + 1e-6:
            seg_len = cut.start_s - source_cursor
            segments.append(
                {
                    "render_start_s": render_cursor,
                    "render_end_s": render_cursor + seg_len,
                    "source_start_s": source_cursor,
                    "source_end_s": cut.start_s,
                    "reorder_source": None,
                }
            )
            render_cursor += seg_len
        source_cursor = max(source_cursor, cut.end_s)
    if source_cursor < render_duration_s - 1e-6:
        seg_len = render_duration_s - source_cursor
        segments.append(
            {
                "render_start_s": render_cursor,
                "render_end_s": render_cursor + seg_len,
                "source_start_s": source_cursor,
                "source_end_s": render_duration_s,
                "reorder_source": None,
            }
        )
    if not segments:
        segments.append(
            {
                "render_start_s": 0.0,
                "render_end_s": render_duration_s,
                "source_start_s": 0.0,
                "source_end_s": render_duration_s,
                "reorder_source": None,
            }
        )
    return segments


def _frame_export_hints(beats: tuple[Beat, ...]) -> list[dict[str, Any]]:
    """One §11.1 export hint per non-transition beat (mid-beat timestamp)."""
    hints: list[dict[str, Any]] = []
    for beat in beats:
        if beat.purpose == "transition":
            continue
        if beat.duration_s <= 0:
            continue
        mid = beat.start_s + beat.duration_s / 2.0
        score = 0.6
        if beat.purpose in ("reveal", "hook"):
            score = 0.9
        elif beat.purpose in ("emotional_hold", "reaction"):
            score = 0.8
        elif beat.purpose == "resolution":
            score = 0.7
        hints.append(
            {
                "timestamp_s": round(mid, 3),
                "beat_id": beat.beat_id,
                "suitability_score": score,
            }
        )
    return hints


# ---- §15 creative_intensity_score ---------------------------------


def compute_creative_intensity_score(
    *,
    beats: tuple[Beat, ...],
    zoom_curve: ZoomCurve,
    cuts: CutPlan,
    subtitle: SubtitlePlan,
    render_duration_s: float,
) -> float:
    """§15 weighted score.

    Each component is mapped to ``[0, 1]`` then weighted:

    * ``num_applied_cuts / render_duration_s`` capped at 1.0 (1 cut/s).
    * ``zoom_velocity_peak`` — observed peak |Δzoom/Δt| over the curve,
      normalised to MOTION_VELOCITY_CAP=0.08 ⇒ 1.0.
    * ``color_delta_magnitude`` — placeholder (intentionally 0 until we
      have a reliable per-beat measure; the v2.1 path doesn't ship
      per-beat colorgrade either). Kept in the formula for forward-compat.
    * ``subtitle_intensity`` — share of beats with karaoke_emphasis.

    Weights sum to 1.0. The score is clamped to ``[0, 1]``.
    """
    if render_duration_s <= 0:
        return 0.0

    applied_cuts = sum(1 for d in cuts.decisions if d.apply)
    cuts_component = min(1.0, applied_cuts / max(render_duration_s, 1.0))

    zoom_component = 0.0
    if len(zoom_curve.segments) >= 2:
        peak = 0.0
        prev = zoom_curve.segments[0]
        for cur in zoom_curve.segments[1:]:
            dt = max(cur.duration_s, 1e-3)
            dv = abs(cur.max_zoom - prev.max_zoom) / dt
            peak = max(peak, dv)
            prev = cur
        zoom_component = min(1.0, peak / 0.08)

    color_component = 0.0

    subtitle_component = 0.0
    if beats:
        karaoke = sum(
            1 for ev in subtitle.events if ev.style_id == "karaoke_emphasis"
        )
        subtitle_component = karaoke / max(len(beats), 1)

    score = (
        0.35 * cuts_component
        + 0.35 * zoom_component
        + 0.10 * color_component
        + 0.20 * subtitle_component
    )
    return max(0.0, min(1.0, score))


def build_manifest_reasoning(
    *,
    director_brief: Mapping[str, Any],
    beats: tuple[Beat, ...],
    cuts: CutPlan,
    mirror: MirrorPlan,
    hook: HookPlan,
    subtitle: SubtitlePlan,
    intensity_score: float,
    profile: str,
) -> tuple[str, ...]:
    """Deterministic §12.3 manifest reasoning (no LLM here).

    Callers that want LLM rewrites must replace this field after
    validation. The output is bounded to ≤8 lines so log volume is
    predictable.
    """
    lines: list[str] = []
    goal = director_brief.get("creative_goal")
    if isinstance(goal, Mapping):
        text = str(goal.get("text", "")).strip()
        if text:
            lines.append(f"creative_goal: {text[:160]}")

    hook_summary = (
        f"hook beat applies {hook.primary_move}"
        + (f" + {hook.supports[0]}" if hook.supports else "")
    )
    if hook.overlay is not None:
        hook_summary += " + hook_overlay"
    lines.append(hook_summary + ".")

    applied = sum(1 for d in cuts.decisions if d.apply)
    declined = sum(1 for d in cuts.decisions if not d.apply)
    if cuts.decisions:
        lines.append(
            f"cuts_final: {applied} applied, {declined} declined (preservation rules)."
        )

    mirror_apply = sum(1 for d in mirror.decisions if d.apply)
    if mirror.decisions:
        lines.append(
            f"mirror_final: {mirror_apply}/{len(mirror.decisions)} clusters applied."
        )

    karaoke = sum(
        1 for ev in subtitle.events if ev.style_id == "karaoke_emphasis"
    )
    if beats:
        lines.append(
            f"subtitle_plan: {karaoke}/{len(beats)} beats karaoke_emphasis."
        )

    ceiling = CREATIVE_INTENSITY_CEILINGS.get(profile, 0.6)
    if intensity_score > ceiling + 1e-9:
        lines.append(
            f"creative_intensity_score={intensity_score:.2f} exceeds {profile} ceiling {ceiling}."
        )
    else:
        lines.append(
            f"creative_intensity_score={intensity_score:.2f} within {profile} ceiling {ceiling}."
        )

    return tuple(lines)


# ---- public entry point -------------------------------------------


def run_creative_planner(
    director_brief: Mapping[str, Any] | None,
    *,
    profile: str = "medium",
    lock: StyleLock | None = None,
    force_skip: bool = False,
    clip_id: str = "",
    source_path: str = "",
    source_aspect_ratio: float = 0.5625,
    produced_at_iso: str | None = None,
) -> CreativePlanResult:
    """Run the §1.2 forward pipeline and return a :class:`CreativePlanResult`.

    Inputs:

    * ``director_brief`` — parsed dict from ``director_brief.json``.
      ``None`` indicates the sidecar is missing — the worker uses
      that to fall back to v2.0 / v2.1; we DO NOT short-circuit here
      (we still emit an empty plan with ``status="skipped"`` so the
      worker has a consistent return shape).
    * ``profile`` — one of ``light`` / ``medium`` / ``heavy``.
    * ``lock`` — caller-resolved :class:`StyleLock`. If omitted,
      the orchestrator resolves it from
      ``director_brief.style_lock_id``.
    * ``force_skip`` — ``EDITOR_V6_FORCE_SKIP=true`` short-circuit
      per §13.1. The worker sets this when its own dispatch wants v6
      bypassed for a single job.
    * ``clip_id`` / ``source_path`` — populate the §4 metadata block.
    * ``source_aspect_ratio`` — float W/H feed for blur-fill planning;
      defaults to a portrait 0.5625 (9:16) so callers that don't
      probe still get a coherent plan.
    * ``produced_at_iso`` — explicit timestamp override for tests.

    The function NEVER raises on optional-stage failure. It only
    raises :class:`CreativePlanError` when a HARD-rule planner
    raises and the failure cannot be recovered (e.g. an explicit
    :class:`BudgetViolation`). Callers catch and fall back to v2.0.
    """
    diagnostics: list[Diagnostic] = []

    if produced_at_iso is None:
        produced_at_iso = _utc_now_iso()

    # ---- (0) ``force_skip`` short-circuit --------------------------
    if force_skip:
        return CreativePlanResult(
            status="skipped",
            edit_plan=_make_skipped_plan(
                clip_id=clip_id,
                source_path=source_path,
                produced_at_iso=produced_at_iso,
                profile=profile,
                lock_id=lock.lock_id if lock is not None else None,
                reason="env_force_skip",
            ),
            validation=ValidationResult(creative_planner_status="skipped"),
            diagnostics=(),
            intensity_score=0.0,
            reasoning=(
                "creative_planner_status=skipped (EDITOR_V6_FORCE_SKIP=true).",
            ),
        )

    if director_brief is None:
        return CreativePlanResult(
            status="skipped",
            edit_plan=_make_skipped_plan(
                clip_id=clip_id,
                source_path=source_path,
                produced_at_iso=produced_at_iso,
                profile=profile,
                lock_id=lock.lock_id if lock is not None else None,
                reason="missing_director_brief",
            ),
            validation=ValidationResult(creative_planner_status="skipped"),
            diagnostics=(),
            intensity_score=0.0,
            reasoning=(
                "creative_planner_status=skipped (no director_brief sidecar).",
            ),
        )

    # ---- (1) validation -------------------------------------------
    validation = validate_director_brief(director_brief)

    if validation.is_error:
        return CreativePlanResult(
            status="error",
            edit_plan=_make_error_plan(
                clip_id=clip_id,
                source_path=source_path,
                produced_at_iso=produced_at_iso,
                profile=profile,
                director_brief=director_brief,
                validation=validation,
            ),
            validation=validation,
            diagnostics=validation.diagnostics,
            intensity_score=0.0,
            reasoning=(
                "creative_planner_status=error: validation HARD findings.",
            ),
        )

    # Mirror validator findings into the orchestrator diagnostics
    # list so downstream consumers see one merged tuple.
    diagnostics.extend(validation.diagnostics)

    resolved_lock, lock_id = _resolve_lock(director_brief, lock)

    # ---- (2) beat sheet ------------------------------------------
    beats = _safe_call(
        build_beat_sheet,
        director_brief,
        lock=resolved_lock,
        fallback=(),
        diags=diagnostics,
        stage_name="beat_sheet",
    )

    window = director_brief.get("window") or {}
    try:
        w_start = float(window.get("start_s", 0.0))
        w_end = float(window.get("end_s", 0.0))
    except (TypeError, ValueError):
        w_start = 0.0
        w_end = 0.0
    render_duration_s = max(0.0, w_end - w_start)

    if not beats:
        # The validator should have caught a brief with zero scenes;
        # but if we got here with no beats anyway, emit an empty —
        # but valid — plan and mark the result as degraded.
        diagnostics.append(
            Diagnostic(
                severity="degrade",
                code="beat_sheet_empty",
                message="beat_sheet returned no beats; downstream planners skipped",
                path="beat_sheet",
            )
        )

    # ---- (3) zoom ------------------------------------------------
    zoom_curve = _safe_call(
        plan_zoom_curve,
        beats,
        profile=profile,
        lock=resolved_lock,
        fallback=ZoomCurve(profile=profile, lock_id=lock_id),
        diags=diagnostics,
        stage_name="zoom_curve",
    )

    # ---- (4) cuts ------------------------------------------------
    cuts_plan = _safe_call(
        plan_cuts_final,
        director_brief,
        beats,
        profile=profile,
        lock=resolved_lock,
        fallback=CutPlan(),
        diags=diagnostics,
        stage_name="cuts_final",
    )

    # ---- (5) mirror ---------------------------------------------
    mirror_plan = _safe_call(
        plan_mirror_final,
        director_brief,
        fallback=MirrorPlan(),
        diags=diagnostics,
        stage_name="mirror_final",
    )

    # ---- (6) subtitle -------------------------------------------
    subtitle_plan = _safe_call(
        plan_subtitle,
        beats,
        director_brief,
        profile=profile,
        lock=resolved_lock,
        fallback=SubtitlePlan(profile=profile, lock_id=lock_id),
        diags=diagnostics,
        stage_name="subtitle_plan",
    )

    # ---- (7) audio ----------------------------------------------
    audio_plan = _safe_call(
        plan_audio,
        beats,
        subtitle_plan,
        profile=profile,
        lock=resolved_lock,
        fallback=AudioPlan(profile=profile, lock_id=lock_id),
        diags=diagnostics,
        stage_name="audio_plan",
    )

    # ---- (8) color ----------------------------------------------
    color_plan = _safe_call(
        plan_color,
        beats,
        profile=profile,
        lock=resolved_lock,
        fallback=ColorPlan(profile=profile, lock_id=lock_id),
        diags=diagnostics,
        stage_name="color_plan",
    )

    # ---- (9) blur-fill ------------------------------------------
    blur_fill_plan = _safe_call(
        plan_blur_fill,
        beats,
        source_aspect_ratio=source_aspect_ratio,
        profile=profile,
        lock=resolved_lock,
        fallback=BlurFillPlan(
            source_aspect_ratio=source_aspect_ratio,
            profile=profile,
            lock_id=lock_id,
        ),
        diags=diagnostics,
        stage_name="blur_fill_plan",
    )

    # ---- (10) hook ----------------------------------------------
    try:
        hook_plan = plan_hook(
            beats,
            director_brief,
            profile=profile,
            lock=resolved_lock,
        )
    except BudgetViolation as exc:
        # §8.7 budget violation is a HARD finding — we cannot ship a
        # plan that mis-budgets the hook.
        diagnostics.append(
            Diagnostic(
                severity="hard",
                code="hook_budget_violation",
                message=f"hook budget violated: {exc}",
                path="hook_plan",
            )
        )
        return CreativePlanResult(
            status="error",
            edit_plan=_make_error_plan(
                clip_id=clip_id,
                source_path=source_path,
                produced_at_iso=produced_at_iso,
                profile=profile,
                director_brief=director_brief,
                validation=validation,
                extra_diags=tuple(diagnostics),
            ),
            validation=validation,
            diagnostics=tuple(diagnostics),
            intensity_score=0.0,
            reasoning=(
                "creative_planner_status=error: hook budget violation.",
            ),
        )
    except Exception as exc:  # noqa: BLE001 — degrade per §1.3
        diagnostics.append(
            Diagnostic(
                severity="degrade",
                code="hook_plan_failed",
                message=f"hook_plan raised {type(exc).__name__}: {exc}",
                path="hook_plan",
            )
        )
        hook_plan = HookPlan(profile=profile, lock_id=lock_id)

    # ---- (11) compose §4 edit_plan dict -------------------------
    color_intent_by_id = {s.beat_id: s.intent.name for s in color_plan.segments}
    subtitle_intensity_by_id = {ev.beat_id: ev.style_id for ev in subtitle_plan.events}
    audio_policy_by_id = {p.beat_id: p.policy for p in audio_plan.beat_policies}
    blur_fill_by_id = {s.beat_id: s.mode for s in blur_fill_plan.segments}

    beat_sheet_dicts = [
        _beat_to_dict(
            beat,
            color_intent_by_id=color_intent_by_id,
            subtitle_intensity_by_id=subtitle_intensity_by_id,
            audio_policy_by_id=audio_policy_by_id,
            blur_fill_by_id=blur_fill_by_id,
        )
        for beat in beats
    ]

    intensity_score = compute_creative_intensity_score(
        beats=beats,
        zoom_curve=zoom_curve,
        cuts=cuts_plan,
        subtitle=subtitle_plan,
        render_duration_s=render_duration_s,
    )

    ceiling = CREATIVE_INTENSITY_CEILINGS.get(profile, 0.6)
    if intensity_score > ceiling + 1e-9:
        diagnostics.append(
            Diagnostic(
                severity="degrade",
                code="creative_intensity_exceeds_ceiling",
                message=(
                    f"creative_intensity_score={intensity_score:.3f} exceeds "
                    f"profile={profile} ceiling={ceiling}"
                ),
                path="creative_intensity_score",
            )
        )

    creative_goal = director_brief.get("creative_goal")
    if not isinstance(creative_goal, Mapping):
        creative_goal_dict: dict[str, Any] = {
            "text": "",
            "source": "absent",
            "confidence": 0.0,
        }
    else:
        creative_goal_dict = {
            "text": str(creative_goal.get("text", "")),
            "source": str(creative_goal.get("source", "")),
            "confidence": float(creative_goal.get("confidence", 0.0) or 0.0),
        }

    handles = window.get("handles_pre_s", 0.0) if isinstance(window, Mapping) else 0.0
    handles_post = (
        window.get("handles_post_s", 0.0) if isinstance(window, Mapping) else 0.0
    )

    edit_plan: dict[str, Any] = {
        "schema_version": CREATIVE_PLAN_SCHEMA_VERSION,
        "clip_id": clip_id,
        "produced_at_iso": produced_at_iso,
        "source_path": source_path,
        "input_window": {
            "start_s": w_start,
            "end_s": w_end,
        },
        "handles_used": {
            "pre_s": float(handles or 0.0),
            "post_s": float(handles_post or 0.0),
        },
        "profile": profile,
        "style_lock_id": lock_id,
        "creative_planner_status": "full",  # may be downgraded below
        "creative_goal": creative_goal_dict,
        "beat_sheet": beat_sheet_dicts,
        "scene_plans": [],
        "zoom_curve": _zoom_to_dict(zoom_curve),
        "cuts_final": _cuts_to_dict(cuts_plan),
        "mirror_final": _mirror_to_dict(mirror_plan),
        "mirror_decisions": _mirror_to_dict(mirror_plan),
        "timeline_map": _timeline_map_from_cuts(
            beats, cuts_plan, render_duration_s
        ),
        "subtitle_plan": _subtitle_to_dict(subtitle_plan),
        "hook_overlay": _hook_overlay_inline(hook_plan),
        "audio_plan": _audio_to_dict(audio_plan),
        "color_plan": _color_to_dict(color_plan),
        "blur_fill_plan": _blur_fill_to_dict(blur_fill_plan),
        "blur_fill": {
            "mode": (
                blur_fill_plan.segments[0].mode
                if blur_fill_plan.segments
                else "off"
            ),
            "source_aspect": f"{source_aspect_ratio:.4f}",
        },
        "hook_plan": _hook_to_dict(hook_plan),
        "frame_export_hints": _frame_export_hints(beats),
        "manifest_reasoning": [],
        "creative_intensity_score": round(intensity_score, 4),
        "validation": {
            "hard_errors": [
                d.code for d in diagnostics if d.severity == "hard"
            ],
            "degrade_warnings": [
                d.code for d in diagnostics if d.severity == "degrade"
            ],
            "errors": [
                {"code": d.code, "message": d.message, "path": d.path}
                for d in diagnostics
                if d.severity == "hard"
            ],
            "warnings": [
                {"code": d.code, "message": d.message, "path": d.path}
                for d in diagnostics
                if d.severity == "degrade"
            ],
        },
    }

    # Final status decision.
    any_hard = any(d.severity == "hard" for d in diagnostics)
    if any_hard:
        edit_plan["creative_planner_status"] = "error"
        final_status = "error"
    elif validation.is_degraded or any(
        d.severity == "degrade" for d in diagnostics
    ):
        edit_plan["creative_planner_status"] = "degraded"
        final_status = "degraded"
    else:
        edit_plan["creative_planner_status"] = "full"
        final_status = "full"

    reasoning = build_manifest_reasoning(
        director_brief=director_brief,
        beats=beats,
        cuts=cuts_plan,
        mirror=mirror_plan,
        hook=hook_plan,
        subtitle=subtitle_plan,
        intensity_score=intensity_score,
        profile=profile,
    )
    edit_plan["manifest_reasoning"] = list(reasoning)

    return CreativePlanResult(
        status=final_status,
        edit_plan=edit_plan,
        validation=validation,
        diagnostics=tuple(diagnostics),
        intensity_score=intensity_score,
        reasoning=reasoning,
    )


def planner_cut_kind_to_worker(kind: str) -> str:
    """Translate a planner CUT_KIND_ENUM value to the worker-side enum."""
    return _CUT_KIND_PLANNER_TO_WORKER.get(kind, "filler_word")


# ---- internal helpers --------------------------------------------


def _make_skipped_plan(
    *,
    clip_id: str,
    source_path: str,
    produced_at_iso: str,
    profile: str,
    lock_id: str | None,
    reason: str,
) -> dict[str, Any]:
    """Stub plan for the ``skipped`` status path."""
    return {
        "schema_version": CREATIVE_PLAN_SCHEMA_VERSION,
        "clip_id": clip_id,
        "produced_at_iso": produced_at_iso,
        "source_path": source_path,
        "input_window": {"start_s": 0.0, "end_s": 0.0},
        "handles_used": {"pre_s": 0.0, "post_s": 0.0},
        "profile": profile,
        "style_lock_id": lock_id,
        "creative_planner_status": "skipped",
        "creative_goal": {"text": "", "source": "absent", "confidence": 0.0},
        "beat_sheet": [],
        "scene_plans": [],
        "zoom_curve": [],
        "cuts_final": [],
        "mirror_final": [],
        "mirror_decisions": [],
        "timeline_map": [],
        "subtitle_plan": {"events": []},
        "hook_overlay": None,
        "audio_plan": {},
        "color_plan": {},
        "blur_fill_plan": {},
        "blur_fill": {"mode": "off", "source_aspect": ""},
        "hook_plan": {},
        "frame_export_hints": [],
        "manifest_reasoning": [f"creative_planner skipped: {reason}"],
        "creative_intensity_score": 0.0,
        "validation": {
            "hard_errors": [],
            "degrade_warnings": [],
            "errors": [],
            "warnings": [],
            "skipped_reason": reason,
        },
    }


def _make_error_plan(
    *,
    clip_id: str,
    source_path: str,
    produced_at_iso: str,
    profile: str,
    director_brief: Mapping[str, Any],
    validation: ValidationResult,
    extra_diags: tuple[Diagnostic, ...] = (),
) -> dict[str, Any]:
    """Stub plan for the ``error`` status path."""
    window = director_brief.get("window") or {}
    try:
        w_start = float(window.get("start_s", 0.0)) if isinstance(window, Mapping) else 0.0
        w_end = float(window.get("end_s", 0.0)) if isinstance(window, Mapping) else 0.0
    except (TypeError, ValueError):
        w_start = 0.0
        w_end = 0.0
    if not math.isfinite(w_start) or not math.isfinite(w_end):
        w_start, w_end = 0.0, 0.0

    diag = tuple(validation.diagnostics) + extra_diags
    return {
        "schema_version": CREATIVE_PLAN_SCHEMA_VERSION,
        "clip_id": clip_id,
        "produced_at_iso": produced_at_iso,
        "source_path": source_path,
        "input_window": {"start_s": w_start, "end_s": w_end},
        "handles_used": {"pre_s": 0.0, "post_s": 0.0},
        "profile": profile,
        "style_lock_id": None,
        "creative_planner_status": "error",
        "creative_goal": {"text": "", "source": "absent", "confidence": 0.0},
        "beat_sheet": [],
        "scene_plans": [],
        "zoom_curve": [],
        "cuts_final": [],
        "mirror_final": [],
        "mirror_decisions": [],
        "timeline_map": [],
        "subtitle_plan": {"events": []},
        "hook_overlay": None,
        "audio_plan": {},
        "color_plan": {},
        "blur_fill_plan": {},
        "blur_fill": {"mode": "off", "source_aspect": ""},
        "hook_plan": {},
        "frame_export_hints": [],
        "manifest_reasoning": [
            "creative_planner_status=error; worker MUST fall back to v2.0/v2.1.",
        ],
        "creative_intensity_score": 0.0,
        "validation": {
            "hard_errors": [d.code for d in diag if d.severity == "hard"],
            "degrade_warnings": [
                d.code for d in diag if d.severity == "degrade"
            ],
            "errors": [
                {"code": d.code, "message": d.message, "path": d.path}
                for d in diag
                if d.severity == "hard"
            ],
            "warnings": [
                {"code": d.code, "message": d.message, "path": d.path}
                for d in diag
                if d.severity == "degrade"
            ],
        },
    }
