"""Hook planner (final_spec §8.5, §8.7).

Builds the §4 ``hook_plan`` block:

* ``hook_overlay`` — optional dual-channel text overlay (§8.5).
  ``{text, start_s, end_s, y_position, suppresses_dialogue_subtitle}``.
* ``primary_move`` — one of {``hook_push_in``, ``hold_wide_reveal``,
  ``match_cut_zoom``} (§8.7).
* ``support`` — exactly **one** of {``brightness_pulse``,
  ``hook_overlay_visual``, ``light_vignette``} (§8.7).
* ``forbidden_effects`` — closed set never to be emitted in hook.

§8.7 budget: ONE primary + ONE support per hook beat. Stacking
raises :class:`BudgetViolation`. Heavy profile carve-out: a second
support is permitted only when both
``hook_text`` is provided AND ``hook_confidence ≥ 0.7``; otherwise
the second support raises.

§8.5 dual-channel mutual exclusion: if
``dialogue_density_in_hook_beat`` exceeds the threshold (default
12 chars/sec) the planner SKIPS the overlay and KEEPS dialogue
subtitle. Otherwise the planner renders the overlay AND signals
``suppresses_dialogue_subtitle=True`` for the hook beat.

Pure function. Deterministic. **No LLM** — text is verbatim from
``director_brief.hook_text`` (§12.2 ``manifest_reasoning`` LLM is
done elsewhere).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .beat_sheet import Beat
from .locks import StyleLock

__all__ = (
    "HOOK_PRIMARY_ENUM",
    "HOOK_SUPPORT_ENUM",
    "HOOK_FORBIDDEN_DEFAULTS",
    "DEFAULT_HOOK_DIALOGUE_DENSITY_THRESHOLD",
    "DEFAULT_HOOK_OVERLAY_Y",
    "BudgetViolation",
    "HookOverlay",
    "HookPlan",
    "plan_hook",
)


HOOK_PRIMARY_ENUM: tuple[str, ...] = (
    "hook_push_in",
    "hold_wide_reveal",
    "match_cut_zoom",
)

HOOK_SUPPORT_ENUM: tuple[str, ...] = (
    "brightness_pulse",
    "hook_overlay_visual",
    "light_vignette",
)

# §8.7 forbidden across all profiles.
HOOK_FORBIDDEN_DEFAULTS: tuple[str, ...] = (
    "shake",
    "jitter",
    "flash",
    "stacked_supports",
)


# §8.5 default density threshold for overlay suppression decision.
DEFAULT_HOOK_DIALOGUE_DENSITY_THRESHOLD: float = 12.0  # chars / second
DEFAULT_HOOK_OVERLAY_Y: float = 0.20  # top-third per §8.5

# §8.5 visible window for the overlay.
HOOK_OVERLAY_MIN_S: float = 1.5
HOOK_OVERLAY_MAX_S: float = 3.0


class BudgetViolation(RuntimeError):
    """Raised when a hook treatment exceeds the §8.7 budget."""


@dataclass(frozen=True, slots=True)
class HookOverlay:
    """One §8.5 dual-channel overlay decision."""

    text: str
    start_s: float
    end_s: float
    y_position: float
    suppresses_dialogue_subtitle: bool
    reason: str = ""


@dataclass(frozen=True, slots=True)
class HookPlan:
    """Container for :func:`plan_hook` output."""

    primary_move: str = "hook_push_in"
    supports: tuple[str, ...] = ()
    overlay: HookOverlay | None = None
    forbidden_effects: tuple[str, ...] = HOOK_FORBIDDEN_DEFAULTS
    profile: str = "medium"
    lock_id: str | None = None
    reason: str = ""
    warnings: tuple[str, ...] = field(default_factory=tuple)


# ---- helpers -------------------------------------------------------


def _first_hook_beat(beat_sheet: tuple[Beat, ...]) -> Beat | None:
    for b in beat_sheet:
        if b.purpose == "hook":
            return b
    if beat_sheet:
        return beat_sheet[0]
    return None


def _hook_overlay_window(beat: Beat) -> tuple[float, float]:
    """Pick a 1.5 – 3.0 s sub-window starting at the beat start."""
    dur = beat.duration_s
    if dur <= HOOK_OVERLAY_MIN_S:
        return beat.start_s, beat.end_s
    end = beat.start_s + min(dur, HOOK_OVERLAY_MAX_S)
    return beat.start_s, end


def _resolve_primary(
    beat: Beat,
    director_brief: Mapping[str, Any],
) -> str:
    """§8.7 — pick primary move. Hook beat already encodes the choice."""
    move = beat.primary_move
    if move == "match_cut":
        # Only allowed when analyzer flags match_cut_compatible.
        if bool(director_brief.get("match_cut_compatible", False)):
            return "match_cut_zoom"
        return "hook_push_in"
    if move in HOOK_PRIMARY_ENUM:
        return move
    # Map other planner moves to the closest hook primary.
    if move in ("slow_push_in",):
        return "hook_push_in"
    if move in ("hold", "static"):
        return "hold_wide_reveal"
    return "hook_push_in"


def _dialogue_density_in_hook(
    director_brief: Mapping[str, Any],
    beat: Beat,
) -> float:
    """Compute chars-per-second of transcript text within the hook beat."""
    window = director_brief.get("window") or {}
    if isinstance(window, Mapping):
        try:
            window_start_s = float(window.get("start_s", 0.0))
        except (TypeError, ValueError):
            window_start_s = 0.0
    else:
        window_start_s = 0.0

    segs = director_brief.get("transcript_segments") or []
    if not isinstance(segs, list):
        return 0.0
    total_chars = 0
    for s in segs:
        if not isinstance(s, Mapping):
            continue
        try:
            ss = float(s["start_s"]) - window_start_s
            se = float(s["end_s"]) - window_start_s
            text = str(s.get("text", ""))
        except (KeyError, TypeError, ValueError):
            continue
        # Compute overlap with beat window.
        overlap_start = max(ss, beat.start_s)
        overlap_end = min(se, beat.end_s)
        if overlap_end <= overlap_start:
            continue
        seg_dur = max(1e-6, se - ss)
        ratio = (overlap_end - overlap_start) / seg_dur
        total_chars += int(len(text) * ratio)
    dur = max(1e-6, beat.duration_s)
    return total_chars / dur


def _is_heavy_carveout_eligible(
    director_brief: Mapping[str, Any],
    profile: str,
    hook_text: str,
) -> bool:
    """§8.7 — heavy + hook_text + hook_confidence ≥ 0.7."""
    if profile != "heavy":
        return False
    if not hook_text:
        return False
    try:
        conf = float(director_brief.get("hook_confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    return conf >= 0.7


def _select_support(
    *,
    has_overlay: bool,
    profile: str,
) -> str:
    """Pick exactly one §8.7 support effect."""
    if has_overlay:
        return "hook_overlay_visual"
    if profile == "light":
        # Light profile: only brightness pulse if any subtitle hook present.
        return "brightness_pulse"
    return "brightness_pulse"


# ---- planner -------------------------------------------------------


def plan_hook(
    beat_sheet: tuple[Beat, ...],
    director_brief: Mapping[str, Any],
    profile: str = "medium",
    lock: StyleLock | None = None,
) -> HookPlan:
    """§8.5 + §8.7 hook-treatment plan.

    Raises :class:`BudgetViolation` if a caller forces more than ONE
    support (or two without the §8.7 heavy carve-out).
    """
    lock_id = lock.lock_id if lock is not None else None
    forbidden = HOOK_FORBIDDEN_DEFAULTS
    if lock is not None and lock.forbidden_effects:
        # Lock can add more forbidden effects; never remove.
        forbidden = tuple(
            sorted(set(forbidden) | set(lock.forbidden_effects))
        )

    if not beat_sheet:
        return HookPlan(
            primary_move="hook_push_in",
            supports=(),
            overlay=None,
            forbidden_effects=forbidden,
            profile=profile,
            lock_id=lock_id,
            reason="no_beats",
        )

    beat = _first_hook_beat(beat_sheet)
    if beat is None:
        return HookPlan(
            primary_move="hook_push_in",
            supports=(),
            overlay=None,
            forbidden_effects=forbidden,
            profile=profile,
            lock_id=lock_id,
            reason="no_hook_beat",
        )

    primary = _resolve_primary(beat, director_brief)
    if primary not in HOOK_PRIMARY_ENUM:
        raise BudgetViolation(
            f"primary_move {primary!r} not in HOOK_PRIMARY_ENUM"
        )

    raw_text = director_brief.get("hook_text", "")
    hook_text = str(raw_text).strip() if isinstance(raw_text, str) else ""

    threshold = float(
        director_brief.get(
            "hook_dialogue_density_threshold",
            DEFAULT_HOOK_DIALOGUE_DENSITY_THRESHOLD,
        )
    )
    density = _dialogue_density_in_hook(director_brief, beat)
    has_overlay = bool(hook_text)
    overlay: HookOverlay | None = None

    if has_overlay:
        # §8.5 mutual exclusion.
        if density > threshold:
            has_overlay = False
            reason = (
                f"density={density:.1f}>{threshold:.1f}_skip_overlay_keep_subs"
            )
        else:
            start_s, end_s = _hook_overlay_window(beat)
            overlay = HookOverlay(
                text=hook_text,
                start_s=start_s,
                end_s=end_s,
                y_position=DEFAULT_HOOK_OVERLAY_Y,
                suppresses_dialogue_subtitle=True,
                reason=f"density={density:.1f}<={threshold:.1f}_render_overlay",
            )
            reason = overlay.reason
    else:
        reason = "no_hook_text"

    # §8.7 budget — pick exactly one support.
    support = _select_support(has_overlay=has_overlay, profile=profile)
    if support not in HOOK_SUPPORT_ENUM:
        raise BudgetViolation(
            f"support {support!r} not in HOOK_SUPPORT_ENUM"
        )
    supports: tuple[str, ...] = (support,)

    # Heavy carve-out — *opt-in only*, manifest-logged. We never opt
    # in by default; callers that want the second support invoke
    # :func:`augment_hook_with_extra_support`.

    if (
        "stacked_supports" in forbidden
        and len(supports) > 1
        and not _is_heavy_carveout_eligible(director_brief, profile, hook_text)
    ):
        raise BudgetViolation(
            "stacked supports forbidden outside heavy carve-out"
        )

    plan_reason = f"primary={primary};support={support};{reason}"

    return HookPlan(
        primary_move=primary,
        supports=supports,
        overlay=overlay,
        forbidden_effects=forbidden,
        profile=profile,
        lock_id=lock_id,
        reason=plan_reason,
    )


def augment_hook_with_extra_support(
    plan: HookPlan,
    extra_support: str,
    *,
    director_brief: Mapping[str, Any],
    hook_text: str,
) -> HookPlan:
    """§8.7 heavy carve-out — opt-in second support.

    Raises :class:`BudgetViolation` if the carve-out is not met OR
    if ``extra_support`` is unknown / duplicates the primary support.
    """
    if extra_support not in HOOK_SUPPORT_ENUM:
        raise BudgetViolation(
            f"extra support {extra_support!r} not in HOOK_SUPPORT_ENUM"
        )
    if extra_support in plan.supports:
        raise BudgetViolation(
            f"extra support {extra_support!r} duplicates existing"
        )
    if not _is_heavy_carveout_eligible(
        director_brief, plan.profile, hook_text
    ):
        raise BudgetViolation(
            "heavy carve-out requires profile=heavy + hook_text + "
            "hook_confidence>=0.7"
        )
    supports = tuple(sorted(set(plan.supports) | {extra_support}))
    return HookPlan(
        primary_move=plan.primary_move,
        supports=supports,
        overlay=plan.overlay,
        forbidden_effects=plan.forbidden_effects,
        profile=plan.profile,
        lock_id=plan.lock_id,
        reason=f"{plan.reason};carve_out_extra_support={extra_support}",
        warnings=plan.warnings,
    )
