"""Mirror-planner (final_spec §7.6).

Builds one :class:`MirrorDecision` per analyzer-supplied
``mirror_clusters[]`` entry. Three hard rules:

* **All-or-none per cluster** (§7.6.2). One decision per cluster_id,
  covering all scenes in that cluster.
* **Editor veto authority** (§7.6.1). Any analyzer-shipped
  ``veto_signals[]`` triggers ``apply=False`` with the matching
  closed-enum reason. Editor never overrides an analyzer veto.
* **Adjacent cluster consistency** (§7.6.3). When two clusters
  declare ``location_continuity=True`` and the editor would mirror
  one but not the other, both flip to ``apply=False`` with reason
  ``inconsistent_with_adjacent_cluster``.

Default behaviour when no veto signals → editor applies all clusters
(analyzer authoritative absent signal).

Pure function. No I/O.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = (
    "VETO_REASONS",
    "MirrorDecision",
    "MirrorPlan",
    "plan_mirror_final",
)


# §7.6.1 closed enum of veto reasons.
VETO_REASONS: frozenset[str] = frozenset(
    {
        "action_direction_matters",
        "character_looks_specific_side",
        "important_text_or_object_orientation",
        "scene_visually_busy",
        "inconsistent_with_adjacent_cluster",
    }
)


@dataclass(frozen=True, slots=True)
class MirrorDecision:
    """One §4 ``mirror_final[]`` entry.

    ``apply`` is the final editor decision; ``reason`` is one of the
    closed-enum veto reasons (or ``mirror_safe`` when applying).
    ``cluster_id`` matches the analyzer cluster, ``scene_ids`` are
    the scenes covered (for downstream tooling).
    """

    cluster_id: int
    apply: bool
    reason: str
    scene_ids: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class MirrorPlan:
    """Container for :func:`plan_mirror_final` output."""

    decisions: tuple[MirrorDecision, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)


def _normalise_scene_ids(raw: Any) -> tuple[int, ...]:
    if isinstance(raw, list):
        out: list[int] = []
        for s in raw:
            if isinstance(s, int):
                out.append(s)
            else:
                try:
                    out.append(int(s))
                except (TypeError, ValueError):
                    continue
        return tuple(out)
    return ()


def _normalise_veto_signals(raw: Any) -> tuple[str, ...]:
    if isinstance(raw, list):
        return tuple(str(s) for s in raw if isinstance(s, str))
    return ()


def plan_mirror_final(director_brief: Mapping[str, Any]) -> MirrorPlan:
    """§7.6 all-or-none cluster mirror decisions.

    Order in the result matches analyzer ``mirror_clusters[]`` order
    (deterministic, no implicit sort). Each ``cluster_id`` appears
    exactly once. ``mirror_clusters[]`` absent / empty / not-a-list
    → empty plan (the worker will then fall back to the §7.6.5
    1.5s middle window).
    """
    clusters_raw = director_brief.get("mirror_clusters")
    if not isinstance(clusters_raw, list) or not clusters_raw:
        return MirrorPlan()

    decisions: list[MirrorDecision] = []
    warnings: list[str] = []

    # Walk analyzer order. Reject duplicate cluster ids early — by
    # §7.6.2 each cluster_id has exactly one entry.
    seen_ids: set[int] = set()
    raw_by_id: dict[int, Mapping[str, Any]] = {}
    for i, c in enumerate(clusters_raw):
        if not isinstance(c, Mapping):
            warnings.append(f"cluster_{i}_skipped:not_object")
            continue
        try:
            cid = int(c["cluster_id"])
        except (KeyError, TypeError, ValueError):
            warnings.append(f"cluster_{i}_skipped:bad_cluster_id")
            continue
        if cid in seen_ids:
            warnings.append(f"cluster_{cid}_skipped:duplicate_id")
            continue
        seen_ids.add(cid)
        raw_by_id[cid] = c

        scene_ids = _normalise_scene_ids(c.get("scenes", c.get("scene_ids")))
        vetoes = _normalise_veto_signals(c.get("veto_signals"))
        mirror_safe = bool(c.get("mirror_safe", True))

        # Veto cascade: explicit vetoes take precedence over
        # ``mirror_safe``. Otherwise honour ``mirror_safe``.
        apply = True
        reason = "mirror_safe"
        if vetoes:
            # Pick the first veto matching the closed enum so the
            # manifest reads a stable reason.
            for v in vetoes:
                if v in VETO_REASONS:
                    apply = False
                    reason = v
                    break
            else:
                # All vetoes were free-form; record them but still
                # honour the veto signal.
                apply = False
                reason = vetoes[0]
                warnings.append(
                    f"cluster_{cid}:unknown_veto_signal:{vetoes[0]}"
                )
        elif not mirror_safe:
            apply = False
            reason = "mirror_unsafe"

        decisions.append(
            MirrorDecision(
                cluster_id=cid,
                apply=apply,
                reason=reason,
                scene_ids=scene_ids,
            )
        )

    # §7.6.3 — adjacent-cluster consistency.
    decisions = _enforce_adjacent_consistency(
        decisions,
        raw_by_id,
        warnings,
    )

    return MirrorPlan(decisions=tuple(decisions), warnings=tuple(warnings))


def _enforce_adjacent_consistency(
    decisions: list[MirrorDecision],
    raw_by_id: dict[int, Mapping[str, Any]],
    warnings: list[str],
) -> list[MirrorDecision]:
    """Apply §7.6.3: location-continuity siblings must agree."""
    if len(decisions) < 2:
        return decisions
    out = list(decisions)
    # Iterate to a fixed point so a chain of three+ continuous
    # clusters collapses to a single agreement set.
    changed = True
    while changed:
        changed = False
        for i in range(1, len(out)):
            cur = out[i]
            prev = out[i - 1]
            cur_raw = raw_by_id.get(cur.cluster_id, {})
            prev_raw = raw_by_id.get(prev.cluster_id, {})
            cur_cont = bool(cur_raw.get("location_continuity", False))
            prev_cont = bool(prev_raw.get("location_continuity", False))
            if not (cur_cont and prev_cont):
                continue
            if cur.apply == prev.apply:
                continue
            # Conflict: one applies, the other doesn't. Force both
            # to apply=False so neither side flips orientation
            # relative to the other.
            if cur.apply:
                out[i] = MirrorDecision(
                    cluster_id=cur.cluster_id,
                    apply=False,
                    reason="inconsistent_with_adjacent_cluster",
                    scene_ids=cur.scene_ids,
                )
            else:
                out[i - 1] = MirrorDecision(
                    cluster_id=prev.cluster_id,
                    apply=False,
                    reason="inconsistent_with_adjacent_cluster",
                    scene_ids=prev.scene_ids,
                )
            warnings.append(
                f"cluster_{cur.cluster_id}:adjacent_consistency_force_no_mirror"
            )
            changed = True
    return out
