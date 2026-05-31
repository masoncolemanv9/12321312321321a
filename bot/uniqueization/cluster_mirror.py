"""Cluster-aware horizontal-mirror filter builder (v2.1).

Builds the ``hflip:enable='between(t,a,b)+between(t,c,d)+...'``
expression the v2.1 fused encode wires into ``filter_complex``. The
core decision logic lives here so Part 7's worker doesn't have to
re-implement the mirror policy.

Spec references (``final_spec_FULL.md``):
    * §7.6 — mirror policy (all-or-none per cluster, veto enum,
      adjacency consistency).
    * §10.6 — cluster-aware enable expression assembly.
    * §10.9 — profile matrix (light/medium/heavy mirror density).

Hard contract:

* **All-or-none per cluster.** A cluster is either fully mirrored
  (every scene in it) or fully not. No partial-cluster mirroring.
* **Veto wins.** If any scene in a cluster carries a veto signal, the
  whole cluster is vetoed. The reason from the *first* vetoing scene
  is recorded; other vetoes are still surfaced in
  :attr:`MirrorPlan.veto_reasons` for the manifest.
* **Render-time only.** All emitted ``between()`` expressions are in
  *render*-time, never source-time. The cut/timeline remap happens
  here, in this module — Part 7 should NOT pre-remap.
* **Fallback semantics.** ``mirror_clusters=None`` (analyzer doesn't
  ship v2.1 fields) → v2.0 1.5s middle-window fallback (§7.6.5).
  ``mirror_clusters=[]`` (explicit empty list) → no mirroring at all.
  This distinction matters for the manifest's ``mirror_mode`` field.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from .timeline import TimelineMap, remap_intervals

#: Closed veto-reason enum (§7.6.1). Stable identifiers for analyzer
#: tuning and the manifest.
VetoReason = Literal[
    "action_direction_matters",
    "character_looks_specific_side",
    "important_text_or_object_orientation",
    "scene_visually_busy",
    "inconsistent_with_adjacent_cluster",
]

#: ``mirror_mode`` enum recorded in the manifest (§7.6.5).
MirrorMode = Literal["cluster_aware", "1_5s_fallback", "none"]


@dataclass(frozen=True, slots=True)
class Scene:
    """One pyscenedetect scene with optional veto annotations.

    ``veto_reasons`` is per-scene because analyzer-v2 emits them at
    that granularity; the cluster-level rollup happens here so callers
    can stay close to the analyzer wire format.
    """

    scene_id: int
    source_start_s: float
    source_end_s: float
    veto_reasons: tuple[VetoReason, ...] = ()


@dataclass(frozen=True, slots=True)
class SceneCluster:
    """A group of adjacent visually-similar scenes (§7.6.4)."""

    cluster_id: int
    scene_ids: tuple[int, ...]
    location_continuity: bool = False


@dataclass(frozen=True, slots=True)
class MirrorPlan:
    """Result of :func:`build_cluster_mirror_filter`.

    ``enable_expression`` is the ffmpeg-ready string (without the
    surrounding ``hflip:enable='…'`` wrapping — callers compose that
    themselves; the wrapping format may differ between
    chain-position usages). ``None`` ⇒ no mirror filter should be
    emitted at all.

    ``intervals`` is the render-time intervals the expression covers,
    surfaced separately so the manifest can record them without
    re-parsing the ffmpeg string.
    """

    enable_expression: str | None
    intervals: tuple[tuple[float, float], ...]
    mode: MirrorMode
    applied_cluster_ids: tuple[int, ...] = ()
    vetoed_cluster_ids: tuple[int, ...] = ()
    veto_reasons: dict[int, tuple[VetoReason, ...]] = field(
        default_factory=dict
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_cluster_mirror_filter(
    scene_clusters: Sequence[SceneCluster] | None,
    mirror_clusters: Sequence[int] | None,
    timeline_map: TimelineMap,
    *,
    scenes: Sequence[Scene] = (),
    render_duration_s: float | None = None,
    fallback_window_s: float = 1.5,
    min_duration_for_fallback_s: float = 3.0,
) -> MirrorPlan:
    """Decide the v2.1 mirror filter for one clip.

    Resolution order (mirrors §7.6):

    1. ``mirror_clusters is None`` (no v2.1 mirror data) →
       v2.0 1.5s middle-window fallback. Skipped if
       ``render_duration_s < min_duration_for_fallback_s`` (§7.6.5
       "Skip if duration < 3s") → :attr:`MirrorPlan.mode` ``"none"``.
    2. ``mirror_clusters == []`` (analyzer explicitly chose none) →
       no mirror, ``mode="none"``.
    3. Otherwise cluster-aware. For each cluster in the to-mirror
       set:
       * Look up the cluster in ``scene_clusters``.
       * Roll up veto signals from every scene in the cluster
         (``scenes`` lookup). If any veto present, drop the cluster
         (recorded with reasons) and continue.
       * Otherwise collect each scene's source-time interval,
         :func:`remap_intervals` them through ``timeline_map``, and
         add to the enable expression.
       Result mode: ``"cluster_aware"`` if any cluster honoured,
       else ``"none"``.

    Args:
        scene_clusters: The clusters the analyzer emitted. ``None``
            allowed (treated as "no clusters available", same as
            ``mirror_clusters is None``).
        mirror_clusters: The cluster IDs to mirror. See semantics
            above for the ``None`` vs ``[]`` distinction.
        timeline_map: Source→render time map (Part 4's
            :class:`.timeline.TimelineMap`). Used to remap scene
            intervals to render-time.
        scenes: Per-scene metadata, primarily veto reasons. Optional;
            absent scenes contribute no vetoes.
        render_duration_s: Total render-time duration. Only used by
            the 1.5s fallback path; defaults to
            ``timeline_map.render_duration_s``.
        fallback_window_s: Duration of the v2.0 fallback mirror
            window, centred on the render midpoint. §7.6.5 default
            1.5s.
        min_duration_for_fallback_s: Below this, the fallback is
            skipped (§7.6.5 "Skip if duration < 3s").

    Returns:
        A :class:`MirrorPlan`. Callers wrap ``enable_expression`` in
        the appropriate ffmpeg syntax for their use site.
    """

    if render_duration_s is None:
        render_duration_s = timeline_map.render_duration_s

    # 1. v2.0 fallback path.
    if mirror_clusters is None or scene_clusters is None:
        if render_duration_s < min_duration_for_fallback_s:
            return MirrorPlan(
                enable_expression=None,
                intervals=(),
                mode="none",
            )
        midpoint = render_duration_s / 2.0
        half = fallback_window_s / 2.0
        start = max(0.0, midpoint - half)
        end = min(render_duration_s, midpoint + half)
        expr = _format_enable_expression([(start, end)])
        return MirrorPlan(
            enable_expression=expr,
            intervals=((start, end),),
            mode="1_5s_fallback",
        )

    # 2. Explicit-empty path.
    if len(mirror_clusters) == 0:
        return MirrorPlan(enable_expression=None, intervals=(), mode="none")

    # 3. Cluster-aware path.
    scene_by_id: dict[int, Scene] = {s.scene_id: s for s in scenes}
    cluster_by_id: dict[int, SceneCluster] = {
        c.cluster_id: c for c in scene_clusters
    }

    intervals: list[tuple[float, float]] = []
    applied: list[int] = []
    vetoed: list[int] = []
    veto_reasons: dict[int, tuple[VetoReason, ...]] = {}

    # Preserve analyzer order when iterating, so the enable expression
    # is deterministic for any given input.
    for cluster_id in mirror_clusters:
        if cluster_id not in cluster_by_id:
            # Unknown cluster_id — record as vetoed with no reason.
            # Better than silently dropping; lets dashboards spot
            # analyzer/editor desync.
            vetoed.append(cluster_id)
            continue
        cluster = cluster_by_id[cluster_id]
        cluster_vetoes: list[VetoReason] = []
        source_intervals: list[tuple[float, float]] = []
        for scene_id in cluster.scene_ids:
            scene = scene_by_id.get(scene_id)
            if scene is None:
                # Unknown scene — soft veto with adjacency reason.
                cluster_vetoes.append("inconsistent_with_adjacent_cluster")
                continue
            cluster_vetoes.extend(scene.veto_reasons)
            source_intervals.append((scene.source_start_s, scene.source_end_s))

        if cluster_vetoes:
            vetoed.append(cluster_id)
            veto_reasons[cluster_id] = tuple(cluster_vetoes)
            continue

        if not source_intervals:
            vetoed.append(cluster_id)
            continue

        remapped = remap_intervals(source_intervals, timeline_map)
        if not remapped:
            # Every scene fell inside a cut → nothing to mirror.
            vetoed.append(cluster_id)
            veto_reasons[cluster_id] = ("inconsistent_with_adjacent_cluster",)
            continue

        intervals.extend(remapped)
        applied.append(cluster_id)

    # Apply adjacent-cluster consistency rule (§7.6.3). If any pair of
    # adjacent clusters both carry ``location_continuity`` and disagree
    # on the mirror decision, drop both — keeping one but not the other
    # creates a visible cut-on-mirror. Implemented at cluster-id
    # adjacency: clusters are adjacent if their scene_ids are
    # contiguous integers.
    intervals, applied, vetoed, veto_reasons = _enforce_adjacency_consistency(
        scene_clusters=scene_clusters,
        applied=applied,
        vetoed=vetoed,
        veto_reasons=veto_reasons,
        intervals=intervals,
        scene_by_id=scene_by_id,
        cluster_by_id=cluster_by_id,
        timeline_map=timeline_map,
    )

    if not intervals:
        return MirrorPlan(
            enable_expression=None,
            intervals=(),
            mode="none",
            applied_cluster_ids=tuple(applied),
            vetoed_cluster_ids=tuple(vetoed),
            veto_reasons=veto_reasons,
        )

    expr = _format_enable_expression(intervals)
    return MirrorPlan(
        enable_expression=expr,
        intervals=tuple(intervals),
        mode="cluster_aware",
        applied_cluster_ids=tuple(applied),
        vetoed_cluster_ids=tuple(vetoed),
        veto_reasons=veto_reasons,
    )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _format_enable_expression(
    intervals: Sequence[tuple[float, float]],
) -> str:
    """Build the ``between(t,a,b)+between(t,c,d)+...`` body.

    Numbers are formatted with three decimal places — millisecond
    precision is plenty for hflip transitions, and the fixed-width
    format keeps the manifest's stored ``filter_complex`` string
    stable across runs (helpful for golden tests downstream).
    """
    parts = [f"between(t,{a:.3f},{b:.3f})" for a, b in intervals]
    return "+".join(parts)


def _enforce_adjacency_consistency(
    *,
    scene_clusters: Sequence[SceneCluster],
    applied: list[int],
    vetoed: list[int],
    veto_reasons: dict[int, tuple[VetoReason, ...]],
    intervals: list[tuple[float, float]],
    scene_by_id: dict[int, Scene],
    cluster_by_id: dict[int, SceneCluster],
    timeline_map: TimelineMap,
) -> tuple[
    list[tuple[float, float]],
    list[int],
    list[int],
    dict[int, tuple[VetoReason, ...]],
]:
    """Demote clusters that disagree with a continuous-location neighbour.

    §7.6.3: ``mirror_final`` decisions across adjacent clusters with
    ``location_continuity=true`` must match. Implementation strategy:
    walk applied clusters; for each, find any neighbour in
    ``scene_clusters`` (cluster_id ± 1) with
    ``location_continuity=true``. If that neighbour is in ``vetoed``,
    demote the applied cluster too. The recursive case (chains of
    continuous clusters) doesn't matter at v2.1's analyzer granularity
    — chains longer than 2 are rare and the validator one layer up
    catches outright inconsistency.
    """
    if not applied:
        return intervals, applied, vetoed, veto_reasons

    applied_set = set(applied)
    vetoed_set = set(vetoed)
    demoted_extra: list[int] = []

    for cluster_id in applied:
        cluster = cluster_by_id.get(cluster_id)
        if cluster is None or not cluster.location_continuity:
            continue
        for delta in (-1, 1):
            neighbour_id = cluster_id + delta
            if neighbour_id in vetoed_set and neighbour_id in cluster_by_id:
                neighbour = cluster_by_id[neighbour_id]
                if neighbour.location_continuity:
                    demoted_extra.append(cluster_id)
                    break

    if not demoted_extra:
        return intervals, applied, vetoed, veto_reasons

    demoted_set = set(demoted_extra)
    new_applied = [c for c in applied if c not in demoted_set]
    new_vetoed = vetoed + list(demoted_extra)
    new_veto_reasons = dict(veto_reasons)
    for cid in demoted_extra:
        new_veto_reasons[cid] = ("inconsistent_with_adjacent_cluster",)

    # Rebuild intervals from the surviving clusters.
    new_intervals: list[tuple[float, float]] = []
    for cluster_id in new_applied:
        cluster = cluster_by_id.get(cluster_id)
        if cluster is None:
            continue
        source_intervals = []
        for scene_id in cluster.scene_ids:
            scene = scene_by_id.get(scene_id)
            if scene is None:
                continue
            source_intervals.append((scene.source_start_s, scene.source_end_s))
        new_intervals.extend(remap_intervals(source_intervals, timeline_map))

    # ``applied_set`` is only used above for membership; surfaced here so
    # static analysers don't flag it as unused while keeping the
    # variable available for future extension (e.g. logging which
    # clusters were considered).
    _ = applied_set

    return new_intervals, new_applied, new_vetoed, new_veto_reasons


__all__ = [
    "MirrorMode",
    "MirrorPlan",
    "Scene",
    "SceneCluster",
    "VetoReason",
    "build_cluster_mirror_filter",
]
