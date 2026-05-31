"""edit_plan.json → v2.1 payload bridge (final_spec Part 10).

The :func:`apply_edit_plan_to_payload` function takes an
``edit_plan`` dict (produced by
:func:`bot.uniqueization.planner.creative_planner.run_creative_planner`)
and a v2.1 worker payload, returning a new payload with v6
decisions spliced in.

Design intent — *no new ffmpeg stages*. The renderer adapter only
populates the existing v2.1 payload keys consumed by
:func:`bot.workers.editor_v2._prepare_v21_state`:

* ``cuts`` — derived from ``edit_plan.cuts_final[]`` where
  ``apply=True``. Planner kinds (``filler``/``dead_air``/``repeat``)
  are translated to the worker's
  ``{filler_word, silence_pause, dead_air}`` set.
* ``mirror_clusters`` — derived from ``edit_plan.mirror_final[]``
  where ``apply=True``. Planner emits a *decision list*; the worker
  consumes a flat ``list[int]`` of cluster ids to actually mirror.
* ``hook_emphasis`` — derived from ``edit_plan.hook_plan`` so the
  v2.1 hook-emphasis filtergraph picks the planner's primary_move +
  support choice.
* ``analyzer_v2_payload_version`` — preserved (the v2.1 dispatch
  gate keys on this; absent ⇒ v2.0 byte-equivalence wins).
* ``v6_edit_plan`` — added as a sidecar marker (worker manifest
  picks it up; not consumed by filtergraph).

Pure function. No I/O. Deterministic.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

__all__ = (
    "RendererAdapterResult",
    "apply_edit_plan_to_payload",
    "translate_planner_cut_kind",
    "PLANNER_TO_WORKER_CUT_KIND",
)


# Mirrors :data:`creative_planner._CUT_KIND_PLANNER_TO_WORKER`.
PLANNER_TO_WORKER_CUT_KIND: Mapping[str, str] = {
    "filler": "filler_word",
    "dead_air": "dead_air",
    "repeat": "filler_word",
}


def translate_planner_cut_kind(kind: str) -> str:
    """Translate a planner CUT_KIND_ENUM value to the worker-side enum."""
    return PLANNER_TO_WORKER_CUT_KIND.get(kind, "filler_word")


class RendererAdapterResult:
    """Lightweight container for the adapter's return value.

    ``payload`` is the new payload (deep copy of the input with
    splices applied). ``patched_fields`` lists the top-level payload
    keys that were overwritten / inserted by the adapter, in
    insertion order. ``skipped_reasons`` is a short, deterministic
    set of reasons explaining why a particular splice was a no-op
    (e.g. ``"mirror_no_apply"``).
    """

    __slots__ = ("payload", "patched_fields", "skipped_reasons")

    def __init__(
        self,
        payload: dict[str, Any],
        patched_fields: tuple[str, ...] = (),
        skipped_reasons: tuple[str, ...] = (),
    ) -> None:
        self.payload = payload
        self.patched_fields = patched_fields
        self.skipped_reasons = skipped_reasons


def _planner_cuts_to_payload(
    edit_plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    cuts_final = edit_plan.get("cuts_final") or []
    if not isinstance(cuts_final, list):
        return []
    out: list[dict[str, Any]] = []
    for item in cuts_final:
        if not isinstance(item, Mapping):
            continue
        if not bool(item.get("apply", False)):
            continue
        try:
            start_s = float(item["start_s"])
            end_s = float(item["end_s"])
        except (KeyError, TypeError, ValueError):
            continue
        kind = str(item.get("kind", "filler"))
        out.append(
            {
                "start_s": start_s,
                "end_s": end_s,
                "kind": translate_planner_cut_kind(kind),
                "reason": str(item.get("reason", "")) or "v6_planner",
                "candidate_id": str(item.get("candidate_id", "")),
            }
        )
    return out


def _planner_mirror_to_payload(
    edit_plan: Mapping[str, Any],
) -> list[int]:
    """Flatten planner mirror_final[].apply=True → list[int] cluster_ids."""
    mirror_final = edit_plan.get("mirror_final") or edit_plan.get(
        "mirror_decisions"
    ) or []
    if not isinstance(mirror_final, list):
        return []
    out: list[int] = []
    seen: set[int] = set()
    for item in mirror_final:
        if not isinstance(item, Mapping):
            continue
        if not bool(item.get("apply", False)):
            continue
        try:
            cid = int(item["cluster_id"])
        except (KeyError, TypeError, ValueError):
            continue
        if cid in seen:
            continue
        seen.add(cid)
        out.append(cid)
    return out


def _planner_hook_to_payload(
    edit_plan: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Map ``edit_plan.hook_plan`` → v2.1 ``hook_emphasis`` dict.

    The v2.1 hook-emphasis filter accepts a sparse mapping; we emit
    only the keys we know — ``primary_move`` + ``supports`` —
    plus the planner's reason for traceability.
    """
    hook = edit_plan.get("hook_plan")
    if not isinstance(hook, Mapping) or not hook:
        return None
    primary_move = hook.get("primary_move")
    if not isinstance(primary_move, str) or not primary_move:
        return None
    overlay = hook.get("overlay") if isinstance(hook, Mapping) else None
    out: dict[str, Any] = {
        "primary_move": primary_move,
        "supports": list(hook.get("supports") or ()),
        "reason": str(hook.get("reason", "")),
        "profile": str(hook.get("profile", "")),
    }
    if isinstance(overlay, Mapping):
        out["overlay"] = {
            "text": str(overlay.get("text", "")),
            "start_s": float(overlay.get("start_s", 0.0) or 0.0),
            "end_s": float(overlay.get("end_s", 0.0) or 0.0),
            "y_position": float(overlay.get("y_position", 0.20) or 0.20),
            "suppresses_dialogue_subtitle": bool(
                overlay.get("suppresses_dialogue_subtitle", False)
            ),
        }
    return out


def _planner_subtitle_words(
    edit_plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Translate emphasis_words into v2.1 subtitle_words wire format."""
    subtitle_plan = edit_plan.get("subtitle_plan")
    if not isinstance(subtitle_plan, Mapping):
        return []
    events = subtitle_plan.get("events") or []
    if not isinstance(events, list):
        return []
    out: list[dict[str, Any]] = []
    for ev in events:
        if not isinstance(ev, Mapping):
            continue
        style_id = str(ev.get("style_id", "minimal"))
        if style_id == "off":
            continue
        for w in ev.get("emphasis_words") or ():
            if not isinstance(w, Mapping):
                continue
            try:
                start_s = float(w["start_s"])
                end_s = float(w["end_s"])
            except (KeyError, TypeError, ValueError):
                continue
            out.append(
                {
                    "word": str(w.get("word", "")),
                    "start_s": start_s,
                    "end_s": end_s,
                    "emphasis": True,
                    "style_id": style_id,
                    "semantic_role": str(w.get("semantic_role", "")),
                    "salience": float(w.get("salience", 0.0) or 0.0),
                }
            )
    return out


def apply_edit_plan_to_payload(
    edit_plan: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    override_existing: bool = False,
) -> RendererAdapterResult:
    """Splice ``edit_plan`` decisions into a v2.1 worker payload.

    Returns a fresh dict (deep copy) so callers can compare before/
    after without aliasing. The caller MUST hold the v2.0 byte-
    equivalence guarantee: this adapter ONLY runs when the worker has
    already decided v6 dispatch is active.

    Parameters
    ----------
    edit_plan:
        Mapping produced by :func:`run_creative_planner`. Must have
        ``creative_planner_status`` in ``{"full", "degraded"}``.
        Adapter is a no-op for ``"skipped"`` / ``"error"`` plans.
    payload:
        The original v2.1 payload as supplied to the worker.
    override_existing:
        When True (default False) overwrites analyzer-shipped cuts /
        mirror_clusters / hook_emphasis if the planner produced
        decisions. When False the analyzer's wire payload wins for
        any key the planner does not touch, AND the planner's
        decisions take precedence ONLY for empty / absent keys.

    Returns
    -------
    :class:`RendererAdapterResult`
    """
    status = str(edit_plan.get("creative_planner_status", "skipped"))
    new_payload = deepcopy(dict(payload))

    if status not in {"full", "degraded"}:
        return RendererAdapterResult(
            payload=new_payload,
            patched_fields=(),
            skipped_reasons=(f"creative_planner_status={status}",),
        )

    patched: list[str] = []
    skipped: list[str] = []

    cuts = _planner_cuts_to_payload(edit_plan)
    if cuts:
        existing = new_payload.get("cuts") or []
        if override_existing or not existing:
            new_payload["cuts"] = cuts
            patched.append("cuts")
        else:
            skipped.append("cuts_kept_analyzer_shipped")
    else:
        skipped.append("cuts_no_apply")

    mirror_ids = _planner_mirror_to_payload(edit_plan)
    if mirror_ids:
        existing_mc = new_payload.get("mirror_clusters")
        if override_existing or existing_mc in (None, []):
            new_payload["mirror_clusters"] = mirror_ids
            patched.append("mirror_clusters")
        else:
            skipped.append("mirror_clusters_kept_analyzer_shipped")
    else:
        # Planner declined every cluster — DOWN-stream worker still
        # needs an explicit empty list so the 1.5 s fallback is
        # suppressed (planner is authoritative on §7.6 vetoes).
        if "mirror_clusters" not in new_payload:
            new_payload["mirror_clusters"] = []
            patched.append("mirror_clusters")
        skipped.append("mirror_no_apply")

    hook = _planner_hook_to_payload(edit_plan)
    if hook is not None:
        existing_hook = new_payload.get("hook_emphasis")
        if override_existing or not existing_hook:
            new_payload["hook_emphasis"] = hook
            patched.append("hook_emphasis")
        else:
            skipped.append("hook_emphasis_kept_analyzer_shipped")
    else:
        skipped.append("hook_no_plan")

    subtitle_words = _planner_subtitle_words(edit_plan)
    if subtitle_words:
        existing_sw = new_payload.get("subtitle_words")
        if override_existing or not existing_sw:
            new_payload["subtitle_words"] = subtitle_words
            patched.append("subtitle_words")
        else:
            skipped.append("subtitle_words_kept_analyzer_shipped")

    # Stash the planner sidecar so the manifest can attach it.
    new_payload["v6_edit_plan"] = dict(edit_plan)
    patched.append("v6_edit_plan")

    return RendererAdapterResult(
        payload=new_payload,
        patched_fields=tuple(patched),
        skipped_reasons=tuple(skipped),
    )
