"""Typed-policy validation for ``director_brief.json`` (final_spec §12.4).

Two-level classification (§12.4):

* **HARD** — error: ``creative_planner_status: error``. The brief is
  structurally broken or the schema MAJOR mismatches. The worker
  must abort and let the v2.0 / v2.1 fallback take over.
* **DEGRADE** — warning: ``creative_planner_status: degraded``.
  An optional capability is missing or an unknown field appeared.
  Planners fall back at the field level (§12.5).
* **OK** — no diagnostics: ``creative_planner_status: full``.

This module is pure: it never reads files, never logs to stdout. It
ingests a parsed dict (whatever the worker loaded from
``director_brief.json``) and emits a :class:`ValidationResult`.

The classification table is reified in :data:`RULES` so tests can
walk the schema mechanically (one fixture per branch) per §14.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, TypeGuard

__all__ = (
    "Severity",
    "Diagnostic",
    "ValidationResult",
    "REQUIRED_FLOOR_FIELDS",
    "REQUIRED_WINDOW_FIELDS",
    "REQUIRED_SCENE_FIELDS",
    "REQUIRED_TRANSCRIPT_SEGMENT_FIELDS",
    "OPTIONAL_CAPABILITIES",
    "validate_director_brief",
    "supported_schema_major",
)


# ---- public types --------------------------------------------------


Severity = str  # "hard" | "degrade" | "info"


@dataclass(frozen=True, slots=True)
class Diagnostic:
    """One validation finding.

    ``code`` is a stable kebab/snake identifier intended for
    aggregating in manifest reasoning; ``message`` is human-readable
    free text bounded by §12.3 (≤80 chars for per-decision reasons,
    we keep ≤200 here to allow context for hard failures). ``path``
    is a JSON-pointer-ish dotted breadcrumb so consumers can highlight
    the offending field in tooling.
    """

    severity: str  # Severity
    code: str
    message: str
    path: str = ""


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Outcome of :func:`validate_director_brief`.

    ``creative_planner_status`` maps directly to the §12.5 taxonomy:

    * ``"error"`` — at least one HARD finding;
    * ``"degraded"`` — only DEGRADE findings;
    * ``"full"`` — no findings.

    A ``"skipped"`` status is *not* this validator's call — that path
    is taken upstream when no ``director_brief.json`` exists at all,
    before validation runs.
    """

    creative_planner_status: str
    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)

    @property
    def is_error(self) -> bool:
        return self.creative_planner_status == "error"

    @property
    def is_degraded(self) -> bool:
        return self.creative_planner_status == "degraded"

    @property
    def hard_errors(self) -> tuple[Diagnostic, ...]:
        return tuple(d for d in self.diagnostics if d.severity == "hard")

    @property
    def warnings(self) -> tuple[Diagnostic, ...]:
        return tuple(d for d in self.diagnostics if d.severity == "degrade")


# ---- floor schema --------------------------------------------------


# §3.2 minimum required floor.
REQUIRED_FLOOR_FIELDS: tuple[str, ...] = (
    "schema_version",
    "capabilities",
    "window",
    "scenes",
    "transcript_segments",
)

REQUIRED_WINDOW_FIELDS: tuple[str, ...] = (
    "start_s",
    "end_s",
)

# handles_pre_s / handles_post_s are optional with default 0.0
# (§7.5). Validator treats their absence as DEGRADE only when the
# `handles` capability is declared.

REQUIRED_SCENE_FIELDS: tuple[str, ...] = ("start_s", "end_s")
REQUIRED_TRANSCRIPT_SEGMENT_FIELDS: tuple[str, ...] = (
    "start_s",
    "end_s",
    "text",
)


# §3.3 closed enum of capability strings the brief may declare.
OPTIONAL_CAPABILITIES: frozenset[str] = frozenset(
    {
        "face_emotion",
        "face_gaze",
        "face_bbox",
        "action_level",
        "object_detection",
        "word_timestamps",
        "silence_segments",
        "mirror_clusters",
        "cuts_candidates",
        "emphasis_words",
        "handles",
        "reorder_license",
        "hook_text",
        "match_cut_compatible",
        "impact_points",
        "action_subdivision_safe",
        "creative_goal_seed",
        "emotional_intent_per_beat",
    }
)


def supported_schema_major() -> int:
    """Editor v6.0 binds to ``director_brief`` MAJOR 1 (§3.4)."""
    return 1


# ---- validation core -----------------------------------------------


def _is_float(value: Any) -> TypeGuard[float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _add(diags: list[Diagnostic], severity: str, code: str, msg: str, path: str = "") -> None:
    diags.append(Diagnostic(severity=severity, code=code, message=msg, path=path))


def _validate_schema_version(brief: Mapping[str, Any], diags: list[Diagnostic]) -> None:
    version = brief.get("schema_version")
    if not isinstance(version, str) or not version:
        _add(
            diags,
            "hard",
            "schema_version_missing",
            "director_brief.schema_version must be a non-empty string",
            "schema_version",
        )
        return
    parts = version.split(".")
    try:
        major = int(parts[0])
    except (ValueError, IndexError):
        _add(
            diags,
            "hard",
            "schema_version_unparseable",
            f"director_brief.schema_version '{version}' is not semver",
            "schema_version",
        )
        return
    if major != supported_schema_major():
        _add(
            diags,
            "hard",
            "schema_version_major_mismatch",
            (
                f"director_brief.schema_version MAJOR={major} but editor "
                f"binds to MAJOR={supported_schema_major()} (§3.4)"
            ),
            "schema_version",
        )


def _validate_window(brief: Mapping[str, Any], diags: list[Diagnostic]) -> None:
    window = brief.get("window")
    if not isinstance(window, Mapping):
        _add(
            diags,
            "hard",
            "window_missing",
            "director_brief.window is required (§3.2)",
            "window",
        )
        return
    for f in REQUIRED_WINDOW_FIELDS:
        if f not in window:
            _add(
                diags,
                "hard",
                f"window_{f}_missing",
                f"director_brief.window.{f} is required",
                f"window.{f}",
            )
            return
    start = window.get("start_s")
    end = window.get("end_s")
    if not _is_float(start) or not _is_float(end):
        _add(
            diags,
            "hard",
            "window_bounds_type",
            "window.start_s and window.end_s must be numeric",
            "window",
        )
        return
    if float(start) < 0 or float(end) < 0:
        _add(
            diags,
            "hard",
            "window_negative",
            "window bounds must be non-negative",
            "window",
        )
        return
    if float(start) >= float(end):
        _add(
            diags,
            "hard",
            "window_inverted",
            f"window.start_s ({start}) must be < window.end_s ({end})",
            "window",
        )

    handles_pre = window.get("handles_pre_s")
    handles_post = window.get("handles_post_s")
    if handles_pre is not None and not _is_float(handles_pre):
        _add(
            diags,
            "hard",
            "handles_pre_type",
            "window.handles_pre_s must be numeric when set",
            "window.handles_pre_s",
        )
    if handles_post is not None and not _is_float(handles_post):
        _add(
            diags,
            "hard",
            "handles_post_type",
            "window.handles_post_s must be numeric when set",
            "window.handles_post_s",
        )


def _validate_scenes(brief: Mapping[str, Any], diags: list[Diagnostic]) -> None:
    scenes = brief.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        _add(
            diags,
            "hard",
            "scenes_missing",
            "director_brief.scenes must be a non-empty list (§3.2)",
            "scenes",
        )
        return
    window = brief.get("window") or {}
    w_start = float(window.get("start_s", 0.0)) if isinstance(window, Mapping) else 0.0
    w_end = float(window.get("end_s", 0.0)) if isinstance(window, Mapping) else 0.0
    prev_end: float | None = None
    for i, scene in enumerate(scenes):
        if not isinstance(scene, Mapping):
            _add(
                diags,
                "hard",
                "scene_type",
                f"scenes[{i}] is not an object",
                f"scenes[{i}]",
            )
            continue
        for f in REQUIRED_SCENE_FIELDS:
            if f not in scene:
                _add(
                    diags,
                    "hard",
                    f"scene_{f}_missing",
                    f"scenes[{i}].{f} is required",
                    f"scenes[{i}].{f}",
                )
                continue
        s = scene.get("start_s")
        e = scene.get("end_s")
        if not _is_float(s) or not _is_float(e):
            continue
        s_f = float(s)
        e_f = float(e)
        if s_f >= e_f:
            _add(
                diags,
                "hard",
                "scene_inverted",
                f"scenes[{i}] start_s >= end_s",
                f"scenes[{i}]",
            )
            continue
        if w_end > 0 and (s_f < w_start - 1e-6 or e_f > w_end + 1e-6):
            _add(
                diags,
                "hard",
                "scene_out_of_window",
                f"scenes[{i}] [{s_f},{e_f}] outside window [{w_start},{w_end}]",
                f"scenes[{i}]",
            )
        if prev_end is not None and s_f < prev_end - 1e-6:
            _add(
                diags,
                "hard",
                "scene_overlap",
                f"scenes[{i}] starts before previous scene ended",
                f"scenes[{i}]",
            )
        prev_end = e_f


def _validate_transcript(brief: Mapping[str, Any], diags: list[Diagnostic]) -> None:
    segs = brief.get("transcript_segments")
    if not isinstance(segs, list) or not segs:
        _add(
            diags,
            "hard",
            "transcript_segments_missing",
            "director_brief.transcript_segments must be a non-empty list (§3.2)",
            "transcript_segments",
        )
        return
    for i, seg in enumerate(segs):
        if not isinstance(seg, Mapping):
            _add(
                diags,
                "hard",
                "transcript_segment_type",
                f"transcript_segments[{i}] is not an object",
                f"transcript_segments[{i}]",
            )
            continue
        for f in REQUIRED_TRANSCRIPT_SEGMENT_FIELDS:
            if f not in seg:
                _add(
                    diags,
                    "hard",
                    f"transcript_segment_{f}_missing",
                    f"transcript_segments[{i}].{f} is required",
                    f"transcript_segments[{i}].{f}",
                )


def _validate_capabilities(brief: Mapping[str, Any], diags: list[Diagnostic]) -> None:
    caps_raw = brief.get("capabilities", [])
    if not isinstance(caps_raw, list):
        _add(
            diags,
            "hard",
            "capabilities_type",
            "director_brief.capabilities must be a list",
            "capabilities",
        )
        return
    for i, c in enumerate(caps_raw):
        if not isinstance(c, str):
            _add(
                diags,
                "hard",
                "capabilities_item_type",
                f"capabilities[{i}] is not a string",
                f"capabilities[{i}]",
            )
            continue
        if c not in OPTIONAL_CAPABILITIES:
            # Unknown capability is forward-compat (§12.4 DEGRADE
            # rule: "Unknown JSON fields in director_brief →
            # ignored with WARN").
            _add(
                diags,
                "degrade",
                "capability_unknown",
                f"capabilities[{i}]={c!r} not recognised by editor v6.0",
                f"capabilities[{i}]",
            )


def _validate_cuts(brief: Mapping[str, Any], diags: list[Diagnostic]) -> None:
    cuts = brief.get("cuts")
    if cuts is None:
        return
    if not isinstance(cuts, list):
        _add(
            diags,
            "hard",
            "cuts_type",
            "director_brief.cuts must be a list when present",
            "cuts",
        )
        return
    window = brief.get("window") or {}
    if not isinstance(window, Mapping):
        return
    w_start = float(window.get("start_s", 0.0)) if _is_float(window.get("start_s")) else 0.0
    w_end = float(window.get("end_s", 0.0)) if _is_float(window.get("end_s")) else 0.0
    pre = float(window.get("handles_pre_s", 0.0)) if _is_float(window.get("handles_pre_s")) else 0.0
    post = float(window.get("handles_post_s", 0.0)) if _is_float(window.get("handles_post_s")) else 0.0
    lo = w_start - pre
    hi = w_end + post
    for i, c in enumerate(cuts):
        if not isinstance(c, Mapping):
            _add(
                diags,
                "hard",
                "cut_type",
                f"cuts[{i}] is not an object",
                f"cuts[{i}]",
            )
            continue
        s = c.get("start_s")
        e = c.get("end_s")
        if not _is_float(s) or not _is_float(e):
            _add(
                diags,
                "hard",
                "cut_bounds_type",
                f"cuts[{i}].start_s/end_s must be numeric",
                f"cuts[{i}]",
            )
            continue
        if float(s) >= float(e):
            _add(
                diags,
                "hard",
                "cut_inverted",
                f"cuts[{i}].start_s >= end_s",
                f"cuts[{i}]",
            )
            continue
        if w_end > 0 and (float(s) < lo - 1e-6 or float(e) > hi + 1e-6):
            _add(
                diags,
                "hard",
                "cut_out_of_window",
                f"cuts[{i}] outside window±handles [{lo},{hi}]",
                f"cuts[{i}]",
            )


def _validate_mirror_clusters(brief: Mapping[str, Any], diags: list[Diagnostic]) -> None:
    mc = brief.get("mirror_clusters")
    if mc is None:
        return
    if not isinstance(mc, list):
        _add(
            diags,
            "hard",
            "mirror_clusters_type",
            "director_brief.mirror_clusters must be a list when present",
            "mirror_clusters",
        )
        return
    scene_count = len(brief.get("scenes", []) or [])
    seen_ids: set[int] = set()
    for i, m in enumerate(mc):
        if not isinstance(m, Mapping):
            _add(
                diags,
                "hard",
                "mirror_cluster_type",
                f"mirror_clusters[{i}] is not an object",
                f"mirror_clusters[{i}]",
            )
            continue
        cid = m.get("cluster_id")
        if not isinstance(cid, int):
            _add(
                diags,
                "hard",
                "mirror_cluster_id_missing",
                f"mirror_clusters[{i}].cluster_id must be int",
                f"mirror_clusters[{i}].cluster_id",
            )
            continue
        if cid in seen_ids:
            _add(
                diags,
                "hard",
                "mirror_cluster_id_duplicate",
                f"mirror_clusters[{i}].cluster_id={cid} duplicated",
                f"mirror_clusters[{i}].cluster_id",
            )
            continue
        seen_ids.add(cid)
        scene_ids = m.get("scenes") or m.get("scene_ids") or ()
        if isinstance(scene_ids, list):
            for si in scene_ids:
                if not isinstance(si, int) or si < 0 or si >= scene_count:
                    _add(
                        diags,
                        "hard",
                        "mirror_cluster_scene_ref",
                        f"mirror_clusters[{i}] references invalid scene index {si}",
                        f"mirror_clusters[{i}].scenes",
                    )


def _validate_optional_capabilities(
    brief: Mapping[str, Any], diags: list[Diagnostic]
) -> None:
    """Field-level degrade warnings when capabilities promise fields but they're absent."""
    caps_raw = brief.get("capabilities", [])
    caps: set[str] = (
        {c for c in caps_raw if isinstance(c, str)} if isinstance(caps_raw, list) else set()
    )

    if "handles" in caps:
        window = brief.get("window") or {}
        if isinstance(window, Mapping):
            pre = window.get("handles_pre_s", 0.0)
            post = window.get("handles_post_s", 0.0)
            if (
                _is_float(pre)
                and float(pre) == 0.0
                and _is_float(post)
                and float(post) == 0.0
            ):
                _add(
                    diags,
                    "degrade",
                    "handles_declared_but_zero",
                    "capability 'handles' declared but both handles_pre_s and handles_post_s are 0",
                    "window",
                )

    if "cuts_candidates" in caps and not brief.get("cuts"):
        _add(
            diags,
            "degrade",
            "cuts_candidates_empty",
            "capability 'cuts_candidates' declared but cuts[] empty/absent",
            "cuts",
        )

    if "mirror_clusters" in caps and not brief.get("mirror_clusters"):
        _add(
            diags,
            "degrade",
            "mirror_clusters_empty",
            "capability 'mirror_clusters' declared but mirror_clusters[] empty/absent",
            "mirror_clusters",
        )

    if "emphasis_words" in caps and not brief.get("emphasis_candidates"):
        _add(
            diags,
            "degrade",
            "emphasis_candidates_empty",
            "capability 'emphasis_words' declared but emphasis_candidates[] empty/absent",
            "emphasis_candidates",
        )

    if "word_timestamps" in caps and not brief.get("transcript_words"):
        _add(
            diags,
            "degrade",
            "transcript_words_empty",
            "capability 'word_timestamps' declared but transcript_words[] empty/absent",
            "transcript_words",
        )

    if "silence_segments" in caps and not brief.get("silence_segments"):
        _add(
            diags,
            "degrade",
            "silence_segments_empty",
            "capability 'silence_segments' declared but silence_segments[] empty/absent",
            "silence_segments",
        )

    if "reorder_license" in caps and not brief.get("reorder_license"):
        _add(
            diags,
            "degrade",
            "reorder_license_empty",
            "capability 'reorder_license' declared but reorder_license empty/absent",
            "reorder_license",
        )


# ---- entry point ---------------------------------------------------


def validate_director_brief(brief: Any) -> ValidationResult:
    """Run the §12.4 HARD/DEGRADE pass against ``brief``.

    Returns a :class:`ValidationResult` whose ``creative_planner_status``
    is ready for the manifest. Pure function: same brief → same
    result (no randomness, no I/O).
    """
    if brief is None:
        return ValidationResult(
            creative_planner_status="error",
            diagnostics=(
                Diagnostic(
                    severity="hard",
                    code="director_brief_missing",
                    message="director_brief is None (§12.4)",
                ),
            ),
        )
    if not isinstance(brief, Mapping):
        return ValidationResult(
            creative_planner_status="error",
            diagnostics=(
                Diagnostic(
                    severity="hard",
                    code="director_brief_type",
                    message="director_brief must be a JSON object",
                ),
            ),
        )

    diags: list[Diagnostic] = []

    for f in REQUIRED_FLOOR_FIELDS:
        if f not in brief:
            _add(
                diags,
                "hard",
                f"floor_{f}_missing",
                f"director_brief is missing required floor field '{f}' (§3.2)",
                f,
            )

    _validate_schema_version(brief, diags)
    _validate_capabilities(brief, diags)
    _validate_window(brief, diags)
    _validate_scenes(brief, diags)
    _validate_transcript(brief, diags)
    _validate_cuts(brief, diags)
    _validate_mirror_clusters(brief, diags)
    _validate_optional_capabilities(brief, diags)

    if any(d.severity == "hard" for d in diags):
        status = "error"
    elif any(d.severity == "degrade" for d in diags):
        status = "degraded"
    else:
        status = "full"

    return ValidationResult(creative_planner_status=status, diagnostics=tuple(diags))


# ---- introspection helper -------------------------------------------


def _iter_required_fields() -> Iterable[str]:
    """Convenience for tests: enumerate the floor field set."""
    return iter(REQUIRED_FLOOR_FIELDS)


# Pseudo-namespace used by tests to inspect the rule table without
# importing the private helpers above. Order is locked by §3.2 / §3.3.
RULES: Mapping[str, tuple[str, ...]] = {
    "required_floor": REQUIRED_FLOOR_FIELDS,
    "required_window": REQUIRED_WINDOW_FIELDS,
    "required_scene": REQUIRED_SCENE_FIELDS,
    "required_transcript_segment": REQUIRED_TRANSCRIPT_SEGMENT_FIELDS,
    "optional_capabilities": tuple(sorted(OPTIONAL_CAPABILITIES)),
}
