"""``uniqueization.json`` manifest dataclasses + atomic writer.

Schema is locked at v2.0 baseline (``schema_version: 1``) per
``final_spec_FULL.md`` §9.12. v2.1 (Part 7) extends this in place by
adding a top-level ``unique_distance`` section + ``v21_stages_applied``
list. v6 (Part 10) extends again with a ``creative_planner`` section.
This file owns only the v2.0-shaped core; later parts add fields via
the open-ended ``extra`` mapping until they need a dedicated dataclass.

Atomic write: temp file in the same directory + ``os.replace``. Spec
forbids partial JSON on disk under any failure mode.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ManifestError(RuntimeError):
    """Raised when writing the manifest atomically fails."""


@dataclass(frozen=True, slots=True)
class ManifestStage:
    """One row of the ``stages[]`` table — covers the §9.12 sample."""

    name: str
    enabled: bool
    status: str
    params: Mapping[str, Any] = field(default_factory=dict)
    elapsed_ms: int = 0
    reason: str = ""


@dataclass(frozen=True, slots=True)
class UniqManifest:
    """Top-level ``uniqueization.json`` shape (v2.0 baseline).

    Part 7 (v2.1) adds two named top-level sections via the
    open-ended ``extra`` mapping:

    * ``unique_distance``: the §10.7 5-component vector
      (phash / duration / RGB / SSIM / audio) — always present when
      the v2.1 path runs, ``{}``/absent on the v2.0 baseline.
    * ``v21_stages_applied``: ordered list of v2.1 stage names that
      actually ran for this job (``cuts``, ``timeline_remap``,
      ``mirror:cluster_aware``, ``blur_fill``, ``hook_emphasis``,
      ``subtitles``).

    The ``extra`` mapping continues to be the only way to add new
    fields without bumping ``schema_version``, per the §9.12 contract.
    Both keys land at the *top* of the JSON output (see
    :meth:`to_dict`).
    """

    schema_version: int
    editor_version: str
    editor_profile: str
    job: Mapping[str, Any]
    source: Mapping[str, Any]
    output: Mapping[str, Any]
    config_resolved: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    stages: tuple[ManifestStage, ...] = field(default_factory=tuple)
    signature_metrics: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    extra: Mapping[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict ready for ``json.dumps``.

        Tuples become lists (JSON has no tuples) and ``ManifestStage``
        dataclasses are recursively converted. ``extra`` is flattened
        into the top level so v2.1 / v6 callers can publish new
        sections (``unique_distance``, ``v21_stages_applied``,
        ``creative_planner``) without changing this dataclass.
        """
        out: dict[str, Any] = {
            "schema_version": self.schema_version,
            "editor_version": self.editor_version,
            "editor_profile": self.editor_profile,
            "job": dict(self.job),
            "source": dict(self.source),
            "output": dict(self.output),
            "config_resolved": {
                k: dict(v) for k, v in self.config_resolved.items()
            },
            "stages": [asdict(s) for s in self.stages],
            "signature_metrics": dict(self.signature_metrics),
            "warnings": list(self.warnings),
            "created_at": self.created_at or _utc_iso(),
        }
        if self.extra:
            for k, v in self.extra.items():
                if k in out:
                    continue
                out[k] = v
        return out


def build_creative_planner_section(
    *,
    status: str,
    intensity_score: float,
    reasoning: tuple[str, ...] = (),
    patched_payload_fields: tuple[str, ...] = (),
    skipped_reasons: tuple[str, ...] = (),
    edit_plan_path: str = "",
    frame_exports_metadata_path: str = "",
) -> dict[str, Any]:
    """Convenience builder for the v6 ``creative_planner`` extras row.

    Mirrors what :func:`bot.workers.editor_v2._v6_merge_manifest_extras`
    splices into ``UniqManifest.extra``. Exposed at module scope so
    tests / external tools can construct an equivalent dict without
    duplicating field names.
    """
    return {
        "status": status,
        "intensity_score": round(float(intensity_score), 3),
        "reasoning": list(reasoning),
        "patched_payload_fields": list(patched_payload_fields),
        "skipped_reasons": list(skipped_reasons),
        "edit_plan_path": edit_plan_path,
        "frame_exports_metadata_path": frame_exports_metadata_path,
    }


def _utc_iso() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_manifest_atomic(manifest: UniqManifest, path: Path) -> Path:
    """Write ``manifest`` to ``path`` atomically.

    Renders to ``path + ".tmp"`` first, then ``os.replace``. On any
    error the temp file is removed and :class:`ManifestError` is
    raised. Manifest write failure is classified ``abort`` by the
    failure-policy table in §9.13.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    try:
        with temp.open("w", encoding="utf-8") as fh:
            json.dump(manifest.to_dict(), fh, ensure_ascii=False, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
    except OSError as exc:
        with contextlib.suppress(OSError):
            temp.unlink(missing_ok=True)
        raise ManifestError(
            f"failed to write manifest tmp {temp}: {exc}"
        ) from exc
    try:
        os.replace(temp, path)
    except OSError as exc:
        with contextlib.suppress(OSError):
            temp.unlink(missing_ok=True)
        raise ManifestError(
            f"failed to move manifest {temp} -> {path}: {exc}"
        ) from exc
    return path
