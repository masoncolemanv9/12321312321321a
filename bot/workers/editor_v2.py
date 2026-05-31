"""``EditorV2Worker`` — v2.0 baseline + v2.1 extension pipeline.

Implements Parts 3 (v2.0 fused ffmpeg) and 7 (v2.1 wiring of cuts /
timeline / cluster-mirror / blur-fill / hook-emphasis / ASS subtitles
/ unique-distance) of the 11-part Editor Agent split. The v1
:class:`bot.workers.editor.EditorWorker` is left untouched — this class
only runs when ``EDITOR_VERSION`` is set to ``v2`` / ``v2.1`` / ``v6``.
Selection happens in :func:`bot.workers.build_editor_worker`.

v2.0 pipeline (§9.5, 16 steps): payload validation → async ffprobe →
YuNet face sampling → -ss / -to seek → vertical transform → face-aware
zoom/reframe → optional redraw → 1.5s mirror → colorgrade → effects →
logo LAST → music remover → audio-fx → loudnorm → atomic encode →
pHash + thumbnail + manifest.

v2.1 dispatch (§10.1, §10.5): triggered when the payload carries
``analyzer_v2_payload_version`` AND ``EDITOR_VERSION`` is ``v2.1`` or
``v6``. Strict execution order:

#. validate v2.1 payload (cross-check kept_segments vs cuts via
   :mod:`bot.uniqueization.timeline` — mismatches abort the job);
#. apply cuts (§7.10) → derive timeline map;
#. resolve cluster-mirror plan (§7.6 / §10.6) — render-time enable
   expression replaces the v2.0 1.5s middle window;
#. blur-fill BEFORE zoom (§8.11) when source is landscape/square;
#. hook-emphasis curve fed into zoom-reframe + brightness pulse
   (§8.6);
#. ASS subtitle burn-in AFTER all visual transforms, BEFORE logo
   (§8.1–§8.3);
#. audio FX + loudnorm (unchanged from v2.0);
#. metrics: pHash + ``unique_distance`` components (§10.7) → manifest
   extras under ``unique_distance`` + ``v21_stages_applied``.

Fallback / regression-zero: every v2.1 field degrades independently;
when zero v2.1 markers are present, the worker dispatches to the v2.0
path and produces byte-identical output. Part 7's regression test
pins this invariant.

Failure policy (§9.13): every optional stage failure → skip-and-warn,
recorded in manifest; final encode failure → one simplified retry,
then abort; payload / probe / final-output / manifest failures →
abort. v2.1 payload validation errors (kept_segments / timeline_map
mismatch) → abort.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ..jobs import Job
from ..uniqueization import (
    AppliedCuts,
    Bbox,
    BlurFillFilter,
    Cut,
    CutsError,
    FaceSampling,
    FilterGraph,
    FilterGraphOptions,
    HookCurve,
    ManifestStage,
    MirrorPlan,
    ProbeError,
    ProbeResult,
    Scene,
    SceneCluster,
    ShippedKeptSegment,
    SubtitleCue,
    SubtitleWord,
    TimedWord,
    TimelineMap,
    TimelineSegment,
    UniqConfig,
    UniqManifest,
    UniqProfile,
    UniqueDistanceComponents,
    UniqueDistanceError,
    apply_cuts,
    build_blur_fill_filter,
    build_cluster_mirror_filter,
    build_hook_curve,
    build_v20_graph,
    chunk_words_into_cues,
    compute_components,
    config_origin,
    derive_timeline_map,
    extract_phash_samples,
    extract_thumbnail,
    hamming_distance,
    identity_map,
    kept_segments_match,
    load_config,
    probe_source,
    remap_words,
    render_ass,
    resolve_profile,
    run_ffmpeg_atomic,
    sample_faces,
    timeline_map_match,
    write_manifest_atomic,
)
from ..uniqueization import safe_area as safe_area_mod
from ..uniqueization.assets import yunet_model_path
from ..uniqueization.config import apply_profile
from ..uniqueization.frame_exports import (
    build_frame_export_argv as _v6_frame_export_argv,
)
from ..uniqueization.frame_exports import (
    select_frame_export_candidates as _v6_frame_export_select,
)
from ..uniqueization.frame_exports import (
    write_frame_exports_metadata as _v6_frame_export_write,
)
from ..uniqueization.phash import PhashSample
from ..uniqueization.planner.creative_planner import (
    CreativePlanResult,
)
from ..uniqueization.planner.creative_planner import (
    run_creative_planner as _v6_run_creative_planner,
)
from ..uniqueization.planner.locks import get_lock as _v6_get_lock
from ..uniqueization.planner.renderer_adapter import (
    apply_edit_plan_to_payload as _v6_apply_edit_plan_to_payload,
)
from ..uniqueization.randomization import (
    RandomizationReport,
    jitter_config,
    make_seed,
)
from ..uniqueization.runner import FfmpegError, FfmpegTimeoutError
from .base import Worker

logger = logging.getLogger(__name__)


# ---- payload + result helpers --------------------------------------


_PAYLOAD_PASSTHROUGH_KEYS: tuple[str, ...] = (
    "language",
    "transcript_path",
    "score",
)


class EditorV2PayloadError(ValueError):
    """Raised when the ``edit`` payload is missing required v2.0 fields.

    Classified ``abort`` in §9.13: re-raises out of :meth:`process`
    untouched so :class:`Worker._handle` records it in ``jobs.error``.
    """


def _validate_payload(payload: dict[str, Any]) -> tuple[Path, float, float, int, str]:
    """Validate and normalise the ``edit`` job payload.

    Accepts both the v1 minimum (source_path, start_s, end_s,
    clip_index, optional hook) and the analyzer-v1 expanded payload
    (adds language, transcript_path, score). v2.1 / v6 extra fields are
    passed through unchanged and ignored here.
    """
    src = payload.get("source_path")
    if not src:
        raise EditorV2PayloadError("edit payload missing 'source_path'")
    source = Path(src)
    if not source.exists():
        raise FileNotFoundError(f"edit source not found: {source}")

    start_s = float(payload.get("start_s") or 0.0)
    end_s = float(payload.get("end_s") or 0.0)
    if end_s <= start_s:
        raise EditorV2PayloadError(
            f"edit payload has non-positive window: {start_s}..{end_s}"
        )

    clip_index = int(payload.get("clip_index", 0))
    hook = str(payload.get("hook") or "")
    return source, start_s, end_s, clip_index, hook


# ---- core ----------------------------------------------------------


class EditorV2Worker(Worker):
    """v2.0 baseline editor — runs only when ``EDITOR_VERSION != "v1"``.

    Subclasses (or future v2.1 / v6 callers) can plug in extra stages
    by overriding :meth:`build_filtergraph_options`, but the public
    contract — payload in, ``dict`` out, manifest beside the clip —
    stays fixed.
    """

    kind = "edit"
    next_kind = "seo"

    async def process(self, job: Job) -> dict[str, Any]:
        payload: dict[str, Any] = dict(job.payload or {})
        source, start_s, end_s, clip_index, hook = _validate_payload(payload)

        cfg = load_config(_per_job_overrides(payload))
        profile = resolve_profile(cfg.editor_profile or "light")
        cfg = apply_profile(cfg, profile)

        # ---- per-job uniqueness jitter (UI "Уникальность" slider) ----
        # When ``cfg.uniqueness_pct > 0`` we vary the four debate-
        # vetted ffmpeg knobs (zoom / mirror_duration_s /
        # effects_opacity / audio_fx_wet) inside the envelope spanned
        # by the three D1+D2-approved profiles (``light → medium →
        # heavy``, ``bot/uniqueization/profiles.py``). Slider values
        # 1..DEBATE_PCT stay inside that envelope; above it (up to
        # ``MAX_UNIQUENESS_PCT = 4 × DEBATE_PCT``) variation may
        # exceed it but never escapes the physical safety bounds.
        # The seed is derived from (job_id, upload_ts, clip_index)
        # so variation is reproducible per job but each new job gets
        # a fresh configuration. At pct=0 this is a no-op
        # (byte-equivalent to the pre-randomization baseline — all
        # golden tests pass unchanged).
        uniqueness_report: RandomizationReport | None = None
        if cfg.uniqueness_pct > 0:
            _seed = make_seed(
                job_id=job.id,
                upload_ts=payload.get("upload_ts")
                or payload.get("created_at"),
                clip_index=clip_index,
            )
            cfg, uniqueness_report = jitter_config(
                cfg,
                uniqueness_pct=cfg.uniqueness_pct,
                seed=_seed,
            )

        clips_dir = source.parent / "clips"
        clip_path = clips_dir / f"{job.id}_{clip_index}.mp4"
        uniq_dir = clips_dir / f"{job.id}_{clip_index}.uniq"
        uniq_dir.mkdir(parents=True, exist_ok=True)

        warnings: list[str] = []
        stages: list[ManifestStage] = []

        # ---- Step 2: probe ----
        t0 = time.monotonic()
        try:
            probe = await probe_source(source)
        except ProbeError as exc:
            raise EditorV2PayloadError(f"ffprobe failed for {source}: {exc}") from exc
        stages.append(
            ManifestStage(
                name="ffprobe",
                enabled=True,
                status="applied",
                params={"duration_s": probe.duration_s, "width": probe.width, "height": probe.height},
                elapsed_ms=_elapsed_ms_since(t0),
            )
        )

        # ---- Step 3: face sampling (optional / warn-on-fail) ----
        face = await asyncio.to_thread(_safe_sample_faces, probe, cfg, warnings, stages)

        # ---- v6 creative-planner dispatch (Part 10 §1.2, §13.1) ----
        # When EDITOR_V6_ENABLED=true AND the payload carries a
        # ``director_brief`` block, run the creative planner. Its
        # ``edit_plan`` decisions are spliced into ``payload`` (no new
        # ffmpeg stages — see :mod:`renderer_adapter`) so the existing
        # v2.1 path below picks them up. Failure / skip falls through
        # silently — the v2.1 / v2.0 path remains the source of truth.
        v6_state: _V6State | None = _maybe_run_v6_planner(
            payload=payload,
            cfg=cfg,
            probe=probe,
            source=source,
            start_s=start_s,
            end_s=end_s,
            warnings=warnings,
            stages=stages,
        )
        if v6_state is not None:
            payload = v6_state.patched_payload

        # ---- v2.1 detection (Part 7 §10.1) ----
        # When the payload carries the analyzer_v2 marker AND the editor
        # version is v2.1 / v6, run the v2.1 extension path. Absence of
        # the marker means strict v2.0 (§10.1 byte-equivalence rule).
        v21_active = _is_v21_active(payload, cfg)
        v21_state: _V21State | None = None

        # ---- Steps 4..14: build filtergraph ----
        options = self.build_filtergraph_options(payload, cfg)
        effective_cfg = cfg

        if v21_active:
            v21_state = _prepare_v21_state(
                payload=payload,
                cfg=cfg,
                profile=profile,
                probe=probe,
                source=source,
                start_s=start_s,
                end_s=end_s,
                uniq_dir=uniq_dir,
                warnings=warnings,
                stages=stages,
            )
            effective_cfg, options = _apply_v21_overrides(
                cfg=cfg,
                options=options,
                state=v21_state,
            )

        graph = build_v20_graph(
            probe,
            face,
            effective_cfg,
            profile,
            source_path=source,
            start_s=start_s,
            end_s=end_s,
            options=options,
        )

        if v21_state is not None:
            graph = _splice_v21_filters(
                graph=graph,
                cfg=effective_cfg,
                state=v21_state,
                probe=probe,
                start_s=start_s,
                end_s=end_s,
                stages=stages,
                warnings=warnings,
            )

        warnings.extend(graph.warnings)
        for name in graph.stages_applied:
            stages.append(
                ManifestStage(
                    name=name,
                    enabled=True,
                    status="applied",
                    params={},
                    elapsed_ms=0,
                )
            )

        # ---- Step 15: atomic encode (with one simplified retry on failure) ----
        encode_started = time.monotonic()
        try:
            encode_result = await asyncio.to_thread(
                _run_encode,
                graph=graph,
                cfg=cfg,
                clip_path=clip_path,
                uniq_dir=uniq_dir,
                start_s=start_s,
                end_s=end_s,
            )
            encode_stage_name = "final_encode"
        except (FfmpegError, FfmpegTimeoutError) as exc:
            logger.warning("v2.0 final encode failed: %s; retrying simplified", exc)
            warnings.append("final_encode_failed_retry_simplified")
            stages.append(
                ManifestStage(
                    name="final_encode",
                    enabled=True,
                    status="failed",
                    params={"reason": "primary"},
                    elapsed_ms=_elapsed_ms_since(encode_started),
                    reason=str(exc)[:200],
                )
            )
            simplified_graph = _simplify_graph(
                probe, face, effective_cfg, profile, source, start_s, end_s, options
            )
            encode_started = time.monotonic()
            encode_result = await asyncio.to_thread(
                _run_encode,
                graph=simplified_graph,
                cfg=cfg,
                clip_path=clip_path,
                uniq_dir=uniq_dir,
                start_s=start_s,
                end_s=end_s,
            )
            encode_stage_name = "final_encode_simplified"
            graph = simplified_graph  # so manifest reflects what actually ran
        stages.append(
            ManifestStage(
                name=encode_stage_name,
                enabled=True,
                status="applied",
                params={"argv_len": len(encode_result.argv)},
                elapsed_ms=encode_result.elapsed_ms,
            )
        )

        # ---- Step 16: pHash + thumbnail + manifest ----
        signature_metrics = await asyncio.to_thread(
            _safe_phash, source, clip_path, end_s - start_s, cfg, warnings, stages
        )
        thumbnail_path = await asyncio.to_thread(
            _safe_thumbnail, clip_path, uniq_dir, end_s - start_s, cfg, warnings, stages
        )

        # ---- Step 16b: unique-distance components (v2.1 §10.7) ----
        v21_extras: dict[str, Any] = {}
        if v21_state is not None:
            unique_components = await asyncio.to_thread(
                _safe_unique_distance,
                source,
                clip_path,
                cfg,
                warnings,
                stages,
            )
            v21_extras = _build_v21_extras(
                state=v21_state,
                unique_components=unique_components,
            )

        # ---- v6 sidecar + frame_exports (Part 10 §11.1) ----
        edit_plan_path: Path | None = None
        frame_exports_metadata_path: Path | None = None
        if v6_state is not None:
            edit_plan_path = _v6_write_edit_plan(
                v6_state, uniq_dir, warnings
            )
            frame_exports_metadata_path = await asyncio.to_thread(
                _v6_write_frame_exports,
                v6_state,
                clip_path,
                uniq_dir,
                cfg,
                warnings,
                stages,
            )
            v21_extras = _v6_merge_manifest_extras(
                v6_state=v6_state,
                v21_extras=v21_extras,
                edit_plan_path=edit_plan_path,
                frame_exports_metadata_path=frame_exports_metadata_path,
            )

        manifest = _build_manifest(
            cfg=cfg,
            profile=profile,
            job_id=job.id,
            clip_index=clip_index,
            source=source,
            probe=probe,
            clip_path=clip_path,
            start_s=start_s,
            end_s=end_s,
            stages=stages,
            warnings=warnings,
            signature_metrics=signature_metrics,
            face=face,
            v21_extras=v21_extras,
            uniqueness_report=uniqueness_report,
        )
        manifest_path = uniq_dir / "uniqueization.json"
        try:
            write_manifest_atomic(manifest, manifest_path)
        except Exception:
            logger.exception("v2.0 manifest write failed (ABORT)")
            raise

        return _build_result(
            payload=payload,
            source=source,
            clip_index=clip_index,
            clip_path=clip_path,
            start_s=start_s,
            end_s=end_s,
            hook=hook,
            cfg=cfg,
            graph=graph,
            manifest_path=manifest_path,
            thumbnail_path=thumbnail_path,
            signature_metrics=signature_metrics,
            v21_extras=v21_extras,
            edit_plan_path=edit_plan_path,
            frame_exports_metadata_path=frame_exports_metadata_path,
        )

    # ---- subclass hooks ----------------------------------------------

    def build_filtergraph_options(
        self, payload: dict[str, Any], cfg: UniqConfig
    ) -> FilterGraphOptions:
        """Materialize optional knobs from payload + config.

        Subclasses (v2.1 in Part 7, v6 in Part 10) override to inject
        scene-aware mirror windows, cuts/timeline_map, etc.
        """
        logo_path: Path | None = None
        if cfg.overlay_logo_path:
            logo_path = Path(cfg.overlay_logo_path)

        effects: tuple[Path, ...] = ()
        if cfg.effects_dir:
            effects_dir = Path(cfg.effects_dir)
            if effects_dir.is_dir():
                effects = tuple(sorted(effects_dir.glob("*.mp4")))

        music_intervals_raw = payload.get("music_intervals") or ()
        music_intervals: tuple[tuple[float, float], ...] = tuple(
            (float(a), float(b)) for a, b in music_intervals_raw
        )

        return FilterGraphOptions(
            effects_inputs=effects,
            logo_input=logo_path,
            music_intervals=music_intervals,
        )


# ---- helpers -------------------------------------------------------


def _per_job_overrides(payload: dict[str, Any]) -> dict[str, Any]:
    """Per-job overrides come from ``payload["editor_overrides"]`` when present.

    The dict is passed verbatim into :func:`load_config`; the resolver
    silently ignores unknown keys (so analyzer-v2 / v6 can add new
    fields without breaking forward-compat).
    """
    raw = payload.get("editor_overrides")
    if not isinstance(raw, dict):
        return {}
    return dict(raw)


def _elapsed_ms_since(start_mono: float) -> int:
    return int(max(0.0, (time.monotonic() - start_mono)) * 1000)


def _safe_sample_faces(
    probe: ProbeResult,
    cfg: UniqConfig,
    warnings: list[str],
    stages: list[ManifestStage],
) -> FaceSampling:
    """Run sampled YuNet detection; on any failure return center fallback.

    Classified ``warn`` in §9.13 — face-fallback is the documented
    behaviour, not an error.
    """
    started = time.monotonic()
    model = (
        Path(cfg.yunet_model_path)
        if cfg.yunet_model_path
        else yunet_model_path()
    )
    try:
        result = sample_faces(probe, model_path=model)
    except Exception as exc:  # noqa: BLE001 — sampler must never abort the job
        logger.warning("face sampling raised %s — falling back to center", exc)
        warnings.append(f"face_fallback:{type(exc).__name__}")
        stages.append(
            ManifestStage(
                name="zoom-reframe-agent.sample",
                enabled=True,
                status="failed",
                params={"reason": str(exc)[:200]},
                elapsed_ms=_elapsed_ms_since(started),
            )
        )
        return FaceSampling(
            samples=(),
            median_center_pct=(0.5, 0.5),
            fallback=True,
            reason=type(exc).__name__,
        )

    if result.fallback:
        warnings.append(f"face_fallback:{result.reason}")
    stages.append(
        ManifestStage(
            name="zoom-reframe-agent.sample",
            enabled=True,
            status="applied" if not result.fallback else "fallback",
            params={"fallback": result.fallback, "reason": result.reason},
            elapsed_ms=_elapsed_ms_since(started),
        )
    )
    return result


def _run_encode(
    *,
    graph: FilterGraph,
    cfg: UniqConfig,
    clip_path: Path,
    uniq_dir: Path,
    start_s: float,
    end_s: float,
) -> Any:
    """Compose the final ffmpeg argv and hand it to ``FfmpegRunner``.

    The runner appends the temp output path itself (§9.11 contract),
    so we MUST NOT include ``str(clip_path)`` here.
    """
    argv: list[str] = ["ffmpeg", "-y", "-loglevel", "error"]
    argv.extend(graph.input_args)
    if graph.filter_complex_str:
        argv.extend(["-filter_complex", graph.filter_complex_str])
    v_label = graph.mapped_streams.get("v")
    if v_label:
        argv.extend(["-map", v_label])
    a_label = graph.mapped_streams.get("a")
    if a_label:
        argv.extend(["-map", a_label])
    argv.extend(graph.output_args)

    timeout_s = max(cfg.ffmpeg_timeout_s, (end_s - start_s) * 6.0)
    return run_ffmpeg_atomic(
        argv,
        output_path=clip_path,
        timeout_s=timeout_s,
        debug_dir=uniq_dir,
        keep_artifacts=cfg.keep_uniq_artifacts,
    )


def _simplify_graph(
    probe: ProbeResult,
    face: FaceSampling,
    cfg: UniqConfig,
    profile: UniqProfile,
    source: Path,
    start_s: float,
    end_s: float,
    options: FilterGraphOptions | None,
) -> FilterGraph:
    """Rebuild the graph with all optional / heavy stages dropped.

    Per §9.13 simplified-retry recipe: keep aspect transform + zoom + logo
    (if valid) + basic audio transcode. Drop redraw, mirror, colorgrade,
    effects, music remover, audio-fx, loudnorm by zeroing the relevant
    config knobs before re-running the builder.
    """
    simple_cfg = replace(
        cfg,
        redraw_enabled=False,
        mirror_enabled=False,
        colorgrade_enabled=False,
        effects_opacity=0.0,
        effects_dir="",
        music_remover_enabled=False,
        audio_fx_enabled=False,
        loudnorm_enabled=False,
    )
    # Drop effects + music_intervals from options too — keep logo only.
    simple_options = FilterGraphOptions(
        effects_inputs=(),
        logo_input=options.logo_input if options else None,
        music_intervals=(),
        redraw_enabled_override=False,
        mirror_window=None,
    )
    return build_v20_graph(
        probe,
        face,
        simple_cfg,
        profile,
        source_path=source,
        start_s=start_s,
        end_s=end_s,
        options=simple_options,
    )


def _safe_phash(
    source: Path,
    clip_path: Path,
    clip_duration_s: float,
    cfg: UniqConfig,
    warnings: list[str],
    stages: list[ManifestStage],
) -> dict[str, Any]:
    """Compute source/output pHash distances; warn only on failure (§9.13)."""
    if not cfg.phash_enabled:
        stages.append(
            ManifestStage(
                name="phash",
                enabled=False,
                status="skipped",
                params={"reason": "phash_disabled"},
                elapsed_ms=0,
            )
        )
        return {}

    started = time.monotonic()
    try:
        clip_samples: list[PhashSample] = extract_phash_samples(
            clip_path, duration_s=clip_duration_s
        )
    except Exception as exc:  # noqa: BLE001 — warn-only per spec
        logger.warning("phash sample failed: %s", exc)
        warnings.append(f"phash_skipped:{type(exc).__name__}")
        stages.append(
            ManifestStage(
                name="phash",
                enabled=True,
                status="failed",
                params={"reason": str(exc)[:200]},
                elapsed_ms=_elapsed_ms_since(started),
            )
        )
        return {}

    metrics: dict[str, Any] = {
        "phash_sample_percents": [s.percent for s in clip_samples],
        "phash_distances": [],
        "phash_mean_distance": 0.0,
    }

    # Try to also sample the source clip and compute Hamming distances.
    try:
        src_samples = extract_phash_samples(
            source, duration_s=clip_duration_s
        )
        distances: list[int] = []
        for c, s in zip(clip_samples, src_samples, strict=False):
            distances.append(hamming_distance(c.hash_hex, s.hash_hex))
        metrics["phash_distances"] = distances
        if distances:
            metrics["phash_mean_distance"] = round(
                sum(distances) / len(distances), 2
            )
    except Exception as exc:  # noqa: BLE001 — warn-only
        logger.warning("source phash sample failed: %s", exc)
        warnings.append(f"phash_source_skipped:{type(exc).__name__}")

    stages.append(
        ManifestStage(
            name="phash",
            enabled=True,
            status="applied",
            params={"samples": len(clip_samples)},
            elapsed_ms=_elapsed_ms_since(started),
        )
    )
    return metrics


def _safe_thumbnail(
    clip_path: Path,
    uniq_dir: Path,
    clip_duration_s: float,
    cfg: UniqConfig,
    warnings: list[str],
    stages: list[ManifestStage],
) -> Path | None:
    """Render thumbnail at the clip midpoint; warn-only on failure."""
    if not cfg.thumbnail_enabled:
        stages.append(
            ManifestStage(
                name="thumbnail",
                enabled=False,
                status="skipped",
                params={"reason": "thumbnail_disabled"},
                elapsed_ms=0,
            )
        )
        return None

    started = time.monotonic()
    out = uniq_dir / "thumbnail.jpg"
    try:
        extract_thumbnail(clip_path, out, timestamp_s=max(0.5, clip_duration_s / 2.0))
    except Exception as exc:  # noqa: BLE001 — warn-only per spec
        logger.warning("thumbnail extract failed: %s", exc)
        warnings.append(f"thumbnail_skipped:{type(exc).__name__}")
        stages.append(
            ManifestStage(
                name="thumbnail",
                enabled=True,
                status="failed",
                params={"reason": str(exc)[:200]},
                elapsed_ms=_elapsed_ms_since(started),
            )
        )
        return None

    stages.append(
        ManifestStage(
            name="thumbnail",
            enabled=True,
            status="applied",
            params={"path": str(out)},
            elapsed_ms=_elapsed_ms_since(started),
        )
    )
    return out


def _build_manifest(
    *,
    cfg: UniqConfig,
    profile: UniqProfile,
    job_id: int,
    clip_index: int,
    source: Path,
    probe: ProbeResult,
    clip_path: Path,
    start_s: float,
    end_s: float,
    stages: list[ManifestStage],
    warnings: list[str],
    signature_metrics: dict[str, Any],
    face: FaceSampling,
    v21_extras: dict[str, Any] | None = None,
    uniqueness_report: RandomizationReport | None = None,
) -> UniqManifest:
    """Assemble the §9.12 manifest dataclass."""
    config_resolved: dict[str, dict[str, Any]] = {
        key: {"value": getattr(cfg, key), "source": config_origin(cfg, key)}
        for key in (
            "editor_version",
            "editor_profile",
            "zoom",
            "face_reframe_enabled",
            "mirror_enabled",
            "mirror_duration_s",
            "redraw_enabled",
            "colorgrade_enabled",
            "lut_path",
            "effects_opacity",
            "music_remover_enabled",
            "music_remover_mode",
            "audio_fx_enabled",
            "audio_fx_wet",
            "loudnorm_enabled",
            "video_crf",
            "video_preset",
            "audio_bitrate",
            "uniqueness_pct",
        )
    }
    return UniqManifest(
        schema_version=1,
        editor_version=cfg.editor_version,
        editor_profile=profile.name,
        job={
            "id": job_id,
            "clip_index": clip_index,
            "start_s": start_s,
            "end_s": end_s,
        },
        source={
            "path": str(source),
            "width": probe.width,
            "height": probe.height,
            "duration_s": probe.duration_s,
            "has_audio": probe.has_audio,
        },
        output={
            "clip_path": str(clip_path),
            "width": cfg.output_width,
            "height": cfg.output_height,
            "duration_s": end_s - start_s,
        },
        config_resolved=config_resolved,
        stages=tuple(stages),
        signature_metrics=signature_metrics,
        warnings=tuple(warnings),
        extra=_compose_manifest_extra(
            face, v21_extras or {}, uniqueness_report=uniqueness_report
        ),
    )


def _compose_manifest_extra(
    face: FaceSampling,
    v21_extras: dict[str, Any],
    *,
    uniqueness_report: RandomizationReport | None = None,
) -> dict[str, Any]:
    """Merge the v2.0 ``face_sampling`` block with any v2.1 additions.

    v2.0 callers pass ``v21_extras={}`` and get the legacy single-key
    payload back. v2.1 callers add ``unique_distance`` /
    ``v21_stages_applied`` / per-stage diagnostics. Keeping the
    composition here means the v2.0 manifest is byte-stable when no
    v2.1 fields are wired (§9.12 / §10.7 invariants).
    """
    extra: dict[str, Any] = {
        "face_sampling": {
            "fallback": face.fallback,
            "reason": face.reason,
            "median_center_pct": list(face.median_center_pct),
            "detected_face_count": face.detected_face_count,
        },
    }
    if v21_extras:
        extra.update(v21_extras)
    if uniqueness_report is not None and uniqueness_report.uniqueness_pct > 0:
        extra["uniqueness_randomization"] = {
            "uniqueness_pct": uniqueness_report.uniqueness_pct,
            "variation_pct": uniqueness_report.variation_pct,
            "seed": str(uniqueness_report.seed),
            "zone": uniqueness_report.zone_label,
            "picks": {
                field: dict(info)
                for field, info in uniqueness_report.picks.items()
            },
            "debate_source": (
                "final_spec_FULL.md (D1+D2-approved profile envelope) "
                "+ bot/uniqueization/profiles.py"
            ),
        }
    return extra


def _build_result(
    *,
    payload: dict[str, Any],
    source: Path,
    clip_index: int,
    clip_path: Path,
    start_s: float,
    end_s: float,
    hook: str,
    cfg: UniqConfig,
    graph: FilterGraph,
    manifest_path: Path,
    thumbnail_path: Path | None,
    signature_metrics: dict[str, Any],
    v21_extras: dict[str, Any] | None = None,
    edit_plan_path: Path | None = None,
    frame_exports_metadata_path: Path | None = None,
) -> dict[str, Any]:
    """Compose the §9.3 result payload (v1 keys + v2/v2.1 additions)."""
    overlay_used = any(s.startswith("logo") for s in graph.stages_applied) or (
        "logo_overlay" in graph.stages_applied
    )
    out: dict[str, Any] = {
        "source_path": str(source),
        "clip_index": clip_index,
        "clip_path": str(clip_path),
        "duration_s": end_s - start_s,
        "width": cfg.output_width,
        "height": cfg.output_height,
        "hook": hook,
        "overlay_used": overlay_used,
        "editor_version": cfg.editor_version,
        "editor_profile": cfg.editor_profile,
        "uniqueization_manifest_path": str(manifest_path),
        "thumbnail_path": str(thumbnail_path) if thumbnail_path else "",
        "signature_metrics": signature_metrics,
        "stages_applied": list(graph.stages_applied),
    }
    if v21_extras:
        # Expose only the high-signal v2.1 markers on the result; the
        # full extras dict lives in the manifest.
        applied = v21_extras.get("v21_stages_applied")
        if applied is not None:
            out["v21_stages_applied"] = list(applied)
        ass_path = v21_extras.get("subtitles", {}).get("ass_path")
        if ass_path:
            out["subtitle_ass_path"] = ass_path
        out["analyzer_v2_payload_version"] = v21_extras.get(
            "analyzer_v2_payload_version", ""
        )
    if edit_plan_path is not None:
        out["edit_plan_path"] = str(edit_plan_path)
    if frame_exports_metadata_path is not None:
        out["frame_exports_metadata_path"] = str(
            frame_exports_metadata_path
        )
    for key in _PAYLOAD_PASSTHROUGH_KEYS:
        if key in payload and payload[key] is not None:
            out[key] = payload[key]
    return out


# ---- v2.1 wiring (Part 7) ------------------------------------------
#
# Everything below is the Part 7 plumbing: it imports the Parts 4 / 5 /
# 6 pure utilities, runs the analyzer-shipped v2.1 payload through them,
# and splices the resulting ffmpeg fragments into the v2.0 baseline
# filter_complex emitted by ``build_v20_graph``. v2.0 behaviour is left
# untouched — the dispatcher above only enters this code path when the
# payload carries ``analyzer_v2_payload_version`` AND
# ``EDITOR_VERSION`` is ``v2.1`` / ``v6``.


@dataclass(frozen=True, slots=True)
class _V21State:
    """Materialized v2.1 plan for one render job.

    Pure data carrier — no behaviour. Every field stays present even
    when the corresponding stage is disabled (the worker reads
    ``disabled_reason`` / ``skip_reason`` / ``mode`` to decide whether
    to splice). ``stages_applied`` is the ordered list recorded in the
    manifest under ``v21_stages_applied``.
    """

    analyzer_version: str
    cuts_applied: AppliedCuts | None
    timeline_map: TimelineMap
    mirror_plan: MirrorPlan
    hook_curve: HookCurve
    blur_fill: BlurFillFilter
    subtitle_cues: tuple[SubtitleCue, ...]
    ass_path: Path | None
    placement_decisions: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    stages_applied: tuple[str, ...] = field(default_factory=tuple)
    diagnostics: dict[str, Any] = field(default_factory=dict)


_V21_SUPPORTED_VERSIONS: frozenset[str] = frozenset({"v2.1"})


def _is_v21_active(payload: dict[str, Any], cfg: UniqConfig) -> bool:
    """Dispatch gate: v2.1 path only runs when both conditions hold.

    1. Payload carries the ``analyzer_v2_payload_version`` marker. The
       byte-equivalence rule from §10.1 means a v2.0 payload running
       on a ``v2.1`` worker must still produce v2.0 output, so the
       marker — not just the editor version — drives dispatch.
    2. The editor is built for the v2.1 wire shape (``v2.1`` or
       ``v6``). ``v2``/``v2.0``/``v1`` skip the v2.1 splice even when
       the marker is present, matching the §10.1 "ignore" semantics.
    """
    version = payload.get("analyzer_v2_payload_version")
    if not isinstance(version, str) or not version:
        return False
    return cfg.editor_version in {"v2.1", "v6"}


# ---- v6 dispatch (Part 10) -----------------------------------------
#
# The v6 layer sits *above* the v2.1 path: it runs the creative
# planner against a payload-shipped ``director_brief`` block and
# splices the resulting decisions back into the payload via
# :func:`renderer_adapter.apply_edit_plan_to_payload`. Failure-mode
# taxonomy follows §1.3: ``full`` / ``degraded`` / ``skipped`` /
# ``error``. v6 NEVER adds new ffmpeg stages — it only re-shapes the
# wire payload so the existing v2.1 graph builders pick up planner-
# authored cuts / mirror / hook / subtitles.


@dataclass(frozen=True, slots=True)
class _V6State:
    """Materialized v6 creative-planner state for one render job.

    Pure data carrier. ``edit_plan`` is the §4 sidecar dict;
    ``patched_payload`` is the v2.1 payload with planner decisions
    spliced in (deep copy — the original payload is untouched).
    """

    status: str
    edit_plan: dict[str, Any]
    patched_payload: dict[str, Any]
    patched_fields: tuple[str, ...] = ()
    skipped_reasons: tuple[str, ...] = ()
    intensity_score: float = 0.0
    reasoning: tuple[str, ...] = ()


def _v6_resolve_lock(payload: dict[str, Any], cfg: UniqConfig) -> Any:
    """Pick the §12.6 style lock for this job.

    Precedence: per-job override (``editor_overrides.style_lock_id``)
    > payload-level ``style_lock_id`` > config-level
    ``cfg.style_lock_id`` (when present) > ``None``.
    """
    overrides = payload.get("editor_overrides")
    if isinstance(overrides, dict):
        lid = overrides.get("style_lock_id")
        if isinstance(lid, str) and lid:
            return _v6_get_lock(lid)
    lid = payload.get("style_lock_id")
    if isinstance(lid, str) and lid:
        return _v6_get_lock(lid)
    fallback = getattr(cfg, "style_lock_id", None)
    if isinstance(fallback, str) and fallback:
        return _v6_get_lock(fallback)
    return None


def _v6_resolve_force_skip(payload: dict[str, Any], cfg: UniqConfig) -> bool:
    """Honor §13.1 ``EDITOR_V6_FORCE_SKIP`` and per-job overrides."""
    overrides = payload.get("editor_overrides")
    if isinstance(overrides, dict):
        fs = overrides.get("v6_force_skip")
        if isinstance(fs, bool):
            return fs
    return bool(getattr(cfg, "v6_force_skip", False))


def _maybe_run_v6_planner(
    *,
    payload: dict[str, Any],
    cfg: UniqConfig,
    probe: ProbeResult,
    source: Path,
    start_s: float,
    end_s: float,
    warnings: list[str],
    stages: list[ManifestStage],
) -> _V6State | None:
    """Run the v6 creative planner when enabled and applicable.

    Returns :class:`_V6State` whenever the planner produced a usable
    ``edit_plan`` (``full`` / ``degraded``). Returns ``None`` when v6
    is disabled, the planner was force-skipped, or it returned
    ``skipped`` / ``error``. Either way, the legacy v2.1 / v2.0 path
    below is the source of truth — v6 is purely additive splicing.
    """
    if not bool(getattr(cfg, "v6_enabled", False)):
        return None
    if cfg.editor_version not in {"v2.1", "v6"}:
        return None

    director_brief = payload.get("director_brief")
    if not isinstance(director_brief, dict) or not director_brief:
        return None

    force_skip = _v6_resolve_force_skip(payload, cfg)
    lock = _v6_resolve_lock(payload, cfg)

    t0 = time.monotonic()
    try:
        result: CreativePlanResult = _v6_run_creative_planner(
            director_brief,
            profile=cfg.editor_profile or "light",
            lock=lock,
            force_skip=force_skip,
            source_path=str(source),
            source_aspect_ratio=_safe_source_aspect_ratio(probe),
        )
    except Exception as exc:  # noqa: BLE001 — v6 must never crash the v2.1 path.
        logger.exception("v6 creative_planner crashed; falling through")
        warnings.append(f"v6_creative_planner_crash:{exc.__class__.__name__}")
        stages.append(
            ManifestStage(
                name="creative_planner",
                enabled=True,
                status="failed",
                params={"reason": "crash"},
                elapsed_ms=_elapsed_ms_since(t0),
                reason=str(exc)[:200],
            )
        )
        return None

    stages.append(
        ManifestStage(
            name="creative_planner",
            enabled=True,
            status="applied" if result.is_usable else "skipped",
            params={
                "status": result.status,
                "intensity_score": round(result.intensity_score, 3),
            },
            elapsed_ms=_elapsed_ms_since(t0),
        )
    )

    if not result.is_usable:
        # status in {"skipped", "error"} — v2.1 / v2.0 stays the
        # source of truth. Diagnostics still recorded for the
        # manifest via stages_applied above.
        if result.status == "error":
            warnings.append(
                "v6_creative_planner_error;"
                + ";".join(d.code for d in result.diagnostics[:5])
            )
        return None

    adapter = _v6_apply_edit_plan_to_payload(
        result.edit_plan,
        payload,
        override_existing=False,
    )
    return _V6State(
        status=result.status,
        edit_plan=dict(result.edit_plan),
        patched_payload=adapter.payload,
        patched_fields=adapter.patched_fields,
        skipped_reasons=adapter.skipped_reasons,
        intensity_score=result.intensity_score,
        reasoning=result.reasoning,
    )


def _safe_source_aspect_ratio(probe: ProbeResult) -> float:
    """Pick a best-effort source aspect ratio (W/H) for the planner.

    Used only as a planner hint (it picks blur-fill mode); never
    feeds the v2.1 filter graph directly. Default 0.5625 = 9:16 so
    callers without probe-style metadata still get a coherent plan.
    """
    if probe.height <= 0 or probe.width <= 0:
        return 0.5625
    return probe.width / probe.height


def _v6_write_edit_plan(
    state: _V6State, uniq_dir: Path, warnings: list[str]
) -> Path | None:
    """Write ``edit_plan.json`` sidecar atomically (§4)."""
    import json as _json
    import os as _os
    import tempfile as _tempfile

    out_path = uniq_dir / "edit_plan.json"
    try:
        uniq_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = _tempfile.mkstemp(
            prefix=".edit_plan.", suffix=".tmp", dir=str(uniq_dir)
        )
        with _os.fdopen(fd, "w", encoding="utf-8") as fh:
            _json.dump(state.edit_plan, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        _os.replace(tmp_path, out_path)
        return out_path
    except OSError as exc:
        logger.warning("v6 edit_plan.json write failed: %s", exc)
        warnings.append("v6_edit_plan_write_failed")
        return None


def _v6_write_frame_exports(
    state: _V6State,
    clip_path: Path,
    uniq_dir: Path,
    cfg: UniqConfig,
    warnings: list[str],
    stages: list[ManifestStage],
) -> Path | None:
    """Extract candidate jpegs + write metadata.json (§11.1).

    Runs only when the edit plan has ``frame_export_hints``. The
    extraction is best-effort: a single missing jpeg downgrades the
    stage to ``degraded`` rather than aborting the job.
    """
    candidates = _v6_frame_export_select(state.edit_plan)
    if not candidates:
        return None

    t0 = time.monotonic()
    frames_dir = uniq_dir / "frame_exports"
    frames_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Any] = []
    failures: list[str] = []

    for cand in candidates:
        out_jpg = frames_dir / cand.filename
        argv = _v6_frame_export_argv(
            source_clip=clip_path,
            timestamp_s=cand.source_timestamp_s,
            out_path=out_jpg,
        )
        try:
            run_ffmpeg_atomic(
                argv,
                output_path=out_jpg,
                timeout_s=min(60.0, cfg.ffmpeg_timeout_s),
                debug_dir=uniq_dir,
                keep_artifacts=cfg.keep_uniq_artifacts,
            )
            extracted.append(cand)
        except (FfmpegError, FfmpegTimeoutError, OSError) as exc:
            failures.append(f"{cand.filename}:{exc.__class__.__name__}")

    clip_id = clip_path.stem
    metadata_path: Path | None = None
    try:
        metadata_path = _v6_frame_export_write(
            clip_id=clip_id,
            candidates=extracted,
            out_dir=frames_dir,
        )
    except OSError as exc:
        logger.warning("v6 frame_exports/metadata.json failed: %s", exc)
        warnings.append("v6_frame_exports_metadata_failed")

    status = "applied" if extracted and not failures else (
        "degraded" if extracted else "failed"
    )
    if failures:
        warnings.extend(f"v6_frame_export_failure:{f}" for f in failures)
    stages.append(
        ManifestStage(
            name="v6_frame_exports",
            enabled=True,
            status=status,
            params={
                "requested": len(candidates),
                "extracted": len(extracted),
                "failed": len(failures),
            },
            elapsed_ms=_elapsed_ms_since(t0),
        )
    )
    return metadata_path


def _v6_merge_manifest_extras(
    *,
    v6_state: _V6State,
    v21_extras: dict[str, Any],
    edit_plan_path: Path | None,
    frame_exports_metadata_path: Path | None,
) -> dict[str, Any]:
    """Add a ``creative_planner`` section to the manifest extras."""
    extras = dict(v21_extras)
    extras["creative_planner"] = {
        "status": v6_state.status,
        "intensity_score": round(v6_state.intensity_score, 3),
        "reasoning": list(v6_state.reasoning),
        "patched_payload_fields": list(v6_state.patched_fields),
        "skipped_reasons": list(v6_state.skipped_reasons),
        "edit_plan_path": (
            str(edit_plan_path) if edit_plan_path is not None else ""
        ),
        "frame_exports_metadata_path": (
            str(frame_exports_metadata_path)
            if frame_exports_metadata_path is not None
            else ""
        ),
    }
    return extras


# ---- payload parsers ----


def _parse_cuts(raw: Any) -> list[Cut]:
    """Convert the wire-format cuts array into :class:`Cut` objects."""
    if not isinstance(raw, list):
        return []
    out: list[Cut] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            start = float(item["start_s"])
            end = float(item["end_s"])
        except (KeyError, TypeError, ValueError):
            continue
        kind = str(item.get("kind", "filler_word"))
        if kind not in ("filler_word", "silence_pause", "dead_air"):
            kind = "filler_word"
        out.append(
            Cut(
                start_s=start,
                end_s=end,
                kind=kind,  # type: ignore[arg-type]
                reason=str(item.get("reason", "")),
                confidence=float(item.get("confidence", 1.0) or 1.0),
            )
        )
    return out


def _parse_shipped_kept_segments(raw: Any) -> list[ShippedKeptSegment]:
    """Wire-format kept_segments → typed shipped objects."""
    if not isinstance(raw, list):
        return []
    out: list[ShippedKeptSegment] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            ss = float(item["source_start_s"])
            se = float(item["source_end_s"])
        except (KeyError, TypeError, ValueError):
            continue
        out.append(
            ShippedKeptSegment(
                source_start_s=ss,
                source_end_s=se,
                render_start_s=float(item.get("render_start_s", 0.0) or 0.0),
                render_end_s=float(item.get("render_end_s", 0.0) or 0.0),
            )
        )
    return out


def _parse_timeline_map(raw: Any) -> TimelineMap | None:
    """Wire-format timeline_map → :class:`TimelineMap` or ``None``."""
    if not isinstance(raw, list) or not raw:
        return None
    segs: list[TimelineSegment] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            segs.append(
                TimelineSegment(
                    source_start_s=float(item["source_start_s"]),
                    source_end_s=float(item["source_end_s"]),
                    render_start_s=float(item["render_start_s"]),
                    render_end_s=float(item["render_end_s"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    if not segs:
        return None
    return TimelineMap(segments=tuple(segs))


def _parse_scenes(raw: Any) -> list[Scene]:
    if not isinstance(raw, list):
        return []
    out: list[Scene] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            sid = int(item["scene_id"])
            ss = float(item["source_start_s"])
            se = float(item["source_end_s"])
        except (KeyError, TypeError, ValueError):
            continue
        vetoes_raw = item.get("veto_reasons", ())
        vetoes = (
            tuple(str(v) for v in vetoes_raw)
            if isinstance(vetoes_raw, list)
            else ()
        )
        out.append(
            Scene(
                scene_id=sid,
                source_start_s=ss,
                source_end_s=se,
                veto_reasons=vetoes,  # type: ignore[arg-type]
            )
        )
    return out


def _parse_scene_clusters(raw: Any) -> list[SceneCluster]:
    if not isinstance(raw, list):
        return []
    out: list[SceneCluster] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            cid = int(item["cluster_id"])
            scene_ids = tuple(int(s) for s in item.get("scene_ids", ()))
        except (KeyError, TypeError, ValueError):
            continue
        out.append(
            SceneCluster(
                cluster_id=cid,
                scene_ids=scene_ids,
                location_continuity=bool(
                    item.get("location_continuity", False)
                ),
            )
        )
    return out


def _parse_mirror_clusters(raw: Any) -> list[int] | None:
    """``None`` → 1.5s fallback; ``[]`` → no mirror; ``[ids…]`` → cluster-aware."""
    if raw is None:
        return None
    if not isinstance(raw, list):
        return None
    out: list[int] = []
    for v in raw:
        try:
            out.append(int(v))
        except (TypeError, ValueError):
            continue
    return out


def _parse_subtitle_words(raw: Any) -> list[TimedWord[dict[str, Any]]]:
    if not isinstance(raw, list):
        return []
    out: list[TimedWord[dict[str, Any]]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            ss = float(item["start_s"])
            se = float(item["end_s"])
        except (KeyError, TypeError, ValueError):
            continue
        out.append(
            TimedWord(
                source_start_s=ss,
                source_end_s=se,
                payload={
                    "text": str(item.get("text", "")),
                    "emphasized": bool(item.get("emphasized", False)),
                },
            )
        )
    return out


def _parse_face_bboxes(raw: Any) -> list[Bbox]:
    if not isinstance(raw, list):
        return []
    out: list[Bbox] = []
    for item in raw:
        if isinstance(item, dict):
            try:
                out.append(
                    Bbox(
                        x=float(item["x"]),
                        y=float(item["y"]),
                        w=float(item["w"]),
                        h=float(item["h"]),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
        elif isinstance(item, (list, tuple)) and len(item) == 4:
            try:
                out.append(
                    Bbox(
                        x=float(item[0]),
                        y=float(item[1]),
                        w=float(item[2]),
                        h=float(item[3]),
                    )
                )
            except (TypeError, ValueError):
                continue
    return out


# ---- v2.1 state assembly ----


def _prepare_v21_state(
    *,
    payload: dict[str, Any],
    cfg: UniqConfig,
    profile: UniqProfile,
    probe: ProbeResult,
    source: Path,
    start_s: float,
    end_s: float,
    uniq_dir: Path,
    warnings: list[str],
    stages: list[ManifestStage],
) -> _V21State:
    """Translate the analyzer's v2.1 payload into a render plan.

    The order here mirrors §10.5 strictly: cuts first, then timeline,
    then mirror (which depends on the timeline), then independent
    visuals (blur-fill, hook, subtitles). Validation errors raised by
    :mod:`bot.uniqueization.cuts` / :mod:`.timeline` propagate as
    :class:`EditorV2PayloadError` (§9.13 abort path).
    """
    analyzer_version = str(payload.get("analyzer_v2_payload_version") or "")
    if analyzer_version not in _V21_SUPPORTED_VERSIONS:
        warnings.append(
            f"v21_payload_version_unsupported:{analyzer_version or '<empty>'}"
        )

    stages_applied: list[str] = []
    diagnostics: dict[str, Any] = {"analyzer_v2_payload_version": analyzer_version}

    clip_duration_s = end_s - start_s

    # ---- (1) cuts → timeline_map -----------------------------------
    cuts_input = _parse_cuts(payload.get("cuts"))
    scenes = _parse_scenes(payload.get("scenes"))
    scene_boundaries_raw = payload.get("scene_boundaries_s") or []
    if not scene_boundaries_raw and scenes:
        scene_boundaries_raw = sorted(
            {s.source_start_s for s in scenes} | {s.source_end_s for s in scenes}
        )
    try:
        scene_boundaries_s = tuple(float(x) for x in scene_boundaries_raw)
    except (TypeError, ValueError):
        scene_boundaries_s = ()
    emotional_silence_raw = payload.get("emotional_silence_segments_s") or []
    emotional_silence_segments_s: tuple[tuple[float, float], ...] = tuple(
        (float(a), float(b))
        for a, b in emotional_silence_raw
        if isinstance((a, b), tuple)  # ruff/typecheck tolerant
        or True
    )

    cuts_applied: AppliedCuts | None = None
    if cuts_input:
        t0 = time.monotonic()
        try:
            cuts_applied = apply_cuts(
                source_duration_s=clip_duration_s,
                cuts=cuts_input,
                scene_boundaries_s=scene_boundaries_s,
                emotional_silence_segments_s=emotional_silence_segments_s,
            )
        except CutsError as exc:
            raise EditorV2PayloadError(
                f"validation_error_cuts: {exc}"
            ) from exc

        # Cross-validate against shipped kept_segments if any.
        shipped_kept = _parse_shipped_kept_segments(payload.get("kept_segments"))
        if shipped_kept and not kept_segments_match(
            cuts_applied.kept_segments, shipped_kept
        ):
            raise EditorV2PayloadError(
                "validation_error_kept_segments: derived kept_segments differ "
                "from analyzer-shipped values beyond 1ms tolerance"
            )

        stages_applied.append("cuts")
        stages.append(
            ManifestStage(
                name="cuts",
                enabled=True,
                status="applied",
                params={
                    "honored": len(cuts_applied.kept_segments) - 1
                    if cuts_applied.kept_segments
                    else 0,
                    "skipped": len(cuts_applied.skipped),
                },
                elapsed_ms=_elapsed_ms_since(t0),
            )
        )

    # Derive timeline_map (identity when no cuts).
    if cuts_applied is not None and cuts_applied.kept_segments:
        timeline_map = derive_timeline_map(cuts_applied.kept_segments)
    else:
        timeline_map = identity_map(clip_duration_s)

    shipped_tm = _parse_timeline_map(payload.get("timeline_map"))
    if shipped_tm is not None and not timeline_map_match(
        timeline_map, shipped_tm
    ):
        raise EditorV2PayloadError(
            "validation_error_timeline_map: derived timeline_map differs from "
            "analyzer-shipped values beyond 1ms tolerance"
        )

    if shipped_tm is not None or cuts_applied is not None:
        stages_applied.append("timeline_remap")
        diagnostics["timeline_map_segments"] = len(timeline_map.segments)

    # ---- (2) cluster mirror ----------------------------------------
    scene_clusters = _parse_scene_clusters(payload.get("scene_clusters"))
    mirror_clusters = _parse_mirror_clusters(payload.get("mirror_clusters"))
    render_duration_s = timeline_map.render_duration_s

    t0 = time.monotonic()
    mirror_plan = build_cluster_mirror_filter(
        scene_clusters=scene_clusters or None,
        mirror_clusters=mirror_clusters,
        timeline_map=timeline_map,
        scenes=scenes,
        render_duration_s=render_duration_s,
        fallback_window_s=cfg.mirror_duration_s,
    )
    if mirror_plan.enable_expression is not None:
        stages_applied.append(f"mirror:{mirror_plan.mode}")
    diagnostics["mirror"] = {
        "mode": mirror_plan.mode,
        "intervals": [list(iv) for iv in mirror_plan.intervals],
        "applied_cluster_ids": list(mirror_plan.applied_cluster_ids),
        "vetoed_cluster_ids": list(mirror_plan.vetoed_cluster_ids),
    }
    stages.append(
        ManifestStage(
            name="cluster_mirror",
            enabled=mirror_plan.enable_expression is not None,
            status="applied"
            if mirror_plan.enable_expression is not None
            else "skipped",
            params={"mode": mirror_plan.mode},
            elapsed_ms=_elapsed_ms_since(t0),
        )
    )

    # ---- (3) blur fill ---------------------------------------------
    t0 = time.monotonic()
    if cfg.blur_fill_enabled:
        blur_fill = build_blur_fill_filter(
            source_resolution=(probe.width, probe.height),
            target_resolution=(cfg.output_width, cfg.output_height),
            mode="light_blur",
        )
    else:
        blur_fill = build_blur_fill_filter(
            source_resolution=(probe.width, probe.height),
            target_resolution=(cfg.output_width, cfg.output_height),
            mode="off",
        )
    if blur_fill.skip_reason is None:
        stages_applied.append("blur_fill")
    diagnostics["blur_fill"] = {
        "mode": blur_fill.mode,
        "skip_reason": blur_fill.skip_reason,
        "source_aspect": blur_fill.source_aspect,
    }
    stages.append(
        ManifestStage(
            name="blur_fill",
            enabled=cfg.blur_fill_enabled,
            status="applied" if blur_fill.skip_reason is None else "skipped",
            params={
                "mode": blur_fill.mode,
                "skip_reason": blur_fill.skip_reason or "",
            },
            elapsed_ms=_elapsed_ms_since(t0),
        )
    )

    # ---- (4) hook emphasis curve ------------------------------------
    t0 = time.monotonic()
    hook_payload = payload.get("hook_emphasis")
    if isinstance(hook_payload, dict) and cfg.hook_emphasis_enabled:
        try:
            hook_curve = build_hook_curve(
                hook_payload,
                profile=profile.name,  # type: ignore[arg-type]
                clip_duration_s=render_duration_s,
            )
        except Exception as exc:  # noqa: BLE001 — warn-and-fallback per §9.13
            logger.warning("hook_emphasis curve failed: %s", exc)
            warnings.append(f"hook_emphasis_skipped:{type(exc).__name__}")
            hook_curve = build_hook_curve(
                None,
                profile=profile.name,  # type: ignore[arg-type]
                clip_duration_s=render_duration_s,
            )
    else:
        hook_curve = build_hook_curve(
            None,
            profile=profile.name,  # type: ignore[arg-type]
            clip_duration_s=render_duration_s,
        )
    if hook_curve.apply_zoom or hook_curve.apply_brightness:
        stages_applied.append("hook_emphasis")
    diagnostics["hook_emphasis"] = hook_curve.to_manifest_dict()
    stages.append(
        ManifestStage(
            name="hook_emphasis",
            enabled=cfg.hook_emphasis_enabled,
            status="applied"
            if (hook_curve.apply_zoom or hook_curve.apply_brightness)
            else "skipped",
            params={
                "primary": hook_curve.primary,
                "support": hook_curve.support,
                "disabled_reason": hook_curve.disabled_reason or "",
            },
            elapsed_ms=_elapsed_ms_since(t0),
        )
    )

    # ---- (5) subtitles ---------------------------------------------
    subtitle_words_raw = payload.get("subtitle_words") or []
    subtitle_cues: tuple[SubtitleCue, ...] = ()
    placement_decisions: tuple[dict[str, Any], ...] = ()
    ass_path: Path | None = None

    if cfg.subtitle_enabled and isinstance(subtitle_words_raw, list) and subtitle_words_raw:
        t0 = time.monotonic()
        timed_words = _parse_subtitle_words(subtitle_words_raw)
        render_words = remap_words(timed_words, timeline_map)
        if render_words:
            subtitle_words = tuple(
                SubtitleWord(
                    text=str(rw.payload.get("text", "")),
                    render_start_s=rw.render_start_s,
                    render_end_s=rw.render_end_s,
                    emphasized=bool(rw.payload.get("emphasized", False)),
                )
                for rw in render_words
            )
            preferred_y_table: dict[str, float] = {
                str(k): float(v)
                for k, v in safe_area_mod.PREFERRED_Y_BY_PROFILE.items()
            }
            preferred_y = preferred_y_table.get(profile.name, 0.80)
            base_cues = chunk_words_into_cues(
                subtitle_words,
                style="karaoke_emphasis"
                if cfg.subtitle_karaoke_enabled
                else "plain_caption",
                y_position=preferred_y,
                max_words_per_cue=max(1, cfg.subtitle_max_words_per_cue),
            )
            face_bboxes = _parse_face_bboxes(payload.get("face_bboxes"))
            logo_bbox = None
            decisions: list[dict[str, Any]] = []
            adjusted_cues: list[SubtitleCue] = []
            for cue in base_cues:
                decision = safe_area_mod.validate_subtitle_position(
                    face_bboxes=face_bboxes,
                    logo_bbox=logo_bbox,
                    profile=profile.name,  # type: ignore[arg-type]
                    candidate_y=cue.y_position,
                )
                decisions.append(
                    {
                        "render_start_s": cue.render_start_s,
                        "render_end_s": cue.render_end_s,
                        "chosen_y": decision.chosen_y,
                        "opacity": decision.opacity,
                        "drop": decision.drop,
                        "reason": decision.reason,
                    }
                )
                adjusted_cues.append(
                    SubtitleCue(
                        words=cue.words,
                        style=cue.style,
                        render_start_s=cue.render_start_s,
                        render_end_s=cue.render_end_s,
                        y_position=decision.chosen_y,
                        opacity=decision.opacity,
                        drop=decision.drop,
                    )
                )
            placement_decisions = tuple(decisions)

            non_dropped = [c for c in adjusted_cues if not c.drop]
            if non_dropped:
                ass_path = uniq_dir / "subtitles.ass"
                try:
                    render_ass(non_dropped, ass_path)
                    subtitle_cues = tuple(adjusted_cues)
                    stages_applied.append("subtitles")
                    stages.append(
                        ManifestStage(
                            name="subtitles",
                            enabled=True,
                            status="applied",
                            params={
                                "cue_count": len(non_dropped),
                                "dropped": len(adjusted_cues) - len(non_dropped),
                                "ass_path": str(ass_path),
                            },
                            elapsed_ms=_elapsed_ms_since(t0),
                        )
                    )
                except Exception as exc:  # noqa: BLE001 — warn-only per spec
                    logger.warning("subtitle render failed: %s", exc)
                    warnings.append(f"subtitles_skipped:{type(exc).__name__}")
                    ass_path = None
                    subtitle_cues = ()
            else:
                warnings.append("subtitles_all_dropped_safe_area")
                stages.append(
                    ManifestStage(
                        name="subtitles",
                        enabled=True,
                        status="skipped",
                        params={"reason": "all_dropped_safe_area"},
                        elapsed_ms=_elapsed_ms_since(t0),
                    )
                )
        else:
            stages.append(
                ManifestStage(
                    name="subtitles",
                    enabled=True,
                    status="skipped",
                    params={"reason": "no_words_after_timeline_remap"},
                    elapsed_ms=_elapsed_ms_since(t0),
                )
            )

    diagnostics["subtitles"] = {
        "cue_count": len([c for c in subtitle_cues if not c.drop]),
        "dropped_count": len([c for c in subtitle_cues if c.drop]),
        "ass_path": str(ass_path) if ass_path else "",
        "placement_decisions": list(placement_decisions),
    }

    return _V21State(
        analyzer_version=analyzer_version,
        cuts_applied=cuts_applied,
        timeline_map=timeline_map,
        mirror_plan=mirror_plan,
        hook_curve=hook_curve,
        blur_fill=blur_fill,
        subtitle_cues=subtitle_cues,
        ass_path=ass_path,
        placement_decisions=placement_decisions,
        stages_applied=tuple(stages_applied),
        diagnostics=diagnostics,
    )


def _apply_v21_overrides(
    *,
    cfg: UniqConfig,
    options: FilterGraphOptions,
    state: _V21State,
) -> tuple[UniqConfig, FilterGraphOptions]:
    """Tweak the v2.0 inputs so the splice doesn't duplicate work.

    The v2.0 builder still draws the visual chain backbone; we disable
    the parts the v2.1 splice replaces (the 1.5s mirror window) and
    bump the zoom delta when the hook curve requests it. Blur-fill
    replaces the vertical transform downstream, but we leave the v2.0
    chain in place — the splice swaps the exact filter expression
    rather than removing it.
    """
    new_cfg = cfg
    if state.mirror_plan.enable_expression is not None:
        # v2.1 cluster mirror replaces the v2.0 1.5s middle window
        # entirely (§7.6.5). Disable the v2.0 hflip so the splice can
        # own the mirror.
        new_cfg = replace(new_cfg, mirror_enabled=False)
    elif state.mirror_plan.mode == "none":
        # mirror_clusters == [] (analyzer explicitly chose none).
        new_cfg = replace(new_cfg, mirror_enabled=False)

    if state.hook_curve.apply_zoom and state.hook_curve.zoom_delta > 0.0:
        new_cfg = replace(
            new_cfg,
            zoom=max(cfg.zoom, cfg.zoom + state.hook_curve.zoom_delta),
        )

    new_options = FilterGraphOptions(
        effects_inputs=options.effects_inputs,
        logo_input=options.logo_input,
        music_intervals=options.music_intervals,
        redraw_enabled_override=options.redraw_enabled_override,
        mirror_window=None,
    )
    return new_cfg, new_options


# ---- filter_complex splicing ----


def _split_chains(filter_complex_str: str) -> tuple[list[str], list[str]]:
    """Split the v2.0 filter_complex into video / audio chain lists.

    Filter graphs use ``;`` to separate chains and ``,`` between
    filters inside a chain. Anything that mentions an audio label
    (``[<n>:a]`` or ``[a_*]`` / ``[aout]``) is binned as audio so the
    splice helpers don't accidentally rewrite audio chains.
    """
    chains = [c for c in filter_complex_str.split(";") if c.strip()]
    video: list[str] = []
    audio: list[str] = []
    for chain in chains:
        if _is_audio_chain(chain):
            audio.append(chain)
        else:
            video.append(chain)
    return video, audio


def _is_audio_chain(chain: str) -> bool:
    """Heuristic: chain references an audio label."""
    audio_labels = ("[0:a]", "[a_mute]", "[a_noise]", "[a_fx]", "[a_norm]", "[aout]")
    return any(lbl in chain for lbl in audio_labels) or "anull" in chain


def _first_input_label(chain: str) -> str:
    """Return the first ``[…]`` label in the chain (input)."""
    start = chain.find("[")
    if start == -1:
        return ""
    end = chain.find("]", start)
    if end == -1:
        return ""
    return chain[start : end + 1]


def _replace_first_label(chain: str, old_label: str, new_label: str) -> str:
    """Replace the first occurrence of ``old_label`` in ``chain``."""
    idx = chain.find(old_label)
    if idx == -1:
        return chain
    return chain[:idx] + new_label + chain[idx + len(old_label) :]


def _find_pre_logo_index(video_chains: list[str]) -> int:
    """Return the chain index where v2.1 visual splices should land.

    The v2.0 builder always emits either a ``[v_logo]``-producing chain
    (logo overlay) or a ``…null[vout]`` terminator as the last video
    chain. We insert before whichever ends the visual sequence so
    subtitles + mirror land *after* every other transform but *before*
    the logo, per §10.5 / §9.9.
    """
    # Logo chain wins when present.
    for i, c in enumerate(video_chains):
        if c.endswith("[v_logo]"):
            return i
    # Terminator chain otherwise.
    for i, c in enumerate(video_chains):
        if c.endswith("[vout]"):
            return i
    return len(video_chains)


def _splice_cuts(
    *,
    video_chains: list[str],
    audio_chains: list[str],
    cuts_applied: AppliedCuts,
    has_audio: bool,
) -> tuple[list[str], list[str], list[str]]:
    """Prepend trim/concat chains and rewire ``[0:v]`` / ``[0:a]``.

    Cut times are clip-local (relative to the editor's seek window),
    so we feed them straight to ``trim``/``atrim`` without further
    offsets — the worker has already applied ``-ss/-to`` for the
    surrounding window.
    """
    warnings: list[str] = []
    kept = cuts_applied.kept_segments
    if not kept:
        return video_chains, audio_chains, warnings
    if len(kept) == 1:
        seg = kept[0]
        # A single kept segment covering the full window is a no-op.
        if seg.source_start_s <= 1e-3 and seg.duration_s > 0:
            return video_chains, audio_chains, warnings

    # Build trim chains.
    v_segments: list[str] = []
    a_segments: list[str] = []
    concat_v_inputs: list[str] = []
    concat_a_inputs: list[str] = []
    for i, seg in enumerate(kept):
        v_label = f"[v_c{i}]"
        v_segments.append(
            f"[0:v]trim=start={seg.source_start_s:.3f}:"
            f"end={seg.source_end_s:.3f},setpts=PTS-STARTPTS{v_label}"
        )
        concat_v_inputs.append(v_label)
        if has_audio:
            a_label = f"[a_c{i}]"
            a_segments.append(
                f"[0:a]atrim=start={seg.source_start_s:.3f}:"
                f"end={seg.source_end_s:.3f},asetpts=PTS-STARTPTS{a_label}"
            )
            concat_a_inputs.append(a_label)
    n = len(kept)
    if has_audio:
        v_segments.append(
            f"{''.join(concat_v_inputs)}concat=n={n}:v=1:a=0[v_cut]"
        )
        a_segments.append(
            f"{''.join(concat_a_inputs)}concat=n={n}:v=0:a=1[a_cut]"
        )
    else:
        v_segments.append(
            f"{''.join(concat_v_inputs)}concat=n={n}:v=1:a=0[v_cut]"
        )

    # Rewrite the first downstream reference to [0:v] → [v_cut].
    new_video = list(v_segments)
    rewired_v = False
    for chain in video_chains:
        if not rewired_v and "[0:v]" in chain:
            new_video.append(_replace_first_label(chain, "[0:v]", "[v_cut]"))
            rewired_v = True
        else:
            new_video.append(chain)

    new_audio = list(a_segments) if has_audio else []
    rewired_a = False
    for chain in audio_chains:
        if has_audio and not rewired_a and "[0:a]" in chain:
            new_audio.append(_replace_first_label(chain, "[0:a]", "[a_cut]"))
            rewired_a = True
        else:
            new_audio.append(chain)
    if has_audio and not audio_chains:
        # No v2.0 audio chain emitted (silent source). Synthesize
        # passthrough so the concat output reaches [aout].
        new_audio.append("[a_cut]anull[aout]")

    return new_video, new_audio, warnings


def _splice_blur_fill(
    video_chains: list[str],
    *,
    blur_fill: BlurFillFilter,
) -> tuple[list[str], list[str]]:
    """Replace the v2.0 vertical transform with the blur composite.

    The v2.0 builder always emits ``[<src>]scale=…,crop=…,setsar=1
    [v_vert]`` as the first video chain. We rebuild the blur filter
    with the v2.0 input label (``[0:v]`` or ``[v_cut]`` after the cuts
    splice) and ``[v_vert]`` as the output, so downstream chains
    continue to consume ``[v_vert]`` unchanged.
    """
    warnings: list[str] = []
    if not video_chains:
        return video_chains, warnings

    # Find the first chain that produces [v_vert] (the vertical
    # transform chain). The cuts splice prepends new chains in front;
    # we still want to splice the *original* v2.0 vertical transform.
    vert_idx = -1
    for i, chain in enumerate(video_chains):
        if chain.endswith("[v_vert]"):
            vert_idx = i
            break
    if vert_idx == -1:
        warnings.append("blur_fill_skipped_no_v_vert")
        return video_chains, warnings

    original = video_chains[vert_idx]
    in_label = _first_input_label(original)  # e.g. [0:v] or [v_cut]
    if not in_label:
        warnings.append("blur_fill_skipped_no_input_label")
        return video_chains, warnings

    # Rebuild the blur filter with the actual upstream label and the
    # v2.0 output label so the rest of the chain links unchanged.
    rebuilt = build_blur_fill_filter(
        source_resolution=(int(blur_fill.source_aspect * 1000), 1000)
        if blur_fill.source_aspect
        else (1920, 1080),
        target_resolution=(1080, 1920),
        mode=blur_fill.mode,
        input_label=in_label.strip("[]"),
        output_label="v_vert",
    )
    if rebuilt.skip_reason is not None or not rebuilt.filter_complex:
        warnings.append(f"blur_fill_skipped:{rebuilt.skip_reason or 'empty'}")
        return video_chains, warnings

    new_chains = list(video_chains)
    # blur_fill emits multiple ; -separated chains in a single string.
    blur_chains = [c for c in rebuilt.filter_complex.split(";") if c.strip()]
    new_chains[vert_idx : vert_idx + 1] = blur_chains
    return new_chains, warnings


def _splice_mirror(
    video_chains: list[str], *, mirror_plan: MirrorPlan
) -> list[str]:
    """Insert ``hflip=enable='<expr>'`` before the logo / terminator."""
    if mirror_plan.enable_expression is None:
        return video_chains
    idx = _find_pre_logo_index(video_chains)
    if idx == len(video_chains):
        # No anchor — append as a self-contained terminator.
        return video_chains
    target = video_chains[idx]
    in_label = _first_input_label(target)
    if not in_label:
        return video_chains
    new_label = "[v_mirror]"
    new_chain = (
        f"{in_label}hflip=enable='{mirror_plan.enable_expression}'{new_label}"
    )
    updated_target = _replace_first_label(target, in_label, new_label)
    return [*video_chains[:idx], new_chain, updated_target, *video_chains[idx + 1 :]]


def _splice_brightness(
    video_chains: list[str], *, hook_curve: HookCurve
) -> list[str]:
    """Insert ``eq=brightness=<expr>`` before subtitle burn-in / logo.

    The brightness pulse is encoded as a piecewise-linear curve over
    render time; ffmpeg's ``eq`` filter accepts an expression in
    ``t``. We synthesize a linear-decay between the two pulse points
    (``BrightnessPulsePoint(t_s, delta)``) and clamp to 0 outside the
    pulse window so the rest of the clip plays unchanged.
    """
    if not hook_curve.apply_brightness or not hook_curve.brightness_pulse:
        return video_chains
    points = hook_curve.brightness_pulse
    if len(points) < 2:
        return video_chains
    p0, p1 = points[0], points[-1]
    if p1.t_s <= p0.t_s:
        return video_chains
    # eq=brightness expression: peak at t_s=p0.t_s; linear decay to 0
    # at t_s=p1.t_s; zero outside [p0.t_s, p1.t_s].
    expr = (
        f"if(between(t,{p0.t_s:.3f},{p1.t_s:.3f}),"
        f"{p0.delta:.4f}*(1-(t-{p0.t_s:.3f})/({p1.t_s:.3f}-{p0.t_s:.3f})),"
        f"0)"
    )
    idx = _find_pre_logo_index(video_chains)
    if idx == len(video_chains):
        return video_chains
    target = video_chains[idx]
    in_label = _first_input_label(target)
    if not in_label:
        return video_chains
    new_label = "[v_bright]"
    new_chain = f"{in_label}eq=brightness='{expr}'{new_label}"
    updated_target = _replace_first_label(target, in_label, new_label)
    return [*video_chains[:idx], new_chain, updated_target, *video_chains[idx + 1 :]]


def _splice_subtitles(
    video_chains: list[str], *, ass_path: Path
) -> list[str]:
    """Insert ``subtitles=filename=<ass>`` before the logo overlay."""
    idx = _find_pre_logo_index(video_chains)
    if idx == len(video_chains):
        return video_chains
    target = video_chains[idx]
    in_label = _first_input_label(target)
    if not in_label:
        return video_chains
    # ffmpeg subtitles= filter accepts a POSIX-style path; we escape
    # any ``:`` or ``'`` so a path with a drive letter or quote can't
    # break the filter-spec parser.
    safe_path = str(ass_path).replace("\\", "/").replace(":", "\\:").replace(
        "'", "\\'"
    )
    new_label = "[v_subs]"
    new_chain = f"{in_label}subtitles=filename='{safe_path}'{new_label}"
    updated_target = _replace_first_label(target, in_label, new_label)
    return [*video_chains[:idx], new_chain, updated_target, *video_chains[idx + 1 :]]


def _splice_v21_filters(
    *,
    graph: FilterGraph,
    cfg: UniqConfig,
    state: _V21State,
    probe: ProbeResult,
    start_s: float,
    end_s: float,
    stages: list[ManifestStage],
    warnings: list[str],
) -> FilterGraph:
    """Apply the §10.5 strict-order v2.1 augmentations to a v2.0 graph.

    Returns a new :class:`FilterGraph` with the spliced
    ``filter_complex_str`` and an extended ``stages_applied`` tuple
    that prefixes the v2.1 stage names (so the manifest renders them
    in the order they ran).
    """
    video_chains, audio_chains = _split_chains(graph.filter_complex_str)

    if state.cuts_applied is not None and state.cuts_applied.kept_segments:
        # Only splice cuts when there's something to remove (more
        # than one kept segment, or the only segment doesn't cover
        # the full clip).
        kept = state.cuts_applied.kept_segments
        full_clip = (
            len(kept) == 1
            and kept[0].source_start_s <= 1e-3
            and abs(kept[0].duration_s - (end_s - start_s)) <= 1e-3
        )
        if not full_clip:
            video_chains, audio_chains, cut_warnings = _splice_cuts(
                video_chains=video_chains,
                audio_chains=audio_chains,
                cuts_applied=state.cuts_applied,
                has_audio=probe.has_audio,
            )
            warnings.extend(cut_warnings)

    if state.blur_fill.skip_reason is None and state.blur_fill.filter_complex:
        video_chains, blur_warnings = _splice_blur_fill(
            video_chains, blur_fill=state.blur_fill
        )
        warnings.extend(blur_warnings)

    if state.mirror_plan.enable_expression is not None:
        video_chains = _splice_mirror(video_chains, mirror_plan=state.mirror_plan)

    if state.hook_curve.apply_brightness:
        video_chains = _splice_brightness(video_chains, hook_curve=state.hook_curve)

    if state.ass_path is not None:
        video_chains = _splice_subtitles(video_chains, ass_path=state.ass_path)

    new_filter_complex = ";".join(video_chains + audio_chains)

    return FilterGraph(
        filter_complex_str=new_filter_complex,
        input_args=graph.input_args,
        output_args=graph.output_args,
        mapped_streams=dict(graph.mapped_streams),
        warnings=graph.warnings,
        stages_applied=state.stages_applied + graph.stages_applied,
    )


# ---- unique-distance + manifest extras ----


def _safe_unique_distance(
    source: Path,
    clip_path: Path,
    cfg: UniqConfig,
    warnings: list[str],
    stages: list[ManifestStage],
) -> UniqueDistanceComponents | None:
    """Compute the §10.7 unique-distance vector; warn-only on failure."""
    if not cfg.unique_distance_enabled:
        stages.append(
            ManifestStage(
                name="unique_distance",
                enabled=False,
                status="skipped",
                params={"reason": "unique_distance_disabled"},
                elapsed_ms=0,
            )
        )
        return None
    started = time.monotonic()
    try:
        components = compute_components(source, clip_path)
    except UniqueDistanceError as exc:
        logger.warning("unique_distance compute failed: %s", exc)
        warnings.append(f"unique_distance_skipped:{type(exc).__name__}")
        stages.append(
            ManifestStage(
                name="unique_distance",
                enabled=True,
                status="failed",
                params={"reason": str(exc)[:200]},
                elapsed_ms=_elapsed_ms_since(started),
            )
        )
        return None
    except Exception as exc:  # noqa: BLE001 — warn-only per §9.13
        logger.warning("unique_distance unexpected error: %s", exc)
        warnings.append(f"unique_distance_error:{type(exc).__name__}")
        stages.append(
            ManifestStage(
                name="unique_distance",
                enabled=True,
                status="failed",
                params={"reason": str(exc)[:200]},
                elapsed_ms=_elapsed_ms_since(started),
            )
        )
        return None

    stages.append(
        ManifestStage(
            name="unique_distance",
            enabled=True,
            status="applied",
            params={
                "components_version": components.components_version,
                "phash_sample_count": len(components.phash_distances),
            },
            elapsed_ms=_elapsed_ms_since(started),
        )
    )
    return components


def _build_v21_extras(
    *,
    state: _V21State,
    unique_components: UniqueDistanceComponents | None,
) -> dict[str, Any]:
    """Assemble the manifest ``extra`` payload added by Part 7."""
    extras: dict[str, Any] = {
        "analyzer_v2_payload_version": state.analyzer_version,
        "v21_stages_applied": list(state.stages_applied),
        "v21_stages_diagnostics": state.diagnostics,
        "mirror": state.diagnostics.get("mirror", {}),
        "blur_fill": state.diagnostics.get("blur_fill", {}),
        "hook_emphasis": state.diagnostics.get("hook_emphasis", {}),
        "subtitles": state.diagnostics.get("subtitles", {}),
        "timeline": {
            "render_duration_s": state.timeline_map.render_duration_s,
            "segments": [
                {
                    "source_start_s": seg.source_start_s,
                    "source_end_s": seg.source_end_s,
                    "render_start_s": seg.render_start_s,
                    "render_end_s": seg.render_end_s,
                }
                for seg in state.timeline_map.segments
            ],
        },
    }
    if state.cuts_applied is not None:
        extras["cuts"] = {
            "honored": [
                {
                    "source_start_s": ks.source_start_s,
                    "source_end_s": ks.source_end_s,
                }
                for ks in state.cuts_applied.kept_segments
            ],
            "skipped": [
                {
                    "start_s": sc.cut.start_s,
                    "end_s": sc.cut.end_s,
                    "kind": sc.cut.kind,
                    "reason": sc.reason,
                }
                for sc in state.cuts_applied.skipped
            ],
        }
    if unique_components is not None:
        extras["unique_distance"] = unique_components.to_manifest_dict()
    return extras
