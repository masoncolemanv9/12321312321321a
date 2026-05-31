"""Editor Agent v2 — uniqueization package.

Foundation layer (Part 1 of 10) for the Editor Agent rewrite. See
``final_spec_FULL.md`` Part 9 (v2.0 baseline), Part 10 (v2.1 extension),
and Parts 1–8 (v6 creative-planner) for the larger picture.

This package is **opt-in** via the ``EDITOR_VERSION`` env var:

* ``EDITOR_VERSION=v1`` (default): legacy ``bot/workers/editor.py`` runs.
  Nothing in this package executes. The Lilush pipeline behaves exactly
  as before.
* ``EDITOR_VERSION=v2``: ``EditorV2Worker`` (added by Part 3) runs the
  v2.0 fused-ffmpeg pipeline using the helpers in this package.
* ``EDITOR_VERSION=v2.1``: v2.1 extensions layer on top (Parts 4–7).
* ``EDITOR_VERSION=v6`` + ``EDITOR_V6_ENABLED=true``: full v6
  creative-planner path (Parts 8–10).

The public surface re-exported from this package is intentionally small
and stable so the rest of the codebase (and other Devin sessions
landing later parts) can rely on a fixed import path.
"""

from __future__ import annotations

from .assets import (
    AssetError,
    ensure_yunet_model,
    schemas_dir,
    yunet_model_path,
)
from .blur_fill import (
    BlurFillFilter,
    BlurFillMode,
    BlurFillSkipReason,
    build_blur_fill_filter,
    is_landscape_or_square,
)
from .cluster_mirror import (
    MirrorMode,
    MirrorPlan,
    Scene,
    SceneCluster,
    VetoReason,
    build_cluster_mirror_filter,
)
from .config import (
    UniqConfig,
    UniqProfile,
    config_origin,
    load_config,
)
from .cuts import (
    AppliedCuts,
    Cut,
    CutKind,
    CutsError,
    KeptSegment,
    SegmentLabels,
    ShippedKeptSegment,
    SkippedCut,
    SkipReason,
    apply_cuts,
    kept_segments_match,
)
from .face_yunet import (
    FaceSample,
    FaceSampling,
    sample_faces,
)
from .filtergraph import (
    FilterGraph,
    FilterGraphOptions,
    build_v20_graph,
    serialize_filter_complex,
)
from .frame_exports import (
    DEFAULT_FRAMES_MAX,
    DEFAULT_FRAMES_MIN,
    FrameExportCandidate,
    build_frame_export_argv,
    select_frame_export_candidates,
    write_frame_exports_metadata,
)
from .hook_emphasis import (
    BrightnessPulsePoint,
    HookCurve,
    HookDisabledReason,
    HookEmphasisError,
    HookPrimary,
    HookSupport,
    build_hook_curve,
)
from .manifest import (
    ManifestError,
    ManifestStage,
    UniqManifest,
    build_creative_planner_section,
    write_manifest_atomic,
)
from .phash import (
    PhashError,
    extract_phash_samples,
    hamming_distance,
)
from .probe import (
    ProbeError,
    ProbeResult,
    probe_source,
)
from .profiles import (
    PROFILE_DEFAULTS,
    resolve_profile,
)
from .runner import (
    FfmpegError,
    FfmpegResult,
    FfmpegTimeoutError,
    run_ffmpeg_atomic,
)
from .safe_area import (
    ALTERNATIVE_Y_POSITIONS,
    OPACITY_FLOOR,
    OVERLAP_CLEAR_THRESHOLD,
    OVERLAP_DROP_THRESHOLD,
    PREFERRED_Y_BY_PROFILE,
    Bbox,
    PlacementFallbackReason,
    ProfileName,
    SafeAreaDecision,
    validate_subtitle_position,
)
from .subtitle_renderer import (
    STYLE_PROFILES,
    AssOptions,
    StyleProfile,
    SubtitleCue,
    SubtitleStyle,
    SubtitleWord,
    build_ass,
    chunk_words_into_cues,
    render_ass,
)
from .thumbnail import (
    ThumbnailError,
    extract_thumbnail,
)
from .timeline import (
    RenderWord,
    TimedWord,
    TimelineError,
    TimelineMap,
    TimelineSegment,
    derive_timeline_map,
    identity_map,
    remap_intervals,
    remap_words,
    timeline_map_match,
)
from .unique_distance import (
    ComponentsVersion,
    UniqueDistanceComponents,
    UniqueDistanceError,
    compute_components,
)

__all__ = [
    "ALTERNATIVE_Y_POSITIONS",
    "OPACITY_FLOOR",
    "OVERLAP_CLEAR_THRESHOLD",
    "OVERLAP_DROP_THRESHOLD",
    "PREFERRED_Y_BY_PROFILE",
    "PROFILE_DEFAULTS",
    "STYLE_PROFILES",
    "AppliedCuts",
    "AssOptions",
    "AssetError",
    "Bbox",
    "BlurFillFilter",
    "BlurFillMode",
    "BlurFillSkipReason",
    "BrightnessPulsePoint",
    "ComponentsVersion",
    "Cut",
    "CutKind",
    "CutsError",
    "DEFAULT_FRAMES_MAX",
    "DEFAULT_FRAMES_MIN",
    "FaceSample",
    "FaceSampling",
    "FfmpegError",
    "FfmpegResult",
    "FfmpegTimeoutError",
    "FilterGraph",
    "FilterGraphOptions",
    "FrameExportCandidate",
    "HookCurve",
    "HookDisabledReason",
    "HookEmphasisError",
    "HookPrimary",
    "HookSupport",
    "KeptSegment",
    "ManifestError",
    "ManifestStage",
    "MirrorMode",
    "MirrorPlan",
    "PhashError",
    "PlacementFallbackReason",
    "ProbeError",
    "ProbeResult",
    "ProfileName",
    "RenderWord",
    "SafeAreaDecision",
    "Scene",
    "SceneCluster",
    "SegmentLabels",
    "ShippedKeptSegment",
    "SkipReason",
    "SkippedCut",
    "StyleProfile",
    "SubtitleCue",
    "SubtitleStyle",
    "SubtitleWord",
    "ThumbnailError",
    "TimedWord",
    "TimelineError",
    "TimelineMap",
    "TimelineSegment",
    "UniqConfig",
    "UniqManifest",
    "build_creative_planner_section",
    "UniqProfile",
    "UniqueDistanceComponents",
    "UniqueDistanceError",
    "VetoReason",
    "apply_cuts",
    "build_ass",
    "build_blur_fill_filter",
    "build_cluster_mirror_filter",
    "build_frame_export_argv",
    "build_hook_curve",
    "build_v20_graph",
    "chunk_words_into_cues",
    "compute_components",
    "config_origin",
    "derive_timeline_map",
    "ensure_yunet_model",
    "extract_phash_samples",
    "extract_thumbnail",
    "hamming_distance",
    "identity_map",
    "is_landscape_or_square",
    "kept_segments_match",
    "load_config",
    "probe_source",
    "remap_intervals",
    "remap_words",
    "render_ass",
    "resolve_profile",
    "run_ffmpeg_atomic",
    "sample_faces",
    "schemas_dir",
    "select_frame_export_candidates",
    "serialize_filter_complex",
    "timeline_map_match",
    "validate_subtitle_position",
    "write_frame_exports_metadata",
    "write_manifest_atomic",
    "yunet_model_path",
]

__version__ = "0.1.0-part6-blur-hook-uniq"
