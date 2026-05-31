"""v6 creative planner — Parts 8 + 9 + 10.

This subpackage holds the *decision-making* halves of the v6 creative
planner per ``final_spec_FULL`` Parts 5–8 + §12.4 + §12.6:

**Part 8 (modules A — beat / zoom / cut / mirror / validation / locks):**

* :mod:`.beat_sheet` — §5 beat boundary derivation + purpose mapping.
* :mod:`.zoom_planner` — §6 piecewise zoom curve under profile +
  motion-sickness HARD caps.
* :mod:`.cut_planner` — §7.1–§7.5 + §7.10 two-stage cut decisions
  with HARD preservation rules.
* :mod:`.mirror_planner` — §7.6 cluster-mirror veto + all-or-none.
* :mod:`.validation` — §12.4 typed-policy validation of
  ``director_brief.json``.
* :mod:`.locks` — §12.6 built-in style locks
  (``cinematic_drama`` / ``fast_dialogue`` / ``action_sports``).

**Part 9 (modules B — subtitle / audio / color / blur-fill / hook):**

* :mod:`.subtitle_planner` — §8.1 + §8.4 per-beat subtitle style
  + emphasis-word selection.
* :mod:`.audio_planner` — §8.8 per-beat audio policy + emphasis
  volume-bump events under profile caps.
* :mod:`.color_planner` — §8.10 color-intent grammar + adjacent
  cross-fade durations.
* :mod:`.blur_fill_planner` — §8.11 blur-fill mode selection.
* :mod:`.hook_planner` — §8.5 + §8.7 hook overlay + treatment
  budget (raises :class:`BudgetViolation` on stack).

All public names from the modules above are re-exported here so
consumers can `from bot.uniqueization.planner import …` without
chasing module paths.

**Part 10 (orchestration + renderer + publisher handoff):**

* :mod:`.creative_planner` — top-level §1.2 forward pipeline,
  emits the §4 ``edit_plan.json`` dict and the §1.3 status taxonomy.
* :mod:`.renderer_adapter` — bridges an ``edit_plan`` dict into a
  v2.1 worker payload (no new ffmpeg stages — re-uses the existing
  v2.1 filter graph builders from Parts 2-6).
* :mod:`.cover_render` — pure-function helper that materialises the
  publisher's §11.2 ``render_jobs[]`` into ffmpeg argv lists.
"""

from __future__ import annotations

from .audio_planner import (
    AUDIO_POLICY_ENUM,
    DEFAULT_LOUDNESS_TARGET_LUFS,
    DEFAULT_TRUE_PEAK_MAX_DBTP,
    PROFILE_AUDIO_CAPS,
    AudioBeatPolicy,
    AudioBumpEvent,
    AudioPlan,
    plan_audio,
)
from .beat_sheet import (
    PRIMARY_MOVE_ENUM,
    PURPOSE_ENUM,
    TRANSCRIPT_DENSITY_THRESHOLD,
    Beat,
    BeatPurpose,
    PrimaryMove,
    build_beat_sheet,
    default_primary_move_for_purpose,
)
from .blur_fill_planner import (
    BLUR_FILL_ENUM,
    BLUR_RATIO_PANORAMIC_MIN,
    BLUR_RATIO_PORTRAIT_MAX,
    BlurFillPlan,
    BlurFillSegment,
    plan_blur_fill,
)
from .color_planner import (
    BRIGHTNESS_DELTA_CAP,
    COLOR_INTENT_ENUM,
    CONTRAST_DELTA_CAP,
    CROSS_FADE_DEFAULT_MS,
    CROSS_FADE_MAX_MS,
    CROSS_FADE_MIN_MS,
    INTENT_DELTAS,
    PROFILE_BASE_COLORGRADE,
    SATURATION_DELTA_CAP,
    ColorIntent,
    ColorPlan,
    ColorSegment,
    ColorTransition,
    plan_color,
)
from .cover_render import (
    CROP_STRATEGIES,
    PLATFORMS,
    CoverRenderJob,
    CoverRenderJobError,
    CoverRenderPlan,
    TextOverlay,
    build_cover_render_argv,
    build_cover_render_plan,
    parse_render_jobs,
)
from .creative_planner import (
    CREATIVE_INTENSITY_CEILINGS,
    CREATIVE_PLAN_SCHEMA_VERSION,
    CreativePlanError,
    CreativePlanResult,
    build_manifest_reasoning,
    compute_creative_intensity_score,
    planner_cut_kind_to_worker,
    run_creative_planner,
)
from .cut_planner import (
    CUT_KIND_ENUM,
    PROFILE_CUT_THRESHOLDS,
    REACTION_WINDOW_BY_PROFILE,
    CutDecision,
    CutPlan,
    plan_cuts_final,
)
from .hook_planner import (
    DEFAULT_HOOK_DIALOGUE_DENSITY_THRESHOLD,
    DEFAULT_HOOK_OVERLAY_Y,
    HOOK_FORBIDDEN_DEFAULTS,
    HOOK_PRIMARY_ENUM,
    HOOK_SUPPORT_ENUM,
    BudgetViolation,
    HookOverlay,
    HookPlan,
    augment_hook_with_extra_support,
    plan_hook,
)
from .locks import (
    BUILT_IN_LOCK_IDS,
    STYLE_LOCKS,
    StyleLock,
    get_lock,
)
from .mirror_planner import (
    VETO_REASONS,
    MirrorDecision,
    MirrorPlan,
    plan_mirror_final,
)
from .renderer_adapter import (
    PLANNER_TO_WORKER_CUT_KIND,
    RendererAdapterResult,
    apply_edit_plan_to_payload,
    translate_planner_cut_kind,
)
from .subtitle_planner import (
    SEMANTIC_ROLE_ENUM,
    SUBTITLE_STYLE_ENUM,
    Y_FALLBACK_POSITIONS,
    Y_POSITION_BY_PROFILE,
    EmphasisWord,
    SubtitleEvent,
    SubtitlePlan,
    plan_subtitle,
)
from .validation import (
    OPTIONAL_CAPABILITIES,
    REQUIRED_FLOOR_FIELDS,
    REQUIRED_SCENE_FIELDS,
    REQUIRED_TRANSCRIPT_SEGMENT_FIELDS,
    REQUIRED_WINDOW_FIELDS,
    RULES,
    Diagnostic,
    Severity,
    ValidationResult,
    supported_schema_major,
    validate_director_brief,
)
from .zoom_planner import (
    ACTION_LEVEL_ENTER,
    ACTION_LEVEL_EXIT,
    DEFAULT_TRANSITION_MS,
    MAX_TRANSITION_MS,
    MOTION_ACCELERATION_CAP,
    MOTION_VELOCITY_CAP,
    PAN_VELOCITY_CAP,
    PROFILE_CAPS,
    ZOOM_MODE_ENUM,
    ZOOM_TARGET_ENUM,
    ProfileZoomCaps,
    ZoomCurve,
    ZoomMode,
    ZoomSegment,
    ZoomTarget,
    plan_zoom_curve,
)

__all__ = (
    # beat_sheet
    "Beat",
    "BeatPurpose",
    "PRIMARY_MOVE_ENUM",
    "PURPOSE_ENUM",
    "PrimaryMove",
    "TRANSCRIPT_DENSITY_THRESHOLD",
    "build_beat_sheet",
    "default_primary_move_for_purpose",
    # zoom_planner
    "ACTION_LEVEL_ENTER",
    "ACTION_LEVEL_EXIT",
    "DEFAULT_TRANSITION_MS",
    "MAX_TRANSITION_MS",
    "MOTION_ACCELERATION_CAP",
    "MOTION_VELOCITY_CAP",
    "PAN_VELOCITY_CAP",
    "PROFILE_CAPS",
    "ProfileZoomCaps",
    "ZOOM_MODE_ENUM",
    "ZOOM_TARGET_ENUM",
    "ZoomCurve",
    "ZoomMode",
    "ZoomSegment",
    "ZoomTarget",
    "plan_zoom_curve",
    # cut_planner
    "CUT_KIND_ENUM",
    "CutDecision",
    "CutPlan",
    "PROFILE_CUT_THRESHOLDS",
    "REACTION_WINDOW_BY_PROFILE",
    "plan_cuts_final",
    # mirror_planner
    "MirrorDecision",
    "MirrorPlan",
    "VETO_REASONS",
    "plan_mirror_final",
    # validation
    "Diagnostic",
    "OPTIONAL_CAPABILITIES",
    "REQUIRED_FLOOR_FIELDS",
    "REQUIRED_SCENE_FIELDS",
    "REQUIRED_TRANSCRIPT_SEGMENT_FIELDS",
    "REQUIRED_WINDOW_FIELDS",
    "RULES",
    "Severity",
    "ValidationResult",
    "supported_schema_major",
    "validate_director_brief",
    # locks
    "BUILT_IN_LOCK_IDS",
    "STYLE_LOCKS",
    "StyleLock",
    "get_lock",
    # subtitle_planner (Part 9)
    "EmphasisWord",
    "SEMANTIC_ROLE_ENUM",
    "SUBTITLE_STYLE_ENUM",
    "SubtitleEvent",
    "SubtitlePlan",
    "Y_FALLBACK_POSITIONS",
    "Y_POSITION_BY_PROFILE",
    "plan_subtitle",
    # audio_planner (Part 9)
    "AUDIO_POLICY_ENUM",
    "AudioBeatPolicy",
    "AudioBumpEvent",
    "AudioPlan",
    "DEFAULT_LOUDNESS_TARGET_LUFS",
    "DEFAULT_TRUE_PEAK_MAX_DBTP",
    "PROFILE_AUDIO_CAPS",
    "plan_audio",
    # color_planner (Part 9)
    "BRIGHTNESS_DELTA_CAP",
    "COLOR_INTENT_ENUM",
    "CONTRAST_DELTA_CAP",
    "CROSS_FADE_DEFAULT_MS",
    "CROSS_FADE_MAX_MS",
    "CROSS_FADE_MIN_MS",
    "ColorIntent",
    "ColorPlan",
    "ColorSegment",
    "ColorTransition",
    "INTENT_DELTAS",
    "PROFILE_BASE_COLORGRADE",
    "SATURATION_DELTA_CAP",
    "plan_color",
    # blur_fill_planner (Part 9)
    "BLUR_FILL_ENUM",
    "BLUR_RATIO_PANORAMIC_MIN",
    "BLUR_RATIO_PORTRAIT_MAX",
    "BlurFillPlan",
    "BlurFillSegment",
    "plan_blur_fill",
    # hook_planner (Part 9)
    "BudgetViolation",
    "DEFAULT_HOOK_DIALOGUE_DENSITY_THRESHOLD",
    "DEFAULT_HOOK_OVERLAY_Y",
    "HOOK_FORBIDDEN_DEFAULTS",
    "HOOK_PRIMARY_ENUM",
    "HOOK_SUPPORT_ENUM",
    "HookOverlay",
    "HookPlan",
    "augment_hook_with_extra_support",
    "plan_hook",
    # creative_planner (Part 10)
    "CREATIVE_INTENSITY_CEILINGS",
    "CREATIVE_PLAN_SCHEMA_VERSION",
    "CreativePlanError",
    "CreativePlanResult",
    "build_manifest_reasoning",
    "compute_creative_intensity_score",
    "planner_cut_kind_to_worker",
    "run_creative_planner",
    # renderer_adapter (Part 10)
    "PLANNER_TO_WORKER_CUT_KIND",
    "RendererAdapterResult",
    "apply_edit_plan_to_payload",
    "translate_planner_cut_kind",
    # cover_render (Part 10)
    "CROP_STRATEGIES",
    "CoverRenderJob",
    "CoverRenderJobError",
    "CoverRenderPlan",
    "PLATFORMS",
    "TextOverlay",
    "build_cover_render_argv",
    "build_cover_render_plan",
    "parse_render_jobs",
)
