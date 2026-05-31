"""Frozen-dataclass configs for the Editor Agent v2 / v2.1 / v6 pipeline.

Source of truth for env-var names: ``final_spec_FULL.md`` §13.1. Those
names are also added to ``bot/config.py`` so the rest of the repo can
import them as module-level constants (Lilush convention). This module
re-reads them through ``os.environ`` so tests can monkeypatch the env
without bouncing the ``bot.config`` module.

Resolution precedence (§13.1, §13.2):

    hardcoded defaults  <  env vars  <  profile (light/medium/heavy)
                                          <  per-job payload overrides

Per-job overrides are applied by :func:`load_config` callers (typically
:class:`bot.workers.editor_v2.EditorV2Worker`, landing in Part 3) by
passing ``overrides=`` to :func:`load_config`. Profile selection is
deferred to :mod:`bot.uniqueization.profiles`.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from typing import Any


@dataclass(frozen=True, slots=True)
class UniqConfig:
    """Resolved Editor Agent configuration for a single render job.

    Field names mirror env vars from §13.1 (lowercased, no ``EDITOR_``
    prefix). Every field has a hardcoded default that matches the v1
    baseline so that booting with ``EDITOR_VERSION=v2`` without any
    other env knobs produces a sane medium-profile-ish render.
    """

    # --- v2.0 baseline -----------------------------------------------
    editor_version: str = "v1"
    editor_profile: str = "light"
    ffmpeg_timeout_s: float = 360.0
    ffmpeg_threads: str = ""

    zoom: float = 0.0
    face_reframe_enabled: bool = True
    yunet_model_path: str = ""

    mirror_enabled: bool = True
    mirror_duration_s: float = 1.5

    redraw_enabled: bool = False
    colorgrade_enabled: bool = True
    lut_path: str = ""

    effects_dir: str = ""
    effects_opacity: float = 0.15

    music_remover_enabled: bool = False
    music_remover_mode: str = "mute"

    audio_fx_enabled: bool = True
    audio_fx_wet: float = 0.0
    loudnorm_enabled: bool = True

    phash_enabled: bool = True
    thumbnail_enabled: bool = True
    keep_uniq_artifacts: bool = False

    # --- ffmpeg encoder defaults (carried over from v1) --------------
    video_crf: int = 23
    video_preset: str = "medium"
    audio_bitrate: str = "128k"
    output_width: int = 1080
    output_height: int = 1920

    # --- v2.1 layer --------------------------------------------------
    subtitle_enabled: bool = True
    subtitle_karaoke_enabled: bool = True
    subtitle_max_words_per_cue: int = 3
    blur_fill_enabled: bool = True
    hook_emphasis_enabled: bool = True
    unique_distance_enabled: bool = True

    # --- v6 creative planner -----------------------------------------
    v6_enabled: bool = False
    v6_force_skip: bool = False
    hook_dialogue_density_threshold: float = 12.0
    creative_intensity_ceiling_light: float = 0.4
    creative_intensity_ceiling_medium: float = 0.6
    creative_intensity_ceiling_heavy: float = 0.8

    # --- per-job uniqueness slider (UI "Уникальность") -------------
    # 0–400. 0 = byte-equivalent baseline (Part 1-11 pipeline output).
    # 100 = exactly one debate-vetted profile envelope; 400 = 4× the
    # debate envelope (red zone, hard cap). See
    # ``bot.uniqueization.randomization`` for the parameter list
    # (zoom / mirror_duration_s / effects_opacity / audio_fx_wet) and
    # the per-job ``v = uniform(0.6 × slider, slider)`` formula.
    uniqueness_pct: int = 0

    # --- v1 overlay carry-over (used by both v1 and v2 pipelines) ----
    overlay_logo_path: str = ""
    overlay_margin_px: int = 40

    # --- origin tracking (filled by load_config) ---------------------
    _origin: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class UniqProfile:
    """Profile-level overrides applied between env vars and per-job overrides.

    See ``final_spec_FULL.md`` §9.6 for profile semantics. Three named
    profiles ship: ``light`` (default, v1-compatible), ``medium``
    (recommended for v2.0 opt-in), and ``heavy`` (max-quality, slowest).
    """

    name: str
    zoom: float
    video_crf: int
    mirror_duration_s: float
    effects_opacity: float
    audio_fx_wet: float
    redraw_enabled: bool
    yunet_sample_rate_s: float = 0.5


# ---- helpers --------------------------------------------------------


def _env_bool(name: str, default: bool) -> tuple[bool, bool]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default, False
    return raw.strip().lower() == "true", True


def _env_float(name: str, default: float) -> tuple[float, bool]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default, False
    return float(raw), True


def _env_int(name: str, default: int) -> tuple[int, bool]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default, False
    return int(raw), True


def _env_str(name: str, default: str) -> tuple[str, bool]:
    raw = os.environ.get(name)
    if raw is None:
        return default, False
    stripped = raw.strip()
    if stripped == "":
        return default, False
    return stripped, True


def _env_optional_float(name: str, default: float) -> tuple[float, bool]:
    """Read float env var that allows empty-string sentinel (e.g. ``EDITOR_ZOOM``)."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default, False
    return float(raw), True


def load_config(overrides: Mapping[str, Any] | None = None) -> UniqConfig:
    """Load :class:`UniqConfig` from env vars and optional per-job overrides.

    Precedence (lowest → highest):

    1. Hardcoded :class:`UniqConfig` defaults.
    2. Env vars (§13.1 names, read directly from ``os.environ``).
    3. Profile overrides — applied by callers via
       :func:`bot.uniqueization.profiles.resolve_profile` then
       :func:`apply_profile` (callers usually do this in one helper).
    4. Per-job payload ``overrides`` (this argument).

    ``origin`` mapping records the precedence layer that won each key.
    """
    origin: dict[str, str] = {}
    values: dict[str, Any] = {}

    def take(key: str, env_value: Any, present: bool) -> None:
        if present:
            values[key] = env_value
            origin[key] = "env"

    # v2.0 baseline
    take("editor_version", *_env_str("EDITOR_VERSION", "v1"))
    if "editor_version" in values:
        values["editor_version"] = values["editor_version"].lower()
    take("editor_profile", *_env_str("EDITOR_PROFILE", "light"))
    if "editor_profile" in values:
        values["editor_profile"] = values["editor_profile"].lower()
    take("ffmpeg_timeout_s", *_env_float("EDITOR_FFMPEG_TIMEOUT_S", 360.0))
    take("ffmpeg_threads", *_env_str("EDITOR_FFMPEG_THREADS", ""))

    take("zoom", *_env_optional_float("EDITOR_ZOOM", 0.0))
    take("face_reframe_enabled", *_env_bool("EDITOR_FACE_REFRAME_ENABLED", True))
    take("yunet_model_path", *_env_str("EDITOR_YUNET_MODEL_PATH", ""))

    take("mirror_enabled", *_env_bool("EDITOR_MIRROR_ENABLED", True))
    take("mirror_duration_s", *_env_float("EDITOR_MIRROR_DURATION_S", 1.5))

    take("redraw_enabled", *_env_bool("EDITOR_REDRAW_ENABLED", False))
    take("colorgrade_enabled", *_env_bool("EDITOR_COLORGRADE_ENABLED", True))
    take("lut_path", *_env_str("EDITOR_LUT_PATH", ""))

    take("effects_dir", *_env_str("EDITOR_EFFECTS_DIR", ""))
    take("effects_opacity", *_env_float("EDITOR_EFFECTS_OPACITY", 0.15))

    take("music_remover_enabled", *_env_bool("EDITOR_MUSIC_REMOVER_ENABLED", False))
    take("music_remover_mode", *_env_str("EDITOR_MUSIC_REMOVER_MODE", "mute"))
    if "music_remover_mode" in values:
        values["music_remover_mode"] = values["music_remover_mode"].lower()

    take("audio_fx_enabled", *_env_bool("EDITOR_AUDIO_FX_ENABLED", True))
    take("audio_fx_wet", *_env_optional_float("EDITOR_AUDIO_FX_WET", 0.0))
    take("loudnorm_enabled", *_env_bool("EDITOR_LOUDNORM_ENABLED", True))

    take("phash_enabled", *_env_bool("EDITOR_PHASH_ENABLED", True))
    take("thumbnail_enabled", *_env_bool("EDITOR_THUMBNAIL_ENABLED", True))
    take("keep_uniq_artifacts", *_env_bool("EDITOR_KEEP_UNIQ_ARTIFACTS", False))

    # v1 carry-over (ffmpeg encoder defaults + logo overlay)
    take("video_crf", *_env_int("EDITOR_VIDEO_CRF", 23))
    take("video_preset", *_env_str("EDITOR_VIDEO_PRESET", "medium"))
    take("audio_bitrate", *_env_str("EDITOR_AUDIO_BITRATE", "128k"))
    take("output_width", *_env_int("EDITOR_OUTPUT_WIDTH", 1080))
    take("output_height", *_env_int("EDITOR_OUTPUT_HEIGHT", 1920))
    take("overlay_logo_path", *_env_str("OVERLAY_LOGO_PATH", ""))
    take("overlay_margin_px", *_env_int("OVERLAY_MARGIN_PX", 40))

    # v2.1 layer
    take("subtitle_enabled", *_env_bool("EDITOR_SUBTITLE_ENABLED", True))
    take(
        "subtitle_karaoke_enabled",
        *_env_bool("EDITOR_SUBTITLE_KARAOKE_ENABLED", True),
    )
    take(
        "subtitle_max_words_per_cue",
        *_env_int("EDITOR_SUBTITLE_MAX_WORDS_PER_CUE", 3),
    )
    take("blur_fill_enabled", *_env_bool("EDITOR_BLUR_FILL_ENABLED", True))
    take(
        "hook_emphasis_enabled",
        *_env_bool("EDITOR_HOOK_EMPHASIS_ENABLED", True),
    )
    take(
        "unique_distance_enabled",
        *_env_bool("EDITOR_UNIQUE_DISTANCE_ENABLED", True),
    )

    # v6 creative planner
    take("v6_enabled", *_env_bool("EDITOR_V6_ENABLED", False))
    take("v6_force_skip", *_env_bool("EDITOR_V6_FORCE_SKIP", False))
    take(
        "hook_dialogue_density_threshold",
        *_env_float("EDITOR_HOOK_DIALOGUE_DENSITY_THRESHOLD", 12.0),
    )
    take(
        "creative_intensity_ceiling_light",
        *_env_float("EDITOR_CREATIVE_INTENSITY_CEILING_LIGHT", 0.4),
    )
    take(
        "creative_intensity_ceiling_medium",
        *_env_float("EDITOR_CREATIVE_INTENSITY_CEILING_MEDIUM", 0.6),
    )
    take(
        "creative_intensity_ceiling_heavy",
        *_env_float("EDITOR_CREATIVE_INTENSITY_CEILING_HEAVY", 0.8),
    )
    # ``uniqueness_pct`` is the slider position on the Монтажёр
    # "🎯 Уникальность" sub-screen. Clamping happens here (instead of
    # in the slider callback only) so an unsafe env var on the host
    # can never push the worker outside ``[0, MAX_UNIQUENESS_PCT]``.
    from .randomization import clamp_pct as _clamp_uniqueness_pct
    _raw_pct, _raw_origin = _env_int("EDITOR_UNIQUENESS_PCT", 0)
    take("uniqueness_pct", _clamp_uniqueness_pct(_raw_pct), _raw_origin)

    # Apply per-job overrides last, recording origin
    if overrides:
        for key, value in overrides.items():
            if key.startswith("_"):
                continue
            if key not in {f.name for f in fields(UniqConfig)}:
                continue
            values[key] = value
            origin[key] = "override"

    # Default origins for keys not in values
    for fld in fields(UniqConfig):
        if fld.name.startswith("_"):
            continue
        if fld.name not in values:
            origin[fld.name] = "default"

    cfg = UniqConfig(**values, _origin=origin)
    return cfg


def config_origin(cfg: UniqConfig, key: str) -> str:
    """Return the precedence layer that won for ``key`` (``default`` / ``env`` /
    ``profile`` / ``override``)."""
    return cfg._origin.get(key, "default")


def apply_profile(cfg: UniqConfig, profile: UniqProfile) -> UniqConfig:
    """Apply a :class:`UniqProfile` to ``cfg``, respecting precedence.

    Profile values overwrite keys whose origin is currently ``default``
    or ``env`` (per §13.1 precedence: profile beats env). Keys already
    overridden at the per-job ``override`` layer are kept untouched.
    """
    origin = dict(cfg._origin)
    new_values: dict[str, Any] = {}
    profile_keys: dict[str, Any] = {
        "zoom": profile.zoom,
        "video_crf": profile.video_crf,
        "mirror_duration_s": profile.mirror_duration_s,
        "effects_opacity": profile.effects_opacity,
        "audio_fx_wet": profile.audio_fx_wet,
        "redraw_enabled": profile.redraw_enabled,
    }
    for key, value in profile_keys.items():
        layer = origin.get(key, "default")
        if layer in {"default", "env"}:
            new_values[key] = value
            origin[key] = f"profile:{profile.name}"
    return replace(cfg, **new_values, _origin=origin)
