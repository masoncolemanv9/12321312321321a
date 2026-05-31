"""Built-in Editor Agent v2 profiles: ``light`` / ``medium`` / ``heavy``.

Profile semantics, from ``final_spec_FULL.md`` §9.6 and §13.1:

* ``light``: closest to v1 output; defaults shipped on every install.
  Recommended for first-time v2 opt-in.
* ``medium``: balanced quality/CPU trade. Default ``EDITOR_PROFILE``
  when nothing is set in env (but only takes effect when
  ``EDITOR_VERSION`` ≥ ``v2``).
* ``heavy``: max-quality v2.0 output. Slower CPU, lower CRF, longer
  mirror window, heavier effects layer.

Profiles intentionally do NOT touch v2.1 / v6 toggles — those layers
have their own switches (subtitles, blur-fill, hook emphasis, …).
"""

from __future__ import annotations

from collections.abc import Mapping

from .config import UniqProfile

PROFILE_DEFAULTS: Mapping[str, UniqProfile] = {
    "light": UniqProfile(
        name="light",
        zoom=0.0,
        video_crf=23,
        mirror_duration_s=1.5,
        effects_opacity=0.10,
        audio_fx_wet=0.05,
        redraw_enabled=False,
        yunet_sample_rate_s=1.0,
    ),
    "medium": UniqProfile(
        name="medium",
        zoom=0.35,
        video_crf=21,
        mirror_duration_s=1.5,
        effects_opacity=0.15,
        audio_fx_wet=0.10,
        redraw_enabled=False,
        yunet_sample_rate_s=0.5,
    ),
    "heavy": UniqProfile(
        name="heavy",
        zoom=0.45,
        video_crf=19,
        mirror_duration_s=2.0,
        effects_opacity=0.20,
        audio_fx_wet=0.15,
        redraw_enabled=True,
        yunet_sample_rate_s=0.25,
    ),
}


def resolve_profile(name: str) -> UniqProfile:
    """Look up a built-in profile by name (case-insensitive).

    Falls back to ``light`` for an unknown name. Spec §13.1 explicitly
    treats profile as a soft selector — unknown names should not abort
    rendering. The manifest writer records the resolved profile name
    so debugging is easy.
    """
    key = (name or "").strip().lower()
    return PROFILE_DEFAULTS.get(key, PROFILE_DEFAULTS["light"])
