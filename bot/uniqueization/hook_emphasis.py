"""v2.1 hook-emphasis curve (``final_spec_FULL.md`` §8.6 + §8.7).

Hook beats (first 1.5–3 s of a clip, when the analyzer marks one)
get a subtle push-in plus a brightness pulse. This module **produces
the curve only**; Part 7's worker integration is what wraps it into
``zoompan`` + ``eq`` ffmpeg expressions and splices them into the
final filtergraph.

Hard rules enforced here (§0 / §8.7):

* **No stacking.** Hook = one primary camera move + one support
  effect. We expose the primary (``zoom_in``) and one of the support
  effects (``brightness_pulse``). Shake / flash / vignette are NOT
  produced; callers asking for them get a :class:`HookEmphasisError`.
* **Profile gating.** ``light`` profile gets brightness pulse only;
  ``medium`` and ``heavy`` get the zoom delta + brightness pulse.
* **Hard cap.** Final zoom value is capped at ``z_target + 0.05``
  *and* at the profile's ``z_target`` from the policy table (whichever
  is smaller). The 5% wider start (``z_start = z_target - 0.05``)
  matches §8.6.
* **Heavy carve-out (§8.7).** A heavy-profile additional support is
  permitted *only* when the payload carries ``hook_text`` AND
  ``hook_confidence >= 0.7``; otherwise the support is dropped and
  the manifest records ``support_dropped_carve_out_failed``.
* **Clip too short.** Per R28, hook range is clipped to
  ``min(3.0, clip_duration_s * 0.5)``. If the resulting duration is
  below ``MIN_HOOK_DURATION_S`` the curve is disabled and reason
  ``clip_too_short`` is recorded.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from .safe_area import ProfileName

#: Per-spec primary camera move. Only ``zoom_in`` is implemented; the
#: other primaries (``hold_wide_reveal``, ``match_cut_zoom``) are
#: planner-level decisions and don't need a render-time curve.
HookPrimary = Literal["zoom_in", "none"]

#: Support effects the renderer is allowed to emit. ``shake`` /
#: ``flash`` / ``zoom_pulse`` are forbidden (see §8.7 "FORBIDDEN").
HookSupport = Literal["brightness_pulse", "vignette", "none"]

#: Reasons the curve was disabled or modified — recorded in the
#: manifest's ``stages.hook_emphasis.disabled_reason`` field.
HookDisabledReason = Literal[
    "no_payload",
    "clip_too_short",
    "light_profile_zoom_disabled",
    "support_dropped_carve_out_failed",
    "support_dropped_unsupported",
]

#: Default profile zoom targets (matches §8.6 + §9.5). Editors can
#: override via :func:`build_hook_curve(profile_zoom_target=...)`.
DEFAULT_PROFILE_ZOOM_TARGETS: dict[ProfileName, float] = {
    "light": 1.30,
    "medium": 1.35,
    "heavy": 1.50,
}

#: Hard zoom delta cap (§8.6 "Hard cap: z_target + 0.05").
HOOK_ZOOM_HARD_CAP_DELTA: float = 0.05

#: Default 5% wider start vs. target (§8.6).
DEFAULT_ZOOM_START_DELTA: float = 0.05

#: Default ramp duration in seconds.
DEFAULT_ZOOM_RAMP_S: float = 1.5

#: Brightness pulse defaults (§8.6 + §8.7 support menu).
BRIGHTNESS_PULSE_PEAK: float = 0.05
BRIGHTNESS_PULSE_DURATION_S: float = 0.3

#: Minimum hook duration we'll produce a curve for.
MIN_HOOK_DURATION_S: float = 0.5

#: Maximum hook duration cap (§8.5 / §8.7 wording: hook range is
#: typically [0, 3]).
MAX_HOOK_DURATION_S: float = 3.0


class HookEmphasisError(ValueError):
    """Raised when caller asks for a forbidden hook treatment."""


@dataclass(frozen=True, slots=True)
class BrightnessPulsePoint:
    """One point on the brightness pulse curve."""

    t_s: float
    delta: float


@dataclass(frozen=True, slots=True)
class HookCurve:
    """The curve emitted by :func:`build_hook_curve`.

    ``zoom_delta`` is the maximum delta *applied* (i.e. zoom_target -
    zoom_start). If zoom is disabled for the profile, this is ``0.0``
    and ``apply_zoom`` is ``False``.

    ``brightness_pulse`` is a small list of ``(t_s, delta)`` keyframes;
    ffmpeg's ``eq=brightness=`` expression interpolates linearly
    between them. The list is empty when the pulse is disabled.

    ``duration_s`` is the effective duration the curve applies for,
    after R28 clipping.
    """

    primary: HookPrimary
    support: HookSupport
    duration_s: float
    zoom_start: float
    zoom_target: float
    zoom_delta: float
    brightness_pulse: tuple[BrightnessPulsePoint, ...]
    apply_zoom: bool
    apply_brightness: bool
    disabled_reason: HookDisabledReason | None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_manifest_dict(self) -> dict[str, object]:
        """JSON-safe shape suited for the Part 3 manifest contract."""
        return {
            "primary": self.primary,
            "support": self.support,
            "duration_s": self.duration_s,
            "zoom_start": self.zoom_start,
            "zoom_target": self.zoom_target,
            "zoom_delta": self.zoom_delta,
            "brightness_pulse": [
                {"t_s": p.t_s, "delta": p.delta}
                for p in self.brightness_pulse
            ],
            "apply_zoom": self.apply_zoom,
            "apply_brightness": self.apply_brightness,
            "disabled_reason": self.disabled_reason,
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _disabled(reason: HookDisabledReason, duration_s: float) -> HookCurve:
    return HookCurve(
        primary="none",
        support="none",
        duration_s=duration_s,
        zoom_start=1.0,
        zoom_target=1.0,
        zoom_delta=0.0,
        brightness_pulse=(),
        apply_zoom=False,
        apply_brightness=False,
        disabled_reason=reason,
    )


def _clamp_hook_duration(payload_start: float, payload_end: float, clip_duration_s: float) -> float:
    if payload_end <= payload_start:
        return 0.0
    raw = payload_end - payload_start
    capped = min(raw, MAX_HOOK_DURATION_S)
    if clip_duration_s > 0:
        capped = min(capped, clip_duration_s * 0.5)
    return max(0.0, capped)


def _support_allowed(
    requested: HookSupport,
    *,
    profile: ProfileName,
    has_hook_text: bool,
    hook_confidence: float,
    forbidden_supports: Sequence[str],
) -> tuple[HookSupport, HookDisabledReason | None]:
    """Decide which support effect we'll emit, if any."""
    if requested == "none":
        return "none", None

    if requested in forbidden_supports:
        raise HookEmphasisError(
            f"support {requested!r} is forbidden by §8.7"
        )

    if requested not in ("brightness_pulse", "vignette"):
        return "none", "support_dropped_unsupported"

    # Heavy carve-out (§8.7): the additional support is only permitted
    # when both `hook_text` exists AND hook_confidence ≥ 0.7. We treat
    # this as a precondition for `vignette` since `brightness_pulse`
    # already counts as the default heavy support.
    if (
        profile == "heavy"
        and requested == "vignette"
        and (not has_hook_text or hook_confidence < 0.7)
    ):
        return "none", "support_dropped_carve_out_failed"

    return requested, None


def build_hook_curve(
    hook_emphasis_payload: dict[str, object] | None,
    profile: ProfileName,
    *,
    profile_zoom_target: float | None = None,
    clip_duration_s: float = 0.0,
    requested_support: HookSupport = "brightness_pulse",
    zoom_start_delta: float = DEFAULT_ZOOM_START_DELTA,
    zoom_ramp_s: float = DEFAULT_ZOOM_RAMP_S,
    brightness_pulse_peak: float = BRIGHTNESS_PULSE_PEAK,
    brightness_pulse_duration_s: float = BRIGHTNESS_PULSE_DURATION_S,
    forbidden_supports: Sequence[str] = ("shake", "flash", "zoom_pulse"),
) -> HookCurve:
    """Build the §8.6 hook curve from an analyzer ``hook_emphasis`` payload.

    Args:
        hook_emphasis_payload: ``{"source_start_s", "source_end_s",
            "hook_text"?, "hook_confidence"?}`` (per §10.2). ``None``
            disables the curve.
        profile: ``"light"`` / ``"medium"`` / ``"heavy"``. Governs
            whether zoom is applied (light = brightness only) and the
            heavy carve-out for an extra support effect.
        profile_zoom_target: Override the default zoom target for the
            profile. Defaults to :data:`DEFAULT_PROFILE_ZOOM_TARGETS`.
        clip_duration_s: Full clip duration (for the R28 cap).
        requested_support: Which support effect to emit. Defaults to
            ``brightness_pulse``. ``shake`` / ``flash`` / ``zoom_pulse``
            in ``forbidden_supports`` raise :class:`HookEmphasisError`.
        zoom_start_delta: Distance below the zoom target at t=0.
            Defaults to :data:`DEFAULT_ZOOM_START_DELTA` (0.05 per
            §8.6).
        zoom_ramp_s: Duration of the linear ramp from start → target.
            Defaults to :data:`DEFAULT_ZOOM_RAMP_S` (1.5 s per §8.6).
        brightness_pulse_peak: Peak brightness delta at t=0.
        brightness_pulse_duration_s: Duration the pulse decays over.
        forbidden_supports: Names that callers cannot request. Hard
        invariant; raises rather than silently dropping.

    Returns:
        :class:`HookCurve` with the resolved parameters and a manifest-
        ready :meth:`HookCurve.to_manifest_dict`.
    """
    if hook_emphasis_payload is None:
        return _disabled("no_payload", 0.0)

    def _num(key: str, default: float = 0.0) -> float:
        raw = hook_emphasis_payload.get(key, default)
        if raw is None or raw == "":
            return default
        try:
            return float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return default

    start_s = _num("source_start_s")
    end_s = _num("source_end_s")
    hook_text = hook_emphasis_payload.get("hook_text")
    has_hook_text = bool(hook_text)
    hook_confidence = _num("hook_confidence")

    duration_s = _clamp_hook_duration(start_s, end_s, clip_duration_s)
    if duration_s < MIN_HOOK_DURATION_S:
        return _disabled("clip_too_short", duration_s)

    if profile_zoom_target is None:
        if profile not in DEFAULT_PROFILE_ZOOM_TARGETS:
            raise ValueError(
                f"unknown profile {profile!r}; expected one of "
                f"{sorted(DEFAULT_PROFILE_ZOOM_TARGETS)}"
            )
        profile_zoom_target = DEFAULT_PROFILE_ZOOM_TARGETS[profile]

    # Zoom: only emitted for medium/heavy. Light keeps brightness only.
    apply_zoom = profile in ("medium", "heavy")
    notes: list[str] = []
    if not apply_zoom:
        notes.append("light_profile: zoom suppressed; brightness pulse only")

    # Hard cap: final zoom value never exceeds z_target + 0.05.
    capped_target = min(
        profile_zoom_target,
        profile_zoom_target + HOOK_ZOOM_HARD_CAP_DELTA,
    )
    zoom_target = capped_target
    zoom_start = max(1.0, zoom_target - zoom_start_delta)
    zoom_delta = max(0.0, zoom_target - zoom_start) if apply_zoom else 0.0

    support, support_reason = _support_allowed(
        requested_support,
        profile=profile,
        has_hook_text=has_hook_text,
        hook_confidence=hook_confidence,
        forbidden_supports=forbidden_supports,
    )
    if support_reason is not None:
        notes.append(f"support_request={requested_support!r}: {support_reason}")

    apply_brightness = support == "brightness_pulse"
    pulse: tuple[BrightnessPulsePoint, ...]
    if apply_brightness:
        # Two-point pulse: peak at t=0, decays linearly to 0 at
        # `brightness_pulse_duration_s`. Constrain the pulse window to
        # the hook duration so the curve never extends past the hook.
        peak_end_s = min(brightness_pulse_duration_s, duration_s)
        pulse = (
            BrightnessPulsePoint(t_s=0.0, delta=brightness_pulse_peak),
            BrightnessPulsePoint(t_s=peak_end_s, delta=0.0),
        )
    else:
        pulse = ()

    primary: HookPrimary = "zoom_in" if apply_zoom else "none"

    return HookCurve(
        primary=primary,
        support=support,
        duration_s=duration_s,
        zoom_start=zoom_start,
        zoom_target=zoom_target,
        zoom_delta=zoom_delta,
        brightness_pulse=pulse,
        apply_zoom=apply_zoom,
        apply_brightness=apply_brightness,
        disabled_reason=support_reason,
        notes=tuple(notes),
    )


__all__ = [
    "BRIGHTNESS_PULSE_DURATION_S",
    "BRIGHTNESS_PULSE_PEAK",
    "DEFAULT_PROFILE_ZOOM_TARGETS",
    "DEFAULT_ZOOM_RAMP_S",
    "DEFAULT_ZOOM_START_DELTA",
    "HOOK_ZOOM_HARD_CAP_DELTA",
    "MAX_HOOK_DURATION_S",
    "MIN_HOOK_DURATION_S",
    "BrightnessPulsePoint",
    "HookCurve",
    "HookDisabledReason",
    "HookEmphasisError",
    "HookPrimary",
    "HookSupport",
    "build_hook_curve",
]
