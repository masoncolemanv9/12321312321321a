"""Pure-function ffmpeg ``-filter_complex`` builder for the v2.0 baseline.

This module owns the entire ffmpeg invocation **shape** for the v2.0
fused single-encode pipeline. It is deliberately I/O-free: it neither
runs ffmpeg nor reads the source file. Inputs go in (a
:class:`ProbeResult`, a :class:`FaceSampling`, a :class:`UniqConfig`),
a :class:`FilterGraph` dataclass comes out, and the caller (Part 3's
``EditorV2Worker``) feeds it to :func:`bot.uniqueization.runner.run_ffmpeg_atomic`.

Spec sections covered:

* §6.5 — v2.0 baseline crop math (face-aware reframe).
* §6.6 — aspect-aware vertical transform (1080×1920 target).
* §9.5 — 16-step pipeline order.
* §9.7 — optional ``ffmpeg_redraw`` cheap-redraw stage.
* §9.9 — logo overlay LAST among visual stages.
* §9.10 — ffmpeg command constraints (argv list, encoder flags, faststart).

Determinism: given identical inputs, this function returns
character-identical output. Tests use golden fixtures to enforce that.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .config import UniqConfig, UniqProfile
from .face_yunet import FaceSampling
from .probe import ProbeResult

logger = logging.getLogger(__name__)

OutputAspect = Literal["portrait", "landscape", "square"]

# Tight output canvas; matches the v1 default and §6.6 fixed target.
TARGET_W = 1080
TARGET_H = 1920


@dataclass(frozen=True, slots=True)
class FilterGraphOptions:
    """Optional knobs passed in per-job by :class:`EditorV2Worker`.

    Defaults are deliberately conservative so the v2.0 baseline matches
    the v1 visual output as closely as possible. v2.1 / v6 callers pass
    explicit values.
    """

    effects_inputs: tuple[Path, ...] = ()
    logo_input: Path | None = None
    music_intervals: tuple[tuple[float, float], ...] = ()
    redraw_enabled_override: bool | None = None
    mirror_window: tuple[float, float] | None = None  # explicit override


@dataclass(frozen=True, slots=True)
class FilterGraph:
    """Result of :func:`build_v20_graph`.

    Glued together by :class:`EditorV2Worker` into a single ffmpeg
    invocation:

    ``[ffmpeg, *input_args, "-filter_complex", filter_complex_str,
       "-map", mapped_streams["v"], "-map", mapped_streams["a"],
       *output_args, <output_path>]``

    ``warnings`` is the list of non-fatal issues to bubble into the
    manifest (e.g. ``logo_skipped_missing_file``, ``redraw_disabled``).
    """

    filter_complex_str: str
    input_args: tuple[str, ...]
    output_args: tuple[str, ...]
    mapped_streams: dict[str, str] = field(default_factory=dict)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    stages_applied: tuple[str, ...] = field(default_factory=tuple)


# ---------- helpers --------------------------------------------------


def _classify_aspect(width: int, height: int) -> OutputAspect:
    if height <= 0:
        return "landscape"
    if width > height:
        return "landscape"
    if width < height:
        return "portrait"
    return "square"


def _vertical_transform_chain() -> str:
    """§6.6 aspect-aware vertical transform — source-safe for any input.

    The two-branch ``if(gte(iw/ih, 1080/1920), …)`` expression keeps
    odd dimensions away from libx264 by leaning on ``-2`` (auto-even).
    """
    return (
        "scale=w='if(gte(iw/ih,1080/1920),-2,1080)':"
        "h='if(gte(iw/ih,1080/1920),1920,-2)',"
        f"crop={TARGET_W}:{TARGET_H},setsar=1"
    )


def _crop_math(
    center_pct: tuple[float, float], zoom: float
) -> tuple[int, int, int, int]:
    """§6.5 v2.0 baseline crop math returning ``(crop_w, crop_h, x, y)``.

    ``zoom`` is the **delta** above 1.0 (e.g. 0.35 for medium). The
    spec's ``zoom_factor = 1.0 + zoom`` is applied here. All four
    outputs are even integers so libx264 stays happy.
    """
    zoom_factor = 1.0 + max(0.0, zoom)
    crop_w = TARGET_W / zoom_factor
    crop_h = TARGET_H / zoom_factor
    center_x = center_pct[0] * TARGET_W
    center_y = center_pct[1] * TARGET_H
    x = max(0.0, min(TARGET_W - crop_w, center_x - crop_w / 2.0))
    y = max(0.0, min(TARGET_H - crop_h, center_y - crop_h / 2.0))
    return (
        _even(int(math.floor(crop_w))),
        _even(int(math.floor(crop_h))),
        _even(int(math.floor(x))),
        _even(int(math.floor(y))),
    )


def _even(n: int) -> int:
    return n if n % 2 == 0 else max(0, n - 1)


def _mirror_window(
    duration_s: float, configured_s: float
) -> tuple[float, float] | None:
    """§7.6.5 v2.0 fallback: 1.5s middle window; None for clips <3s."""
    if duration_s < 3.0:
        return None
    width = max(0.0, configured_s)
    if width <= 0.0:
        return None
    start = max(0.0, (duration_s - width) / 2.0)
    end = min(duration_s, start + width)
    if end <= start:
        return None
    return start, end


def _format_ts(value: float) -> str:
    """Format a timestamp/zoom number with stable 3-decimal precision."""
    return f"{value:.3f}"


# ---------- main builder ---------------------------------------------


def build_v20_graph(
    probe: ProbeResult,
    face: FaceSampling,
    cfg: UniqConfig,
    profile: UniqProfile,
    *,
    source_path: Path,
    start_s: float,
    end_s: float,
    options: FilterGraphOptions | None = None,
) -> FilterGraph:
    """Build the v2.0 baseline filter graph for one clip.

    ``profile`` is the resolved :class:`UniqProfile` (already merged
    into ``cfg`` by ``apply_profile``). It is passed separately so
    profile-specific overlay defaults stay obvious in callers' logs.

    Stages assembled, in §9.5 order:

    4  source seek (``-ss <start_s> -to <end_s> -i <source_path>``)
    5  aspect-aware vertical transform
    6  face-aware zoom/reframe (crop+scale to 1080×1920)
    7  optional cheap redraw (lanczos up→down)
    8  1.5s mirror middle window (``hflip=enable=between(t,a,b)``)
    9  colorgrade (eq + optional LUT)
    10 effects overlays
    11 logo overlay LAST
    12 music remover (mute|noise — no bundled music)
    13 audio FX wet/dry
    14 loudnorm (when enabled)

    Returns a :class:`FilterGraph`; never raises on optional-stage
    issues — those go into ``warnings`` and the manifest.
    """
    opts = options or FilterGraphOptions()
    warnings: list[str] = []
    stages: list[str] = []

    # ---- input args (stage 4: source seek) ---------------------------
    input_args: list[str] = [
        "-y",
        "-loglevel",
        "error",
        "-ss",
        _format_ts(start_s),
        "-to",
        _format_ts(end_s),
        "-i",
        str(source_path),
    ]
    input_index = 0  # source = 0:v / 0:a

    # Optional ancillary inputs registered in declared §9.5 order.
    # ffmpeg numbers inputs by their position among ``-i`` flags. Source
    # is always input 0; logo (if present) is always input 1; effects
    # come after.
    logo_input_idx: int | None = None
    next_input_idx = 1
    if opts.logo_input is not None:
        if opts.logo_input.exists():
            input_args.extend(["-i", str(opts.logo_input)])
            logo_input_idx = next_input_idx
            next_input_idx += 1
        else:
            warnings.append("logo_skipped_missing_file")
            logger.warning("logo overlay file missing: %s", opts.logo_input)

    effect_inputs: list[int] = []
    for eff in opts.effects_inputs:
        if not eff.exists():
            warnings.append("effect_skipped_missing_file")
            continue
        input_args.extend(["-stream_loop", "-1", "-i", str(eff)])
        effect_inputs.append(next_input_idx)
        next_input_idx += 1

    # ---- video chain assembly ----------------------------------------
    video_chain: list[str] = []
    current_label = f"[{input_index}:v]"

    # Stage 5: aspect-aware vertical transform.
    video_chain.append(f"{current_label}{_vertical_transform_chain()}[v_vert]")
    current_label = "[v_vert]"
    stages.append("aspect_transform")

    # Stage 6: face-aware zoom/reframe.
    if cfg.face_reframe_enabled and (cfg.zoom or 0.0) > 0.0:
        center = face.median_center_pct if not face.fallback else (0.5, 0.5)
        crop_w, crop_h, x, y = _crop_math(center, cfg.zoom)
        video_chain.append(
            f"{current_label}crop={crop_w}:{crop_h}:{x}:{y},"
            f"scale={TARGET_W}:{TARGET_H}[v_zoom]"
        )
        current_label = "[v_zoom]"
        stages.append("zoom_reframe")
        if face.fallback:
            warnings.append(f"face_fallback:{face.reason}")

    # Stage 7: optional cheap redraw (§9.7).
    redraw_enabled = (
        opts.redraw_enabled_override
        if opts.redraw_enabled_override is not None
        else cfg.redraw_enabled
    )
    if redraw_enabled:
        video_chain.append(
            f"{current_label}scale={TARGET_W * 2}:{TARGET_H * 2}:"
            f"flags=lanczos+accurate_rnd,"
            f"scale={TARGET_W}:{TARGET_H}:flags=lanczos+accurate_rnd[v_redraw]"
        )
        current_label = "[v_redraw]"
        stages.append("ffmpeg_redraw")

    # Stage 8: 1.5s mirror middle window (§7.6.5).
    mirror_win = (
        opts.mirror_window
        if opts.mirror_window is not None
        else _mirror_window(end_s - start_s, cfg.mirror_duration_s)
    )
    if cfg.mirror_enabled and mirror_win is not None:
        mstart, mend = mirror_win
        video_chain.append(
            f"{current_label}hflip=enable='between(t,"
            f"{_format_ts(mstart)},{_format_ts(mend)})'[v_mirror]"
        )
        current_label = "[v_mirror]"
        stages.append("mirror_1_5s")
    elif cfg.mirror_enabled:
        warnings.append("mirror_skipped_short_clip")

    # Stage 9: colorgrade (LUT optional).
    if cfg.colorgrade_enabled:
        cg_parts = ["eq=saturation=1.05:contrast=1.02"]
        if cfg.lut_path:
            lut_path = Path(cfg.lut_path)
            if lut_path.exists() and lut_path.suffix.lower() == ".cube":
                cg_parts.append(f"lut3d=file={lut_path.as_posix()}")
            else:
                warnings.append("lut_skipped_invalid")
        video_chain.append(f"{current_label}{','.join(cg_parts)}[v_grade]")
        current_label = "[v_grade]"
        stages.append("colorgrade")

    # Stage 10: effects overlays (each registered above as an input).
    for n, _eff_idx in enumerate(effect_inputs):
        eff_label_in = f"[{_eff_idx}:v]"
        eff_label_pre = f"[eff_{n}]"
        video_chain.append(
            f"{eff_label_in}scale={TARGET_W}:{TARGET_H},"
            f"format=yuva420p,colorchannelmixer=aa={cfg.effects_opacity:.2f}"
            f"{eff_label_pre}"
        )
        out_label = "[v_eff]" if n == len(effect_inputs) - 1 else f"[v_eff_{n}]"
        video_chain.append(
            f"{current_label}{eff_label_pre}overlay=shortest=1{out_label}"
        )
        current_label = out_label
    if effect_inputs:
        stages.append("effects")

    # Stage 11: logo LAST so it isn't mirrored / recoloured (§9.9).
    if logo_input_idx is not None:
        margin = max(0, int(cfg.overlay_margin_px))
        video_chain.append(
            f"{current_label}[{logo_input_idx}:v]"
            f"overlay=W-w-{margin}:H-h-{margin}[v_logo]"
        )
        current_label = "[v_logo]"
        stages.append("logo_overlay")

    # Final video label always known as [vout].
    if current_label != "[vout]":
        video_chain.append(f"{current_label}null[vout]")

    # ---- audio chain assembly ----------------------------------------
    audio_chain: list[str] = []
    current_audio_label: str | None = f"[{input_index}:a]"

    if probe.has_audio:
        # Stage 12: music remover (intervals → mute / lowpass).
        if cfg.music_remover_enabled and opts.music_intervals:
            mode = cfg.music_remover_mode
            if mode == "mute":
                expr = "+".join(
                    f"between(t,{_format_ts(s)},{_format_ts(e)})"
                    for s, e in opts.music_intervals
                )
                audio_chain.append(
                    f"{current_audio_label}volume=enable='{expr}':volume=0[a_mute]"
                )
                current_audio_label = "[a_mute]"
                stages.append("music_remover_mute")
            elif mode == "noise":
                expr = "+".join(
                    f"between(t,{_format_ts(s)},{_format_ts(e)})"
                    for s, e in opts.music_intervals
                )
                audio_chain.append(
                    f"{current_audio_label}lowpass=f=300:enable='{expr}'[a_noise]"
                )
                current_audio_label = "[a_noise]"
                stages.append("music_remover_lowpass")
            else:
                warnings.append("music_remover_mode_unsupported")

        # Stage 13: audio FX wet/dry (modest; v2.0 uses aecho only).
        wet = max(0.0, min(1.0, cfg.audio_fx_wet))
        if cfg.audio_fx_enabled and wet > 0.0:
            audio_chain.append(
                f"{current_audio_label}"
                f"aecho=0.8:0.5:60|120:{wet:.2f}|{wet:.2f}[a_fx]"
            )
            current_audio_label = "[a_fx]"
            stages.append("audio_fx")

        # Stage 14: loudnorm.
        if cfg.loudnorm_enabled:
            audio_chain.append(
                f"{current_audio_label}loudnorm=I=-16:LRA=11:TP=-1.5[a_norm]"
            )
            current_audio_label = "[a_norm]"
            stages.append("loudnorm")

        if current_audio_label != "[aout]":
            audio_chain.append(f"{current_audio_label}anull[aout]")
        audio_map = "[aout]"
    else:
        # Silent source → map source video only; no audio chain.
        audio_chain = []
        audio_map = ""
        warnings.append("video_only_source")

    filter_complex_str = ";".join(video_chain + audio_chain)

    # ---- output args (§9.10) ----------------------------------------
    output_args: list[str] = [
        "-c:v",
        "libx264",
        "-preset",
        cfg.video_preset,
        "-crf",
        str(cfg.video_crf),
        "-pix_fmt",
        "yuv420p",
    ]
    if cfg.ffmpeg_threads:
        output_args.extend(["-threads", cfg.ffmpeg_threads])
    if audio_map:
        output_args.extend(
            ["-c:a", "aac", "-b:a", cfg.audio_bitrate, "-ar", "48000"]
        )
    else:
        output_args.extend(["-an"])
    output_args.extend(["-movflags", "+faststart"])

    mapped: dict[str, str] = {"v": "[vout]"}
    if audio_map:
        mapped["a"] = audio_map

    return FilterGraph(
        filter_complex_str=filter_complex_str,
        input_args=tuple(input_args),
        output_args=tuple(output_args),
        mapped_streams=mapped,
        warnings=tuple(warnings),
        stages_applied=tuple(stages),
    )


# ---------- convenience export --------------------------------------


def serialize_filter_complex(graph: FilterGraph) -> str:
    """Identity wrapper kept for callers that want a stable name when
    writing the long-form ``filter_complex.txt`` sidecar."""
    return graph.filter_complex_str
