"""ASS karaoke subtitle renderer — Part 5 of the Editor Agent split.

Generates `Advanced SubStation Alpha (ASS)`_ files for v2.1 (§8.1
intensity matrix, §8.3 ASS implementation detail). The renderer is a
**pure string templating step**: it takes a list of pre-decided
:class:`SubtitleCue` events (positions resolved upstream by
:mod:`.safe_area`, words remapped to render-time by
:func:`.timeline.remap_words`) and emits the ``.ass`` content as a
string. The optional :func:`render_ass` wrapper writes the string to
disk for the Part 7 worker that wires the file into
``filter_complex`` (``-vf subtitles=…``).

What lives here:

* Closed enum :data:`SubtitleStyle` of the four style ids exposed by
  Part 4's prompt: ``hook_overlay`` / ``karaoke_emphasis`` /
  ``plain_caption`` / ``minimalist``.
* :class:`StyleProfile` font / colour / size defaults per style id —
  matches the v2.0 reference look from §9.10 and §8.5.
* :func:`build_ass` — returns the full ASS file content as a single
  ``str``. Pure function. Same input → byte-identical output (the
  unit tests pin this against ``tests/fixtures/subtitles/*.ass``).
* :func:`render_ass` — writes ``build_ass`` output to disk with
  UTF-8 encoding (no BOM, as the spec mandates).

What does **not** live here:

* No ffmpeg invocation — Part 7 composes the ``-vf subtitles=…``
  filter using :func:`.runner.run_ffmpeg_atomic`.
* No whisper / no torch — the timestamped words arrive pre-aligned.
* No safe-area decision logic — :mod:`.safe_area` produced the
  positions; we trust the ``y_position`` on each cue.

.. _Advanced SubStation Alpha (ASS): https://en.wikipedia.org/wiki/SubStation_Alpha
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

#: Closed enum of style ids — exposed exactly as the Part 5 prompt
#: required. Maps to ASS ``Style:`` lines via :data:`STYLE_PROFILES`.
SubtitleStyle = Literal[
    "hook_overlay",
    "karaoke_emphasis",
    "plain_caption",
    "minimalist",
]


@dataclass(frozen=True, slots=True)
class StyleProfile:
    """ASS ``Style:`` line definition for one :data:`SubtitleStyle`.

    Colours follow the ASS ``&Hbbggrr`` convention (BGR, no alpha
    byte). Sizes are point-equivalent at the working resolution
    (``PlayResX`` × ``PlayResY``); the renderer keeps them stable so
    layouts look the same across landscape / portrait / square.
    """

    fontname: str = "Inter"
    fontsize: int = 56
    primary_colour: str = "&H00FFFFFF"  # white text
    secondary_colour: str = "&H0000FFFF"  # karaoke-on yellow
    outline_colour: str = "&H00000000"  # black outline
    back_colour: str = "&H64000000"  # 60% black shadow box
    bold: bool = True
    italic: bool = False
    outline: float = 2.5
    shadow: float = 1.0
    alignment: int = 2  # ASS \an2 = bottom-center
    margin_l: int = 60
    margin_r: int = 60
    margin_v: int = 80  # bottom margin (px) — overridden per-cue


#: Default profile per style id. Hook overlay is bigger and brighter
#: (§8.5 "larger 1.4× bolder"); karaoke uses the same dialogue base
#: with karaoke `\k` codes enabled; plain_caption drops karaoke;
#: minimalist drops outline weight + saturation entirely.
STYLE_PROFILES: dict[SubtitleStyle, StyleProfile] = {
    "hook_overlay": StyleProfile(
        fontsize=78,
        outline=3.0,
        shadow=1.5,
        margin_v=440,  # near top-third Y=0.20 of 1920px portrait
        alignment=8,  # \an8 = top-center
        primary_colour="&H00FFFFFF",
        secondary_colour="&H0000F5FF",  # warm gold
    ),
    "karaoke_emphasis": StyleProfile(
        fontsize=60,
        outline=2.5,
        shadow=1.0,
        primary_colour="&H00FFFFFF",
        secondary_colour="&H0000FFFF",
    ),
    "plain_caption": StyleProfile(
        fontsize=54,
        outline=2.0,
        shadow=1.0,
        bold=False,
        primary_colour="&H00FFFFFF",
    ),
    "minimalist": StyleProfile(
        fontsize=48,
        outline=1.5,
        shadow=0.0,
        bold=False,
        italic=False,
        outline_colour="&H00202020",
        primary_colour="&H00F0F0F0",
    ),
}


@dataclass(frozen=True, slots=True)
class SubtitleWord:
    """One karaoke-timed word in render-time.

    ``emphasized`` controls whether the renderer flags the word as
    bold/coloured during its highlight window — set by the planner
    (Part 9) from the analyzer's ``emphasis_candidates[]`` (§8.4).
    """

    text: str
    render_start_s: float
    render_end_s: float
    emphasized: bool = False


@dataclass(frozen=True, slots=True)
class SubtitleCue:
    """One on-screen subtitle event covering 1–3 words.

    The position and opacity are *decisions* — :mod:`.safe_area`
    already resolved overlap with faces/logo. The renderer trusts
    them verbatim.
    """

    words: tuple[SubtitleWord, ...]
    style: SubtitleStyle
    render_start_s: float
    render_end_s: float
    y_position: float = 0.80  # normalized 0..1
    opacity: float = 1.0  # 0..1
    drop: bool = False  # if True the renderer skips the cue entirely

    def __post_init__(self) -> None:
        if self.render_end_s < self.render_start_s:
            raise ValueError(
                f"cue end {self.render_end_s} before start {self.render_start_s}"
            )
        if not 0.0 <= self.y_position <= 1.0:
            raise ValueError(
                f"y_position must be in [0,1], got {self.y_position}"
            )
        if not 0.0 <= self.opacity <= 1.0:
            raise ValueError(
                f"opacity must be in [0,1], got {self.opacity}"
            )


# ---------------------------------------------------------------------------
# ASS time + escape helpers
# ---------------------------------------------------------------------------


def _ass_time(seconds: float) -> str:
    """Format a render-time second as ASS ``H:MM:SS.cs``.

    ASS uses centiseconds (1/100s) not millis. Anything beyond that
    is truncated, not rounded — ffmpeg's libass parser drops the
    extra digits anyway, and rounding could push end-time before
    start-time on adjacent cues.
    """
    if seconds < 0:
        seconds = 0.0
    total_cs = int(math.floor(seconds * 100))
    cs = total_cs % 100
    total_s = total_cs // 100
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _ass_escape(text: str) -> str:
    """Escape ASS metacharacters in dialogue text.

    libass treats ``{`` as the start of an override block; backslash
    is the override prefix. The newline / line-break sequences (``\\N``
    and ``\\n``) are inserted by callers explicitly — we don't
    accidentally produce them by escaping.
    """
    return (
        text.replace("\\", "\\\\")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def _opacity_to_alpha(opacity: float) -> str:
    """Convert 0..1 opacity to an ASS ``\\alpha&Hxx`` override.

    libass alpha is *inverse* — ``&H00`` is fully opaque, ``&HFF``
    fully transparent.
    """
    opacity = max(0.0, min(1.0, opacity))
    alpha = int(round((1.0 - opacity) * 255))
    return f"\\alpha&H{alpha:02X}&"


# ---------------------------------------------------------------------------
# Cue chunking (1..max_words_per_cue)
# ---------------------------------------------------------------------------


def chunk_words_into_cues(
    words: Sequence[SubtitleWord],
    *,
    style: SubtitleStyle,
    y_position: float,
    max_words_per_cue: int,
) -> list[SubtitleCue]:
    """Slice a flat word list into 1..N-word cues.

    Greedy left-to-right grouping — keeps a cue going until the
    word count hits ``max_words_per_cue``. Each cue's render
    timestamps span from the first word's start to the last word's
    end. The :data:`SubtitleStyle` and ``y_position`` are inherited.

    Used by tests and by Part 9 callers that have a pre-chunked
    plan; both bypass this function when they need different rules.
    """
    if max_words_per_cue < 1:
        raise ValueError(
            f"max_words_per_cue must be >= 1, got {max_words_per_cue}"
        )
    out: list[SubtitleCue] = []
    if not words:
        return out
    for start_i in range(0, len(words), max_words_per_cue):
        chunk = tuple(words[start_i : start_i + max_words_per_cue])
        out.append(
            SubtitleCue(
                words=chunk,
                style=style,
                render_start_s=chunk[0].render_start_s,
                render_end_s=chunk[-1].render_end_s,
                y_position=y_position,
            )
        )
    return out


# ---------------------------------------------------------------------------
# ASS file assembly
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AssOptions:
    """Renderer-level options that don't belong on individual cues.

    ``play_res_x``/``play_res_y`` set the ASS coordinate space; the
    same numbers must be used downstream by ffmpeg's ``subtitles``
    filter (libass auto-scales to actual frame size). Default is the
    portrait 1080×1920 the v2.0 baseline targets.
    """

    play_res_x: int = 1080
    play_res_y: int = 1920
    karaoke_enabled: bool = True
    max_words_per_cue: int = 3
    title: str = "lilush-bot2 v2.1 subtitles"
    style_profiles: dict[SubtitleStyle, StyleProfile] = field(
        default_factory=lambda: dict(STYLE_PROFILES)
    )


_SCRIPT_INFO_TEMPLATE = """[Script Info]
Title: {title}
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709
PlayResX: {play_res_x}
PlayResY: {play_res_y}
"""

_STYLE_HEADER = (
    "[V4+ Styles]\n"
    "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
    "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
    "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
    "Alignment, MarginL, MarginR, MarginV, Encoding\n"
)

_EVENTS_HEADER = (
    "[Events]\n"
    "Format: Layer, Start, End, Style, Name, MarginL, MarginR, "
    "MarginV, Effect, Text\n"
)


def _style_line(name: SubtitleStyle, profile: StyleProfile) -> str:
    return (
        f"Style: {name},{profile.fontname},{profile.fontsize},"
        f"{profile.primary_colour},{profile.secondary_colour},"
        f"{profile.outline_colour},{profile.back_colour},"
        f"{-1 if profile.bold else 0},"
        f"{-1 if profile.italic else 0},0,0,"
        f"100,100,0,0,1,"
        f"{profile.outline},{profile.shadow},"
        f"{profile.alignment},"
        f"{profile.margin_l},{profile.margin_r},{profile.margin_v},1\n"
    )


def _render_cue_text(
    cue: SubtitleCue, *, karaoke_enabled: bool
) -> str:
    """Build the ``Text`` field of a single Dialogue line.

    Karaoke is honoured only when (a) the global option is on, and
    (b) the cue's style supports it (``karaoke_emphasis`` and
    ``hook_overlay`` do; ``plain_caption`` and ``minimalist``
    don't). ``\\k`` durations are expressed in ASS centiseconds.
    """
    pos_override = (
        f"\\pos({cue.y_position * 1000:.0f},{cue.y_position * 1000:.0f})"
    )  # placeholder: actual ASS positioning uses MarginV; we keep
    # this hook for tests that want to inspect the raw override string.
    # The MarginV in the per-cue line below is the authoritative
    # vertical offset.
    _ = pos_override  # silence unused-name in case we deprecate
    use_karaoke = karaoke_enabled and cue.style in (
        "karaoke_emphasis",
        "hook_overlay",
    )
    parts: list[str] = []
    alpha_override = (
        _opacity_to_alpha(cue.opacity) if cue.opacity < 1.0 else ""
    )
    if alpha_override:
        parts.append("{" + alpha_override + "}")
    if not use_karaoke:
        # Static text — join words with a single space, optionally
        # bolding emphasized ones inline (``{\b1}…{\b0}``).
        joined: list[str] = []
        for w in cue.words:
            escaped = _ass_escape(w.text)
            if w.emphasized:
                joined.append("{\\b1}" + escaped + "{\\b0}")
            else:
                joined.append(escaped)
        parts.append(" ".join(joined))
        return "".join(parts)
    # Karaoke path — emit one `\k` override per word for the word's
    # *displayed* duration. ``\k`` uses centiseconds.
    for w in cue.words:
        dur_cs = max(1, int(round((w.render_end_s - w.render_start_s) * 100)))
        escaped = _ass_escape(w.text)
        if w.emphasized:
            parts.append(
                "{\\k" + str(dur_cs) + "\\b1}" + escaped + " {\\b0}"
            )
        else:
            parts.append("{\\k" + str(dur_cs) + "}" + escaped + " ")
    # Strip trailing space the loop adds for spacing between words.
    text = "".join(parts).rstrip(" ")
    return text


def _cue_margin_v(cue: SubtitleCue, play_res_y: int) -> int:
    """Convert a normalized Y (0..1) to ASS MarginV (px from anchor).

    For ``\\an2`` (bottom-center), MarginV is distance from the
    bottom edge. For ``\\an8`` (top-center, used by hook_overlay),
    MarginV is distance from the top edge. We pick the right one
    based on the style's stored alignment.
    """
    profile = STYLE_PROFILES[cue.style]
    if profile.alignment in (7, 8, 9):  # top row
        return max(0, int(round(cue.y_position * play_res_y)))
    if profile.alignment in (4, 5, 6):  # middle row
        # Translate normalized Y to an offset from the middle.
        return max(0, int(round((cue.y_position - 0.5) * play_res_y)))
    # bottom row (1/2/3): distance from the bottom.
    return max(0, int(round((1.0 - cue.y_position) * play_res_y)))


def _dialogue_line(
    cue: SubtitleCue,
    *,
    play_res_y: int,
    karaoke_enabled: bool,
) -> str:
    margin_v = _cue_margin_v(cue, play_res_y)
    text = _render_cue_text(cue, karaoke_enabled=karaoke_enabled)
    return (
        f"Dialogue: 0,"
        f"{_ass_time(cue.render_start_s)},"
        f"{_ass_time(cue.render_end_s)},"
        f"{cue.style},,0,0,{margin_v},,{text}\n"
    )


def build_ass(
    cues: Sequence[SubtitleCue],
    *,
    options: AssOptions | None = None,
) -> str:
    """Render a complete ``.ass`` file as a string.

    ``cues`` are emitted in order; dropped cues
    (:attr:`SubtitleCue.drop`) are excluded so the manifest's
    ``subtitle_dropped`` reason is the only place the drop is
    visible. The line count therefore equals
    ``len([c for c in cues if not c.drop])``.

    Style lines are emitted in the order keys appear in
    :data:`STYLE_PROFILES` (Python dicts preserve insertion order
    from 3.7+, so the result is deterministic for golden tests).
    """
    opts = options or AssOptions()

    parts: list[str] = []
    parts.append(
        _SCRIPT_INFO_TEMPLATE.format(
            title=opts.title,
            play_res_x=opts.play_res_x,
            play_res_y=opts.play_res_y,
        )
    )
    parts.append("\n")
    parts.append(_STYLE_HEADER)
    for style_name, profile in opts.style_profiles.items():
        parts.append(_style_line(style_name, profile))
    parts.append("\n")
    parts.append(_EVENTS_HEADER)
    for cue in cues:
        if cue.drop:
            continue
        if not cue.words:
            continue
        parts.append(
            _dialogue_line(
                cue,
                play_res_y=opts.play_res_y,
                karaoke_enabled=opts.karaoke_enabled,
            )
        )
    return "".join(parts)


def render_ass(
    cues: Sequence[SubtitleCue],
    output_path: str | Path,
    *,
    options: AssOptions | None = None,
) -> Path:
    """Write the ASS file to ``output_path`` and return the path.

    UTF-8 encoded without BOM (libass treats a BOM as a stray
    character and renders it as a glyph). The file is written
    atomically — temp suffix then ``Path.replace`` — so a crashed
    write never leaves a half-formed file behind for ffmpeg to
    choke on.
    """
    target = Path(output_path)
    content = build_ass(cues, options=options)
    target.parent.mkdir(parents=True, exist_ok=True)
    # ``foo.ass`` -> ``foo.tmp.ass`` (matches the runner.py temp
    # convention so on-disk artefacts share a single visual rhythm).
    tmp = target.with_name(target.stem + ".tmp" + target.suffix)
    tmp.write_text(content, encoding="utf-8", newline="\n")
    tmp.replace(target)
    return target


__all__ = [
    "STYLE_PROFILES",
    "AssOptions",
    "StyleProfile",
    "SubtitleCue",
    "SubtitleStyle",
    "SubtitleWord",
    "build_ass",
    "chunk_words_into_cues",
    "render_ass",
]
