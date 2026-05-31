"""Random-style presets for pretty_text.

Each preset takes the user's plain text and produces a Markdown-decorated
version. The cardinal rule: **never change a word**. Presets only insert
``**bold**``, ``*italic*``, emojis, line-breaks, indentation — all of
which `format_for_telegram` will then convert to Telegram-HTML.

A preset is a pure function ``(text: str) -> str``. The output is fed
straight into :func:`bot.addons.pretty_text.core.format_for_telegram`,
so a preset returning bare text is also valid (it just won't add any
flair).

The five presets here are intentionally different in tone so cycling
through them with the "🎲 Случайная стилистика" button gives the user
visibly different results to choose from. Order matches the public
:data:`PRESETS` tuple so saved-preset references survive code changes.
"""

from __future__ import annotations

import re
from collections.abc import Callable

__all__ = [
    "PRESETS",
    "PRESET_LABELS",
    "next_preset",
    "render_preset",
    "is_preset_id",
]


# ---- helpers used by multiple presets -----------------------------------


def _split_paragraphs(text: str) -> list[str]:
    """Split on blank-line boundaries, preserving non-empty paragraphs."""
    parts = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in parts if p.strip()]


_NUMBER_RE = re.compile(r"\b\d[\d,.]*\d|\b\d\b")
# Match the first 1-3 word-ish tokens at the start of a sentence — used
# by presets that want to "bold the lede".
_SENTENCE_LEDE_RE = re.compile(
    r"^([\w][\w\-']{0,15}(?:\s+[\w][\w\-']{0,15}){0,2})", flags=re.UNICODE
)


def _bold_numbers(text: str) -> str:
    """Wrap every number-like token in ``**``. Idempotent."""

    def _sub(m: re.Match[str]) -> str:
        s = m.group(0)
        # If already inside **, skip — checked via lookaround in caller.
        return f"**{s}**"

    return _NUMBER_RE.sub(_sub, text)


def _bold_first_words(text: str, n: int = 2) -> str:
    """Bold the first ``n``-ish words of each sentence-like span."""

    def _sub(m: re.Match[str]) -> str:
        words = m.group(1)
        return f"**{words}**"

    return _SENTENCE_LEDE_RE.sub(_sub, text)


# ---- presets ------------------------------------------------------------


def preset_emoji_news(text: str) -> str:
    """News-channel vibe: emoji at the head of each paragraph, bold lede,
    bold numbers. Single blank line between paragraphs.
    """
    paragraphs = _split_paragraphs(text)
    emoji_pool = ["📌", "📰", "📍", "🔥", "💡", "🎯", "⭐", "🟦", "🔔"]
    out: list[str] = []
    for i, p in enumerate(paragraphs):
        emoji = emoji_pool[i % len(emoji_pool)]
        body = _bold_numbers(_bold_first_words(p, n=2))
        out.append(f"{emoji} {body}")
    return "\n\n".join(out)


def preset_minimal_bold(text: str) -> str:
    """Minimalist: no emojis, only bold lede + bold numbers, kept tight.
    Good for serious / technical text.
    """
    paragraphs = _split_paragraphs(text)
    out: list[str] = []
    for p in paragraphs:
        body = _bold_numbers(_bold_first_words(p, n=2))
        out.append(body)
    return "\n\n".join(out)


def preset_bulleted(text: str) -> str:
    """Bullet-list style: each non-empty line becomes a ``- `` bullet,
    with the leading words bolded. Single-line input stays single-line.
    """
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    if len(lines) <= 1:
        # No structure to bullet — fallback to a softer style.
        return _bold_first_words(text.strip(), n=2)
    bulleted: list[str] = []
    for ln in lines:
        ln = ln.lstrip("-•* \t")
        bulleted.append(f"- {_bold_first_words(ln, n=2)}")
    return "\n".join(bulleted)


def preset_airy(text: str) -> str:
    """Spacious: insert a blank line after every sentence. Bold lede of
    each sentence. Good for "premium" / spacious mood.
    """
    # Sentence splitter — handles ., !, ? at end and trailing quotes/closers.
    pieces = re.split(r"(?<=[.!?…])\s+(?=[A-ZА-ЯЁ\d])", text.strip())
    pieces = [p.strip() for p in pieces if p.strip()]
    return "\n\n".join(_bold_first_words(p, n=2) for p in pieces)


def preset_callouts(text: str) -> str:
    """Telegram-post style: 🟦 callout emoji on the first paragraph,
    ⭐ on the last, plain on the middle, bold all numbers, italic the
    final sentence as a soft CTA.
    """
    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return text
    out: list[str] = []
    for i, p in enumerate(paragraphs):
        emoji = ""
        if i == 0:
            emoji = "🟦 "
        elif i == len(paragraphs) - 1:
            emoji = "⭐ "
        body = _bold_numbers(p)
        if i == len(paragraphs) - 1:
            # Wrap the FINAL sentence (within the paragraph) in italic.
            m = re.match(r"^(.*?)([^.!?…]+[.!?…]?)\s*$", body, flags=re.DOTALL)
            if m and m.group(2).strip():
                body = f"{m.group(1)}*{m.group(2).strip()}*"
        out.append(f"{emoji}{body}".strip())
    return "\n\n".join(out)


# ---- registry -----------------------------------------------------------


PRESETS: tuple[tuple[str, Callable[[str], str]], ...] = (
    ("emoji_news", preset_emoji_news),
    ("minimal_bold", preset_minimal_bold),
    ("bulleted", preset_bulleted),
    ("airy", preset_airy),
    ("callouts", preset_callouts),
)

PRESET_LABELS: dict[str, str] = {
    "emoji_news": "📰 Эмодзи-новости",
    "minimal_bold": "🪶 Минимал",
    "bulleted": "📋 Буллеты",
    "airy": "🌬 Просторно",
    "callouts": "🟦 Каналу-пост",
}


def is_preset_id(preset_id: str) -> bool:
    return preset_id in dict(PRESETS)


def next_preset(current: str | None) -> str:
    """Return the next preset id after ``current`` (wrap-around). If
    ``current`` is None or unknown, return the first preset id.
    """
    ids = [pid for pid, _ in PRESETS]
    if current not in ids:
        return ids[0]
    idx = (ids.index(current) + 1) % len(ids)
    return ids[idx]


def render_preset(preset_id: str, text: str) -> str:
    """Apply the preset's transform to ``text``. Unknown ids fall back
    to the first preset.
    """
    fn = dict(PRESETS).get(preset_id, PRESETS[0][1])
    return fn(text)
