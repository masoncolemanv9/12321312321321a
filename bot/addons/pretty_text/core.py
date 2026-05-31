"""Markdown-ish → Telegram-HTML reformatter.

The user often copies long answers from chat UIs (ChatGPT, Devin, etc.)
and wants to forward them through Telegram without the layout falling
apart. Telegram messages support a small HTML subset
(<b>, <i>, <u>, <s>, <code>, <pre>, <a>, <blockquote>) but no tables,
no headings, no fenced code-block language tags, no list bullets.

This module turns a permissive Markdown-ish input into the strictest
Telegram-HTML it can, with two design rules:

1. **Never produce HTML Telegram will reject** — every emitted tag is in
   the documented allowlist and is well-balanced. The whole point of
   the feature is "paste anything, get a clean re-render", so a
   `parse_mode=HTML` Bad Request would defeat it entirely.
2. **Preserve visual structure** — tables become monospace ASCII boxes
   using box-drawing characters, list bullets stay bullets, headings
   stay visually heavier than body text.

The function is pure / side-effect free; call ``format_for_telegram(text)``
to get a single HTML string ready to pass as ``parse_mode="HTML"``.
"""

from __future__ import annotations

import html
import re
import unicodedata

__all__ = ["format_for_telegram"]


# Telegram has a 4096-char hard limit per message; we leave a little
# headroom for the header line ("✨ Красивый текст:") that the bot
# prepends when sending the result back.
TELEGRAM_HARD_LIMIT = 4096
SAFETY_BUDGET = 4000


# Inline pattern compiled once. Order matters: we want the longest /
# most-specific rule (e.g. ``**bold**``) to win over the shorter one
# (e.g. ``*italic*``) so we test them in that order.
_INLINE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # `code` — escape inner content, never re-interpret.
    (re.compile(r"`([^`\n]+)`"), "code"),
    # **bold**
    (re.compile(r"\*\*([^*\n]+)\*\*"), "b"),
    # __bold__
    (re.compile(r"__([^_\n]+)__"), "b"),
    # *italic* (single star, not part of **)
    (re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)"), "i"),
    # _italic_ (single underscore, not part of __)
    (re.compile(r"(?<!_)_([^_\n]+)_(?!_)"), "i"),
    # ~~strikethrough~~
    (re.compile(r"~~([^~\n]+)~~"), "s"),
]


_LINK_RE = re.compile(r"\[([^\]\n]+)\]\(([^)\s]+)\)")


def _render_inline(text: str) -> str:
    """Apply inline Markdown to a single line *after* HTML-escaping it.

    Strategy: pre-escape the whole string with ``html.escape`` so any
    raw ``<`` / ``&`` /``>`` from the user's input become safe entities,
    then walk our pattern list converting Markdown syntax to the
    matching Telegram-HTML tag. Because we escape first, our pattern
    regexes operate on a clean string with no HTML to confuse them.

    Links are handled separately to preserve the URL inside ``href``.
    """
    # Escape HTML metacharacters in the whole string first.
    safe = html.escape(text, quote=False)

    # Links: [label](url) → <a href="url">label</a>
    def _link_sub(m: re.Match[str]) -> str:
        label = m.group(1)
        url = m.group(2)
        # html.escape was already applied to whole string, so label/url
        # are already safe; we just need to put quotes around href.
        return f'<a href="{url}">{label}</a>'

    safe = _LINK_RE.sub(_link_sub, safe)

    # Apply inline patterns in declared order.
    for pat, tag in _INLINE_PATTERNS:
        safe = pat.sub(rf"<{tag}>\1</{tag}>", safe)

    return safe


# ---------------------------------------------------------------------- #
# Table rendering
# ---------------------------------------------------------------------- #


def _is_table_separator(line: str) -> bool:
    """Detect a Markdown header/body separator like ``|---|---|---|``."""
    stripped = line.strip()
    if not stripped.startswith("|") and "|" not in stripped:
        return False
    cells = [c.strip() for c in stripped.strip("|").split("|")]
    if not cells:
        return False
    return all(bool(re.fullmatch(r":?-{2,}:?", c)) for c in cells)


def _split_row(line: str) -> list[str]:
    """Split a ``| a | b | c |`` row into ``["a", "b", "c"]``."""
    stripped = line.strip().strip("|")
    return [c.strip() for c in stripped.split("|")]


def _visible_width(text: str) -> int:
    """Approximate the rendered width of ``text`` in a monospace font.

    Telegram's ``<pre>`` block uses a monospace font where most
    East-Asian characters and emoji take **two** column-cells. The
    standard heuristic is to widen any code point classified as
    ``Wide`` / ``Fullwidth`` in Unicode, plus combine-zero anything
    that's a combining mark.
    """
    w = 0
    for ch in text:
        if unicodedata.combining(ch):
            continue
        if unicodedata.east_asian_width(ch) in ("W", "F"):
            w += 2
        else:
            w += 1
    return w


def _pad(text: str, width: int) -> str:
    """Right-pad ``text`` so its monospace width equals ``width``."""
    pad = width - _visible_width(text)
    return text + (" " * pad if pad > 0 else "")


def _render_table(rows: list[list[str]]) -> str:
    """Render rows into an ASCII box-drawn table.

    Always produces something that looks fine inside Telegram's
    ``<pre>`` block. Columns are widened to fit the largest cell.
    Header row (first row) is separated from the body with a thicker
    rule; otherwise rows use a single line.
    """
    if not rows:
        return ""
    n_cols = max(len(r) for r in rows)
    # Pad short rows with empties so every row has n_cols cells.
    rows = [r + [""] * (n_cols - len(r)) for r in rows]
    widths = [
        max(_visible_width(rows[r][c]) for r in range(len(rows))) for c in range(n_cols)
    ]
    # Minimum width 1 so empty cells still get a space.
    widths = [max(w, 1) for w in widths]

    def horizontal(left: str, mid: str, right: str, fill: str) -> str:
        parts = [fill * (w + 2) for w in widths]
        return left + mid.join(parts) + right

    top = horizontal("┌", "┬", "┐", "─")
    sep_header = horizontal("├", "┼", "┤", "─")
    bottom = horizontal("└", "┴", "┘", "─")

    def row_line(cells: list[str]) -> str:
        padded = [" " + _pad(c, w) + " " for c, w in zip(cells, widths, strict=True)]
        return "│" + "│".join(padded) + "│"

    out_lines = [top, row_line(rows[0])]
    if len(rows) > 1:
        out_lines.append(sep_header)
        for r in rows[1:]:
            out_lines.append(row_line(r))
    out_lines.append(bottom)
    return "\n".join(out_lines)


# ---------------------------------------------------------------------- #
# Block-level parser
# ---------------------------------------------------------------------- #


_FENCE_RE = re.compile(r"^\s*```([\w+\-]*)\s*$")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_UL_RE = re.compile(r"^\s*[-*+]\s+(.*)$")
_OL_RE = re.compile(r"^\s*(\d+)[.)]\s+(.*)$")
_BLOCKQUOTE_RE = re.compile(r"^\s*>\s?(.*)$")
_HR_RE = re.compile(r"^\s*(?:[-*_]\s*){3,}$")


def _looks_like_table_line(line: str) -> bool:
    """Return True if ``line`` could plausibly be a table row.

    We don't require leading pipes — many copy-pasted tables drop them.
    A line is a table candidate if it contains at least one pipe AND
    has at least two non-empty cells when split by pipes.
    """
    if "|" not in line:
        return False
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    return len(cells) >= 2 and sum(1 for c in cells if c) >= 2


def _consume_table(lines: list[str], start: int) -> tuple[str, int]:
    """Eat a table block starting at index ``start``; return (rendered, next_idx).

    Tolerates two cases:

    1. **GFM-style**: header + ``|---|---|`` separator + body.
    2. **Plain pipe rows** (no separator) — still rendered as a table,
       with the first row treated as header.
    """
    i = start
    raw_rows: list[list[str]] = []
    saw_separator = False
    while i < len(lines):
        line = lines[i]
        if _is_table_separator(line):
            saw_separator = True
            i += 1
            continue
        if not _looks_like_table_line(line):
            break
        raw_rows.append(_split_row(line))
        i += 1
    # If we only matched 1 row and didn't see a separator, this isn't
    # really a table — back off and let the paragraph code handle it.
    if not saw_separator and len(raw_rows) < 2:
        return "", start
    rendered = _render_table(raw_rows)
    return rendered, i


def _consume_code_fence(lines: list[str], start: int) -> tuple[str, int]:
    """Eat a fenced code block starting at the fence at index ``start``."""
    m = _FENCE_RE.match(lines[start])
    if not m:
        return "", start
    i = start + 1
    body: list[str] = []
    while i < len(lines):
        if _FENCE_RE.match(lines[i]):
            i += 1  # consume closing fence
            break
        body.append(lines[i])
        i += 1
    return "\n".join(body), i


def _render_paragraph(buf: list[str]) -> str:
    """Render a buffered run of paragraph lines with inline formatting."""
    return "\n".join(_render_inline(line) for line in buf)


def _render_heading(level: int, text: str) -> str:
    """Render Markdown headings without HTML headings (Telegram lacks them).

    Strategy:
    - H1 → bold + leading 🔹 + extra blank line below
    - H2 → bold + leading ▸
    - H3+ → underlined bold
    """
    inner = _render_inline(text.strip())
    if level == 1:
        return f"🔹 <b>{inner}</b>"
    if level == 2:
        return f"▸ <b>{inner}</b>"
    return f"<b><u>{inner}</u></b>"


def _render_list_item(text: str, *, ordered_index: int | None) -> str:
    inner = _render_inline(text)
    if ordered_index is not None:
        return f"{ordered_index}. {inner}"
    return f"• {inner}"


def _render_blockquote(buf: list[str]) -> str:
    inner = "\n".join(_render_inline(line) for line in buf)
    return f"<blockquote>{inner}</blockquote>"


def format_for_telegram(text: str) -> str:
    """Convert ``text`` (Markdown-ish) into Telegram-HTML.

    Always returns a string safe to pass with ``parse_mode="HTML"``;
    if the input is empty or just whitespace, returns the empty string.
    """
    if not text or not text.strip():
        return ""

    # Normalise line endings, drop trailing whitespace per line.
    norm = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in norm.split("\n")]

    out_blocks: list[str] = []
    para_buf: list[str] = []
    bq_buf: list[str] = []

    def flush_paragraph() -> None:
        if para_buf:
            out_blocks.append(_render_paragraph(para_buf))
            para_buf.clear()

    def flush_blockquote() -> None:
        if bq_buf:
            out_blocks.append(_render_blockquote(bq_buf))
            bq_buf.clear()

    i = 0
    while i < len(lines):
        line = lines[i]

        # Blank line → end current block(s).
        if not line.strip():
            flush_paragraph()
            flush_blockquote()
            # Preserve the paragraph break as an empty block.
            out_blocks.append("")
            i += 1
            continue

        # Horizontal rule.
        if _HR_RE.match(line):
            flush_paragraph()
            flush_blockquote()
            out_blocks.append("━" * 24)
            i += 1
            continue

        # Fenced code block.
        if _FENCE_RE.match(line):
            flush_paragraph()
            flush_blockquote()
            body, next_i = _consume_code_fence(lines, i)
            out_blocks.append(f"<pre>{html.escape(body, quote=False)}</pre>")
            i = next_i
            continue

        # Table: header row followed by separator on the next line.
        if (
            _looks_like_table_line(line)
            and i + 1 < len(lines)
            and _is_table_separator(lines[i + 1])
        ):
            flush_paragraph()
            flush_blockquote()
            rendered, next_i = _consume_table(lines, i)
            if rendered:
                out_blocks.append(f"<pre>{html.escape(rendered, quote=False)}</pre>")
                i = next_i
                continue
            # fall through if table consume bailed out

        # Heading.
        m_h = _HEADING_RE.match(line)
        if m_h:
            flush_paragraph()
            flush_blockquote()
            level = len(m_h.group(1))
            out_blocks.append(_render_heading(level, m_h.group(2)))
            i += 1
            continue

        # Blockquote (consecutive ``> `` lines).
        m_bq = _BLOCKQUOTE_RE.match(line)
        if m_bq:
            flush_paragraph()
            bq_buf.append(m_bq.group(1))
            i += 1
            continue
        # Non-blockquote line ends any blockquote in progress.
        flush_blockquote()

        # Unordered list item.
        m_ul = _UL_RE.match(line)
        if m_ul:
            flush_paragraph()
            out_blocks.append(_render_list_item(m_ul.group(1), ordered_index=None))
            i += 1
            continue

        # Ordered list item.
        m_ol = _OL_RE.match(line)
        if m_ol:
            flush_paragraph()
            idx = int(m_ol.group(1))
            out_blocks.append(_render_list_item(m_ol.group(2), ordered_index=idx))
            i += 1
            continue

        # Default: accumulate into paragraph buffer.
        para_buf.append(line)
        i += 1

    flush_paragraph()
    flush_blockquote()

    # Join blocks back together, collapsing runs of empty blocks into
    # at most one blank line so the output isn't full of extra
    # whitespace gaps.
    rendered = "\n".join(out_blocks)
    rendered = re.sub(r"\n{3,}", "\n\n", rendered)
    rendered = rendered.strip()

    if len(rendered) > SAFETY_BUDGET:
        rendered = rendered[: SAFETY_BUDGET - 30].rstrip() + "\n\n…(обрезано)"
    return rendered
