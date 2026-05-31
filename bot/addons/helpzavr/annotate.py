"""Pillow-based annotation: red rectangle + arrow + instruction box on top of a screenshot.

This module is pure code (no AI). It takes:
    - a base screenshot
    - a target bbox (x1, y1, x2, y2) in pixel coordinates
    - an instruction string and an example string

and returns a new PIL.Image with:
    - a red rectangle around the target
    - a callout box with title/instruction/example text
    - a thick red arrow from the box to the rectangle
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/Library/Fonts/Arial.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
]


def find_font(size: int) -> ImageFont.ImageFont:
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


@dataclass
class Style:
    color: tuple[int, int, int] = (220, 30, 30)
    arrow_width: int = 12
    arrow_head: int = 46
    rect_pad: int = 6
    rect_layers: int = 5
    box_fill: tuple[int, int, int, int] = (255, 245, 230, 240)
    box_text: tuple[int, int, int] = (30, 30, 30)
    box_title: tuple[int, int, int] = (180, 20, 20)


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    style: Style,
) -> None:
    x1, y1 = start
    x2, y2 = end
    angle = math.atan2(y2 - y1, x2 - x1)

    shaft_end = (
        x2 - (style.arrow_head * 0.55) * math.cos(angle),
        y2 - (style.arrow_head * 0.55) * math.sin(angle),
    )
    draw.line([start, shaft_end], fill=style.color, width=style.arrow_width)
    r = style.arrow_width // 2
    draw.ellipse([x1 - r, y1 - r, x1 + r, y1 + r], fill=style.color)
    draw.ellipse(
        [shaft_end[0] - r, shaft_end[1] - r, shaft_end[0] + r, shaft_end[1] + r],
        fill=style.color,
    )

    spread = math.pi / 5.5
    left = (
        x2 - style.arrow_head * math.cos(angle - spread),
        y2 - style.arrow_head * math.sin(angle - spread),
    )
    right = (
        x2 - style.arrow_head * math.cos(angle + spread),
        y2 - style.arrow_head * math.sin(angle + spread),
    )
    draw.polygon([end, left, right], fill=style.color, outline=style.color)


def _wrap(
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
    draw: ImageDraw.ImageDraw,
) -> list[str]:
    if not text:
        return [""]
    words = text.split()
    lines: list[str] = []
    cur = ""
    for w in words:
        test = (cur + " " + w).strip()
        if draw.textlength(test, font=font) <= max_width:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _font_size(font: ImageFont.ImageFont) -> int:
    return getattr(font, "size", 22)


def _layout_box(
    img_w: int,
    img_h: int,
    bbox: tuple[int, int, int, int],
    box_w: int,
    box_h: int,
) -> tuple[int, int]:
    """Pick a callout-box position that does **not** overlap ``bbox``.

    Tries 8 anchor positions in priority order and returns the first one
    that fits inside the image margin and doesn't intersect the bbox
    (with a small buffer). If nothing fits cleanly, picks the position
    with the smallest overlap.
    """
    x1, y1, x2, y2 = bbox
    margin = 24
    buf = 30  # keep the callout this many px clear of the bbox edge
    bbox_buf = (x1 - buf, y1 - buf, x2 + buf, y2 + buf)
    bbox_cx, bbox_cy = (x1 + x2) // 2, (y1 + y2) // 2

    # Anchor positions: right-of, left-of, above, below, plus four diagonals.
    raw = [
        # priority, bx, by
        (0, x2 + buf, max(margin, y1 - box_h // 4)),               # right
        (1, x1 - box_w - buf, max(margin, y1 - box_h // 4)),       # left
        (2, max(margin, bbox_cx - box_w // 2), y1 - box_h - buf),  # above
        (3, max(margin, bbox_cx - box_w // 2), y2 + buf),          # below
        (4, x2 + buf, y2 + buf),                                   # br
        (5, x1 - box_w - buf, y2 + buf),                           # bl
        (6, x2 + buf, y1 - box_h - buf),                           # tr
        (7, x1 - box_w - buf, y1 - box_h - buf),                   # tl
    ]

    def fits(bx: int, by: int) -> bool:
        return (
            bx >= margin
            and by >= margin
            and bx + box_w <= img_w - margin
            and by + box_h <= img_h - margin
        )

    # First pass — strict: fully in-image and no bbox overlap.
    for _, bx, by in raw:
        if fits(bx, by) and not _intersects((bx, by, bx + box_w, by + box_h), bbox_buf):
            return bx, by

    # Second pass — relax bbox buffer: in-image, no overlap with raw bbox.
    for _, bx, by in raw:
        if fits(bx, by) and not _intersects((bx, by, bx + box_w, by + box_h), bbox):
            return bx, by

    # Third pass — clamp each candidate to image bounds, pick the one with
    # the smallest overlap area with bbox. This guarantees we never overlap
    # less if a clean spot exists, and gracefully degrades for tiny images.
    best: tuple[float, int, int] | None = None  # (overlap_area, bx, by)
    for _, bx, by in raw:
        cx = max(margin, min(img_w - box_w - margin, bx))
        cy = max(margin, min(img_h - box_h - margin, by))
        area = _overlap_area((cx, cy, cx + box_w, cy + box_h), bbox)
        # Prefer positions farther from bbox centre when overlap is equal.
        dist = -((cx + box_w // 2 - bbox_cx) ** 2 + (cy + box_h // 2 - bbox_cy) ** 2)
        key = (area, dist)
        if best is None or key < (best[0], -((best[1] + box_w // 2 - bbox_cx) ** 2 + (best[2] + box_h // 2 - bbox_cy) ** 2)):
            best = (area, cx, cy)
    assert best is not None
    return best[1], best[2]


def _overlap_area(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> int:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0
    return (ix2 - ix1) * (iy2 - iy1)


def _intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _arrow_endpoints(
    bx: int,
    by: int,
    box_w: int,
    box_h: int,
    bbox: tuple[int, int, int, int],
    pad: int,
) -> tuple[tuple[float, float], tuple[float, float]]:
    x1, y1, x2, y2 = bbox
    box_cx = bx + box_w / 2
    box_cy = by + box_h / 2
    rect_cx = (x1 + x2) / 2
    rect_cy = (y1 + y2) / 2

    if box_cx < x1 - 5:
        sx, sy = bx + box_w, box_cy
    elif box_cx > x2 + 5:
        sx, sy = bx, box_cy
    elif box_cy < y1 - 5:
        sx, sy = box_cx, by + box_h
    else:
        sx, sy = box_cx, by

    if abs(rect_cx - sx) > abs(rect_cy - sy):
        ey = rect_cy
        ex = x1 - pad if sx < rect_cx else x2 + pad
    else:
        ex = rect_cx
        ey = y1 - pad if sy < rect_cy else y2 + pad
    return (sx, sy), (ex, ey)


def annotate_arrow_only(
    base: Image.Image,
    tip_px: tuple[int, int],
    instruction: str,
    style: Style | None = None,
    *,
    title: str | None = "Ответ:",
    no_box_fill: bool = False,
) -> Image.Image:
    """Draw a callout box with text + arrow pointing at ``tip_px`` (no rect).

    Used for the arrow-only fallback after rectangle verification fails:
    when we can't precisely box the object, we at least point at a
    specific pixel.
    """
    style = style or Style()
    img = base.convert("RGB").copy()
    W, H = img.size
    tx, ty = max(0, min(W - 1, int(tip_px[0]))), max(0, min(H - 1, int(tip_px[1])))
    draw = ImageDraw.Draw(img, "RGBA")

    # Treat the tip as a small bbox for layout purposes — we need the callout
    # to stay clear of the tip (so the arrow has room) and pick a clean spot.
    tip_bbox = (tx - 18, ty - 18, tx + 18, ty + 18)
    (
        bx, by, box_w, box_h,
        title_font, body_font,
        title_lines, body_lines, _el_lines, _e_lines,
    ) = _fit_callout(W, H, tip_bbox, title, instruction, None, "", draw)
    line_h_title = _font_size(title_font) + 6
    line_h_body = _font_size(body_font) + 4

    draw.rounded_rectangle(
        [bx, by, bx + box_w, by + box_h],
        radius=18,
        fill=None if no_box_fill else style.box_fill,
        outline=style.color,
        width=3,
    )
    cy = by + 14
    for line in title_lines:
        draw.text((bx + 16, cy), line, font=title_font, fill=style.box_title)
        cy += line_h_title
    if title_lines:
        cy += 6
    for line in body_lines:
        draw.text((bx + 16, cy), line, font=body_font, fill=style.box_text)
        cy += line_h_body

    # Arrow start: pick the box edge **on the side facing the tip**.
    box_cx = bx + box_w / 2
    box_cy = by + box_h / 2
    dx, dy = tx - box_cx, ty - box_cy
    if abs(dx) > abs(dy):
        sx = bx + box_w if dx > 0 else bx
        sy = box_cy
    else:
        sx = box_cx
        sy = by + box_h if dy > 0 else by
    _draw_arrow(draw, (sx, sy), (tx, ty), style)
    return img


def _fit_callout(
    W: int,
    H: int,
    bbox: tuple[int, int, int, int],
    title: str | None,
    instruction: str,
    example_label: str | None,
    example: str,
    draw: ImageDraw.ImageDraw,
) -> tuple[
    int, int, int, int,
    ImageFont.ImageFont, ImageFont.ImageFont,
    list[str], list[str], list[str], list[str],
]:
    """Find a (box_w, font_size) combo that places the callout cleanly.

    Tries progressively smaller widths/fonts until ``_layout_box`` returns a
    spot that doesn't overlap the bbox. Falls back to the smallest combo
    with the least overlap when nothing fits.

    Returns ``(bx, by, box_w, box_h, title_font, body_font, title_lines,
    body_lines, ex_label_lines, ex_lines)``.
    """
    x1, y1, x2, y2 = bbox
    # Generate size variants from largest to smallest.
    desired_w = max(280, min(520, max((x2 - x1) + 220, W // 3)))
    desired_w = min(desired_w, max(200, int(W * 0.65)))
    # Step down until we hit 180 px (very tight, last resort).
    width_steps = []
    w = desired_w
    while w >= 200:
        width_steps.append(w)
        w -= 40
    if width_steps[-1] != 200:
        width_steps.append(200)
    font_variants = [(28, 22), (24, 19), (20, 16), (18, 14)]

    show_example = bool(example_label) and bool(example) and example != "—"
    best: tuple[int, dict] | None = None  # (overlap_area_neg_w, vars)
    for box_w in width_steps:
        for tsz, bsz in font_variants:
            t_font = find_font(tsz)
            b_font = find_font(bsz)
            t_lines: list[str] = [title] if title else []
            b_lines = _wrap(instruction or "(нет описания)", b_font, box_w - 32, draw)
            el_lines: list[str] = [example_label] if show_example else []  # type: ignore[list-item]
            e_lines = _wrap(example, b_font, box_w - 32, draw) if show_example else []
            line_h_t = _font_size(t_font) + 6
            line_h_b = _font_size(b_font) + 4
            box_h = (
                16
                + line_h_t * len(t_lines)
                + (8 if t_lines else 0)
                + line_h_b * len(b_lines)
                + (12 if el_lines else 0)
                + line_h_t * len(el_lines)
                + (4 if el_lines else 0)
                + line_h_b * len(e_lines)
                + 16
            )
            # Skip if too tall for the image — won't fit anywhere.
            if box_h > H - 48:
                continue
            bx, by = _layout_box(W, H, bbox, box_w, box_h)
            ov = _overlap_area((bx, by, bx + box_w, by + box_h), bbox)
            packed = dict(
                bx=bx, by=by, box_w=box_w, box_h=box_h,
                t_font=t_font, b_font=b_font,
                t_lines=t_lines, b_lines=b_lines,
                el_lines=el_lines, e_lines=e_lines,
            )
            if ov == 0:
                return (
                    packed["bx"], packed["by"], packed["box_w"], packed["box_h"],
                    packed["t_font"], packed["b_font"],
                    packed["t_lines"], packed["b_lines"],
                    packed["el_lines"], packed["e_lines"],
                )
            # Track the smallest-overlap fallback (prefer wider/larger fonts
            # when overlaps are equal so we don't shrink unnecessarily).
            key = (ov, -box_w, -tsz)
            if best is None or key < best[0]:
                best = (key, packed)
    assert best is not None, "no size variant tried"
    p = best[1]
    return (
        p["bx"], p["by"], p["box_w"], p["box_h"],
        p["t_font"], p["b_font"],
        p["t_lines"], p["b_lines"],
        p["el_lines"], p["e_lines"],
    )


def annotate(
    base: Image.Image,
    bbox: tuple[int, int, int, int],
    instruction: str,
    example: str,
    style: Style | None = None,
    *,
    title: str | None = "Что написать:",
    example_label: str | None = "Пример:",
    no_box_fill: bool = False,
) -> Image.Image:
    """Draw a red rectangle around bbox + a callout box with text + arrow.

    ``title``/``example_label`` can be customized for non-fill modes
    (e.g. set ``title=\"Ответ:\"`` and ``example_label=None`` for a
    locate-style answer). Passing ``None`` to a label hides that section.

    When ``no_box_fill=True``, the callout box keeps its red border but
    omits the cream background fill — the text is rendered directly on
    the underlying image. Used for fallback candidates 4–5.
    """
    style = style or Style()
    img = base.convert("RGB").copy()
    W, H = img.size

    # clamp bbox into image
    x1 = max(0, min(W - 1, int(bbox[0])))
    y1 = max(0, min(H - 1, int(bbox[1])))
    x2 = max(x1 + 2, min(W - 1, int(bbox[2])))
    y2 = max(y1 + 2, min(H - 1, int(bbox[3])))
    bbox = (x1, y1, x2, y2)

    draw = ImageDraw.Draw(img, "RGBA")

    # 1) red rectangle
    pad = style.rect_pad
    for off in range(style.rect_layers):
        draw.rectangle(
            [x1 - pad - off, y1 - pad - off, x2 + pad + off, y2 + pad + off],
            outline=style.color,
            width=1,
        )

    # 2) callout box layout — auto-fit width/font so we don't overlap.
    (
        bx, by, box_w, box_h,
        title_font, body_font,
        title_lines, body_lines, ex_label_lines, ex_lines,
    ) = _fit_callout(W, H, bbox, title, instruction, example_label, example, draw)

    line_h_title = _font_size(title_font) + 6
    line_h_body = _font_size(body_font) + 4

    draw.rounded_rectangle(
        [bx, by, bx + box_w, by + box_h],
        radius=18,
        fill=None if no_box_fill else style.box_fill,
        outline=style.color,
        width=3,
    )

    cy = by + 14
    for line in title_lines:
        draw.text((bx + 16, cy), line, font=title_font, fill=style.box_title)
        cy += line_h_title
    if title_lines:
        cy += 6
    for line in body_lines:
        draw.text((bx + 16, cy), line, font=body_font, fill=style.box_text)
        cy += line_h_body
    if ex_label_lines:
        cy += 10
        for line in ex_label_lines:
            draw.text((bx + 16, cy), line, font=title_font, fill=style.box_title)
            cy += line_h_title
        for line in ex_lines:
            draw.text((bx + 16, cy), line, font=body_font, fill=style.box_text)
            cy += line_h_body

    # 3) arrow from box -> rect
    start, end = _arrow_endpoints(bx, by, box_w, box_h, bbox, pad)
    _draw_arrow(draw, start, end, style)

    return img
