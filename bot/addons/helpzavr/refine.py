"""Refine an approximate bbox returned by the vision model.

Three strategies, in order of decreasing reliability:

- `refine_bbox_ocr`: when the vision model gives us the *text* that is on or
  next to the target, find that text in the image with Tesseract OCR and snap
  to its bounding rectangle. This is by far the most reliable when the target
  has visible text — OCR returns pixel-precise coordinates.

- `refine_bbox_cv`: snap to a nearby visible rectangular UI outline using
  OpenCV edge detection. Useful as a polish step or when there is no text.

- `refine_bbox_zoom`: re-ask the vision model with a zoomed-in crop. Kept as
  a last resort; in practice we found Llama 4 Scout's mis-placement bias
  carries through to zoomed views, so this is disabled by default.
"""
from __future__ import annotations

import base64
import contextlib
import difflib
import io
import json
import logging
import re

import cv2
import httpx
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageOps
from pytesseract import Output

logger = logging.getLogger(__name__)


# ---------- shared helpers ----------

def _pil_to_cv(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _bbox_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def _dist(a, b) -> float:
    ax, ay = _center(a)
    bx, by = _center(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


# ---------- OpenCV-based snap ----------

def _detect_rectangles(bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 30, 100)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    rects: list[tuple[int, int, int, int]] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < 16 or h < 10:
            continue
        rects.append((x, y, x + w, y + h))
    return rects


def refine_bbox_cv(
    image: Image.Image,
    approx: tuple[int, int, int, int],
    *,
    search_pad: float = 0.45,
    min_iou: float = 0.05,
    max_area_ratio: float = 2.5,
    min_area_ratio: float = 0.25,
) -> tuple[int, int, int, int]:
    W, H = image.size
    x1, y1, x2, y2 = approx
    if x1 >= x2 or y1 >= y2:
        return approx

    bw, bh = x2 - x1, y2 - y1
    pad_x = int(bw * search_pad + 24)
    pad_y = int(bh * search_pad + 24)
    sx1 = max(0, x1 - pad_x)
    sy1 = max(0, y1 - pad_y)
    sx2 = min(W, x2 + pad_x)
    sy2 = min(H, y2 + pad_y)

    cv = _pil_to_cv(image)
    region = cv[sy1:sy2, sx1:sx2]
    if region.size == 0:
        return approx

    local = _detect_rectangles(region)
    rects = [(rx1 + sx1, ry1 + sy1, rx2 + sx1, ry2 + sy1) for (rx1, ry1, rx2, ry2) in local]
    approx_area = bw * bh
    approx_aspect = bw / max(1, bh)

    best = None
    best_score = -1e9
    for r in rects:
        rw, rh = r[2] - r[0], r[3] - r[1]
        rarea = rw * rh
        if rarea > approx_area * max_area_ratio:
            continue
        if rarea < approx_area * min_area_ratio:
            continue
        iou = _bbox_iou(approx, r)
        if iou < min_iou:
            continue
        rect_aspect = rw / max(1, rh)
        aspect_ok = (
            min(approx_aspect, rect_aspect) / max(approx_aspect, rect_aspect)
        )
        center_pen = _dist(approx, r) / max(W, H)
        size_ratio = min(rarea, approx_area) / max(rarea, approx_area)
        score = iou * 1.5 + aspect_ok * 0.6 + size_ratio * 0.4 - center_pen * 0.6
        if score > best_score:
            best_score = score
            best = r

    if best is None:
        return approx
    logger.info("CV snap: %s -> %s (score=%.3f)", approx, best, best_score)
    return best


# ---------- Zoomed-in second-pass with the vision model ----------

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

ZOOM_PROMPT_TMPL = """\
This image is a CROP of a larger screenshot. Inside this crop, there is a target \
UI element described as: "{label}" (kind: {kind}). The user wants to know where \
EXACTLY to type or click. {nearby_hint}

Return STRICT JSON only, no markdown, no commentary:
{{
  "found": <true|false>,
  "bbox_pct": [x1, y1, x2, y2]
}}

Coordinates are PERCENT of THIS CROP (0..100). The rectangle MUST tightly enclose \
the rectangular interactive area (e.g. just the input box outline, not the label \
above it, not the icons next to it). If you cannot find the element, return \
"found": false.
"""


def _extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError(f"No JSON: {text[:200]!r}")
    return json.loads(m.group(0))


def _crop_with_pad(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    pad: float = 1.0,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Return (crop_img, crop_box_in_orig) where pad=1.0 means add 100% of bbox size on each side."""
    W, H = image.size
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    px = int(bw * pad + 40)
    py = int(bh * pad + 40)
    cx1 = max(0, x1 - px)
    cy1 = max(0, y1 - py)
    cx2 = min(W, x2 + px)
    cy2 = min(H, y2 + py)
    return image.crop((cx1, cy1, cx2, cy2)), (cx1, cy1, cx2, cy2)


async def refine_bbox_zoom(
    image: Image.Image,
    approx: tuple[int, int, int, int],
    *,
    label: str,
    kind: str,
    nearby_text: str,
    api_key: str,
    model: str,
    pad: float = 1.0,
    timeout: float = 45.0,
) -> tuple[int, int, int, int] | None:
    crop, crop_box = _crop_with_pad(image, approx, pad=pad)
    crop_w, crop_h = crop.size
    if crop_w < 40 or crop_h < 40:
        return None

    buf = io.BytesIO()
    crop.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    nearby_hint = f"Adjacent text: {nearby_text!r}." if nearby_text else ""

    body = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": 200,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": ZOOM_PROMPT_TMPL.format(
                        label=label or "the field",
                        kind=kind or "input",
                        nearby_hint=nearby_hint,
                    )},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }
        ],
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                GROQ_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
            )
            if resp.status_code >= 400:
                logger.warning("Zoom refine: HTTP %s: %s", resp.status_code, resp.text[:200])
                return None
            data = _extract_json(resp.json()["choices"][0]["message"]["content"])
    except Exception as e:
        logger.warning("Zoom refine failed: %s", e)
        return None

    if not data.get("found", False):
        logger.info("Zoom refine: model said not found")
        return None
    bb = data.get("bbox_pct") or data.get("bbox")
    if not bb or len(bb) != 4:
        return None
    try:
        coords = [float(x) for x in bb]
    except (TypeError, ValueError):
        return None
    if max(coords) <= 1.5:
        coords = [c * 100 for c in coords]
    cx1, cy1, cx2, cy2 = coords
    if cx1 > cx2:
        cx1, cx2 = cx2, cx1
    if cy1 > cy2:
        cy1, cy2 = cy2, cy1
    cx1 = max(0, min(100, cx1))
    cy1 = max(0, min(100, cy1))
    cx2 = max(0, min(100, cx2))
    cy2 = max(0, min(100, cy2))

    # to crop pixel coords
    px1 = int(cx1 * crop_w / 100)
    py1 = int(cy1 * crop_h / 100)
    px2 = int(cx2 * crop_w / 100)
    py2 = int(cy2 * crop_h / 100)
    # to original-image coords
    ox = crop_box[0]
    oy = crop_box[1]
    refined = (ox + px1, oy + py1, ox + px2, oy + py2)
    if refined[2] - refined[0] < 8 or refined[3] - refined[1] < 8:
        return None
    logger.info("Zoom refine: %s -> %s", approx, refined)
    return refined


# ---------- OCR-based snap (most reliable when target has visible text) ----------

def _ocr_lines(
    image: Image.Image,
    *,
    lang: str = "eng+rus",
    psm: int = 6,
) -> list[dict]:
    """Run OCR and group words into lines. Returns list of {"text", "bbox", "conf"}."""
    data = pytesseract.image_to_data(
        image, lang=lang, config=f"--psm {psm}", output_type=Output.DICT
    )
    n = len(data["text"])
    by_line: dict[tuple, list[int]] = {}
    for i in range(n):
        if not data["text"][i].strip():
            continue
        try:
            conf = int(float(data["conf"][i]))
        except (TypeError, ValueError):
            conf = -1
        if conf < 30:
            continue
        key = (data["page_num"][i], data["block_num"][i], data["par_num"][i], data["line_num"][i])
        by_line.setdefault(key, []).append(i)

    lines = []
    for _key, idxs in by_line.items():
        idxs.sort(key=lambda i: data["left"][i])
        words = [data["text"][i] for i in idxs]
        lefts = [data["left"][i] for i in idxs]
        tops = [data["top"][i] for i in idxs]
        rights = [data["left"][i] + data["width"][i] for i in idxs]
        bottoms = [data["top"][i] + data["height"][i] for i in idxs]
        confs = [int(float(data["conf"][i])) for i in idxs]
        bbox = (min(lefts), min(tops), max(rights), max(bottoms))
        lines.append({
            "text": " ".join(words),
            "bbox": bbox,
            "conf": int(np.mean(confs)) if confs else -1,
            "indices": idxs,
            "raw": data,
        })
    return lines


def _normalize(s: str) -> str:
    return re.sub(r"[^\w\s]+", " ", s.lower()).strip()


def _fuzzy_score(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    a, b = _normalize(a), _normalize(b)
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return 0.95
    return difflib.SequenceMatcher(None, a, b).ratio()


def _ocr_lines_cached(image: Image.Image, *, psm: int = 6) -> list[dict]:
    """OCR pass cached per (image, psm). PSM 6 = uniform block; PSM 11 = sparse."""
    cache = getattr(image, "_ocr_lines_cache", None)
    if not isinstance(cache, dict):
        cache = {}
    if psm in cache:
        return cache[psm]
    try:
        lines = _ocr_lines(image, lang="eng+rus", psm=psm)
    except pytesseract.TesseractError:
        lines = []
    cache[psm] = lines
    with contextlib.suppress(AttributeError, TypeError):
        image._ocr_lines_cache = cache  # type: ignore[attr-defined]
    return lines


def _fuzzy_score_strict(matched: str, target: str) -> float:
    """Like _fuzzy_score but gives much lower scores to very-partial matches."""
    if not matched or not target:
        return 0.0
    a, b = _normalize(matched), _normalize(target)
    if not a or not b:
        return 0.0

    a_words = a.split()
    b_words = b.split()

    # Exact / substring match — but only if matched is at least half the target.
    if a == b:
        return 1.0
    if a in b or b in a:
        # Penalise short fragments: "и" inside "регистрация" should NOT score high.
        ratio = min(len(a), len(b)) / max(len(a), len(b))
        # Multi-word target with a single matching word still gets a decent score
        if len(a_words) >= 1 and len(b_words) >= 1:
            word_ratio = min(len(a_words), len(b_words)) / max(len(a_words), len(b_words))
        else:
            word_ratio = ratio
        return 0.4 + 0.55 * (0.5 * ratio + 0.5 * word_ratio)

    # Word-overlap fuzzy
    common_words = set(a_words) & set(b_words)
    if common_words:
        word_ratio = len(common_words) / max(len(a_words), len(b_words))
        seq = difflib.SequenceMatcher(None, a, b).ratio()
        return 0.5 * word_ratio + 0.5 * seq
    return difflib.SequenceMatcher(None, a, b).ratio()


def _preprocess_upscale(
    image: Image.Image,
    scale: float = 2.5,
    contrast: float = 1.8,
) -> Image.Image:
    """Enlarge + sharpen + contrast-boost for OCR-friendliness.

    Useful for small, low-contrast text (e.g. light-grey placeholder inside
    a white input field) that Tesseract misses at the native resolution.
    """
    nw, nh = int(image.size[0] * scale), int(image.size[1] * scale)
    up = image.resize((nw, nh), Image.LANCZOS)
    if contrast and contrast != 1.0:
        up = ImageEnhance.Contrast(up).enhance(contrast)
    return up


def _ocr_find_text(
    image: Image.Image,
    target_text: str,
    *,
    min_score: float = 0.55,
    approx: tuple[int, int, int, int] | None = None,
) -> list[tuple[float, tuple[int, int, int, int], str]]:
    """Search OCR output for matches to `target_text`. Returns [(score, bbox, matched_text)]
    where bbox is in the ORIGINAL image's pixel coordinates."""
    matches: list[tuple[float, tuple[int, int, int, int], str]] = []
    target_norm = _normalize(target_text)
    if not target_norm:
        return matches
    target_words = target_norm.split()

    def _scan(prep_name: str, prep: Image.Image, psm: int = 6,
              scale: float = 1.0,
              offset: tuple[int, int] = (0, 0)) -> int:
        """Run an OCR pass on `prep`. If `prep` is a transformed copy of the
        original image (upscaled or cropped), pass `scale` and `offset` so the
        returned bboxes can be mapped back to the original image space.
        """
        ox, oy = offset
        inv_scale = 1.0 / scale

        def _map(x: int, y: int) -> tuple[int, int]:
            return (int(round(x * inv_scale + ox)), int(round(y * inv_scale + oy)))

        added = 0
        lines = _ocr_lines_cached(prep, psm=psm)
        for line in lines:
            line_text = line["text"]
            score = _fuzzy_score_strict(line_text, target_text)
            if score >= min_score:
                lx1, ly1, lx2, ly2 = line["bbox"]
                mx1, my1 = _map(lx1, ly1)
                mx2, my2 = _map(lx2, ly2)
                matches.append((score, (mx1, my1, mx2, my2), line_text))
                added += 1
            # Sliding windows of length up to len(target_words)+1
            idxs = line["indices"]
            raw = line["raw"]
            words = [raw["text"][i] for i in idxs]
            wmax = min(len(words), len(target_words) + 1)
            for w_size in range(1, wmax + 1):
                for start in range(len(words) - w_size + 1):
                    sel = idxs[start:start + w_size]
                    sub_text = " ".join(words[start:start + w_size])
                    if len(_normalize(sub_text)) < max(3, len(target_norm) // 4):
                        # Reject very short fragments (e.g. matching one letter
                        # of a many-character target).
                        continue
                    sub_score = _fuzzy_score_strict(sub_text, target_text)
                    if sub_score < min_score:
                        continue
                    x1 = min(raw["left"][i] for i in sel)
                    y1 = min(raw["top"][i] for i in sel)
                    x2 = max(raw["left"][i] + raw["width"][i] for i in sel)
                    y2 = max(raw["top"][i] + raw["height"][i] for i in sel)
                    mx1, my1 = _map(x1, y1)
                    mx2, my2 = _map(x2, y2)
                    matches.append((sub_score, (mx1, my1, mx2, my2), sub_text))
                    added += 1
        return added

    def _high_conf() -> bool:
        return any(score >= 0.92 for score, _, _ in matches)

    # 1. Normal image, PSM 6 (most cases)
    _scan("normal-6", image, psm=6)
    if not _high_conf():
        # 2. Inverted image, PSM 6 (white-on-dark text, e.g. CTA buttons)
        try:
            inv = ImageOps.invert(image.convert("RGB"))
        except Exception:
            inv = None
        if inv is not None:
            _scan("inverted-6", inv, psm=6)
        # 3. Inverted PSM 11 (sparse layout — buttons spread across page)
        if inv is not None and not _high_conf():
            _scan("inverted-11", inv, psm=11)
        # 4. Normal PSM 11 (sparse layout, dark-on-light)
        if not _high_conf():
            _scan("normal-11", image, psm=11)
        # 5. ZOOM-CROP: cut the area where Groq said the element is, upscale
        # 2.5×, boost contrast, then OCR there. Catches small / light-grey
        # placeholder text that fails on the full image.
        if approx is not None and not _high_conf():
            ax1, ay1, ax2, ay2 = approx
            pad_x = int((ax2 - ax1) * 1.2 + 32)
            pad_y = int((ay2 - ay1) * 1.6 + 32)
            cx1 = max(0, ax1 - pad_x)
            cy1 = max(0, ay1 - pad_y)
            cx2 = min(image.size[0], ax2 + pad_x)
            cy2 = min(image.size[1], ay2 + pad_y)
            if cx2 > cx1 + 20 and cy2 > cy1 + 20:
                try:
                    crop = image.crop((cx1, cy1, cx2, cy2))
                    up = _preprocess_upscale(crop, scale=2.5, contrast=1.8)
                    _scan("zoom-up", up, psm=6, scale=2.5, offset=(cx1, cy1))
                    if not _high_conf():
                        _scan("zoom-up-11", up, psm=11, scale=2.5, offset=(cx1, cy1))
                    if not _high_conf():
                        inv_up = ImageOps.invert(up.convert("RGB"))
                        _scan("zoom-up-inv", inv_up, psm=6, scale=2.5, offset=(cx1, cy1))
                except Exception as exc:  # pragma: no cover — defensive
                    logger.debug("OCR zoom-crop failed: %s", exc)
        # 6. UPSCALED FULL IMAGE: last-resort whole-image 2× upscale for very
        # small text. Skipped on large images (>1.5MP) to keep latency reasonable.
        if not _high_conf() and image.size[0] * image.size[1] < 1_500_000:
            try:
                up_full = _preprocess_upscale(image, scale=2.0, contrast=1.5)
                _scan("upscale-full-6", up_full, psm=6, scale=2.0)
                if not _high_conf():
                    _scan("upscale-full-11", up_full, psm=11, scale=2.0)
            except Exception as exc:  # pragma: no cover
                logger.debug("OCR full upscale failed: %s", exc)

    # Dedup and sort
    seen: set[tuple] = set()
    deduped: list[tuple[float, tuple[int, int, int, int], str]] = []
    for score, bbox, text in sorted(matches, key=lambda m: -m[0]):
        key = (round(bbox[0] / 4), round(bbox[1] / 4), round(bbox[2] / 4), round(bbox[3] / 4))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((score, bbox, text))
    return deduped


def _border_color(arr: np.ndarray, bbox: tuple[int, int, int, int], margin: int = 5) -> tuple[int, int, int]:
    """Median color of a thin border around bbox (excluding the text itself)."""
    H, W = arr.shape[:2]
    x1, y1, x2, y2 = bbox
    samples = []
    # Top strip just above the text
    if y1 > 0:
        top = arr[max(0, y1 - margin * 2):y1, max(0, x1 - margin):min(W, x2 + margin)]
        if top.size:
            samples.append(top.reshape(-1, 3))
    # Bottom strip just below
    if y2 < H:
        bot = arr[y2:min(H, y2 + margin * 2), max(0, x1 - margin):min(W, x2 + margin)]
        if bot.size:
            samples.append(bot.reshape(-1, 3))
    # Left strip
    if x1 > 0:
        left = arr[y1:y2, max(0, x1 - margin * 2):x1]
        if left.size:
            samples.append(left.reshape(-1, 3))
    # Right strip
    if x2 < W:
        right = arr[y1:y2, x2:min(W, x2 + margin * 2)]
        if right.size:
            samples.append(right.reshape(-1, 3))
    if not samples:
        return (255, 255, 255)
    s = np.concatenate(samples, axis=0)
    med = np.median(s, axis=0)
    return (int(med[0]), int(med[1]), int(med[2]))


def _button_likeness(bg_color: tuple[int, int, int]) -> float:
    """Score how 'button-like' a background colour is. 0..1."""
    r, g, b = bg_color
    mn, mx = min(r, g, b), max(r, g, b)
    saturation = (mx - mn)
    # Almost-white page background → very low
    if mn > 235:
        return 0.0
    score = 0.0
    # Darker than page → likely a button or a panel
    if mx < 100:
        score = max(score, 0.85)  # dark button (black, navy, …)
    elif mx < 180:
        score = max(score, 0.55)  # mid-tone button
    # Saturated colour → CTA button
    if saturation > 70:
        score = max(score, 0.9)
    elif saturation > 40:
        score = max(score, 0.65)
    return score


def refine_bbox_ocr(
    image: Image.Image,
    approx: tuple[int, int, int, int],
    target_text: str,
    *,
    pad_x_ratio: float = 0.45,
    pad_y_ratio: float = 0.7,
    kind: str = "",
) -> tuple[int, int, int, int] | None:
    """Find target_text in the image via OCR; snap bbox to that text plus padding.

    Disambiguates between multiple text occurrences using:
    - exact-vs-partial fuzzy match score
    - distance to the approximate bbox (Y weighted heavier than X, since X is
      typically the noisiest axis on Llama 4 Scout)
    - button-likeness of the background colour (dark/saturated vs page-white)
    """
    if not target_text or not target_text.strip():
        return None

    W, H = image.size
    matches = _ocr_find_text(image, target_text, approx=approx)
    if not matches:
        logger.info("OCR refine: no matches for %r", target_text)
        return None

    arr = np.array(image.convert("RGB"))

    ax, ay = (approx[0] + approx[2]) / 2, (approx[1] + approx[3]) / 2
    aw = max(1, approx[2] - approx[0])
    ah = max(1, approx[3] - approx[1])

    is_button = "button" in (kind or "").lower() or "link" in (kind or "").lower()
    is_input = any(k in (kind or "").lower() for k in ("input", "search", "composer", "textarea", "field"))

    best = None
    best_composite = -1e9
    best_debug = None
    for score, bbox, text in matches:
        bw = bbox[2] - bbox[0]
        bh = bbox[3] - bbox[1]
        if bw < 8 or bh < 6:
            continue
        # Hard reject: matched text spans much more area than the approx bbox.
        # Catches the case where `nearby_text` was a long instruction running
        # across the page, not the actual button label.
        if bw > aw * 4.0 and bh < ah * 0.6:
            continue
        if bh > ah * 4.0 and bw < aw * 0.6:
            continue
        if (bw * bh) > (aw * ah) * 6.0:
            continue
        cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        # Normalise distance by approx size, not image — far matches should be
        # heavily penalised even on huge images.
        dx_px = abs(cx - ax)
        dy_px = abs(cy - ay)
        dx = dx_px / max(W, 200)
        dy = dy_px / max(H, 200)
        # Anisotropic: Y is more reliable than X on Llama 4 Scout.
        d_pen = dx * 0.9 + dy * 2.4
        # Strong extra penalty for matches far from the approx in pixel terms.
        if dy_px > ah * 2.5:
            d_pen += 1.5
        if dx_px > aw * 2.5:
            d_pen += 1.0

        # Size sanity penalty: text shouldn't be much larger than the approximation
        size_pen = 0.0
        if bw > aw * 2.0:
            size_pen += 0.5
        if bh > ah * 2.0:
            size_pen += 0.5

        bg = _border_color(arr, bbox, margin=5)
        button_score = _button_likeness(bg)
        # Buttons benefit from coloured/dark surroundings; inputs benefit from
        # near-white (page) surroundings. We *only* hand out the bonus when the
        # text match is high-confidence — otherwise a wrong-but-button-looking
        # match (e.g. a coloured banner) outscores the correct partial match.
        kind_bonus = 0.0
        confidence = max(0.0, min(1.0, (score - 0.7) / 0.25))
        if is_button:
            kind_bonus = button_score * 0.6 * confidence
        elif is_input:
            kind_bonus = (1.0 - button_score) * 0.3 * confidence

        # Word-count sanity: prefer text whose word count is similar to target.
        target_words = max(1, len(_normalize(target_text).split()))
        text_words = max(1, len(_normalize(text).split()))
        word_ratio = min(target_words, text_words) / max(target_words, text_words)
        word_bonus = 0.15 * word_ratio

        composite = score * 1.2 + kind_bonus + word_bonus - d_pen - size_pen
        if composite > best_composite:
            best_composite = composite
            best = (score, bbox, text)
            best_debug = dict(
                score=score, dx=dx, dy=dy, d_pen=d_pen, size_pen=size_pen,
                bg=bg, button_score=button_score, kind_bonus=kind_bonus,
                word_bonus=word_bonus, composite=composite,
            )

    if best is None or best_composite < 0.2:
        logger.info(
            "OCR refine: no usable match (top=%s, target=%r, best_comp=%s)",
            matches[:3], target_text, best_composite,
        )
        return None

    score, bbox, text = best
    bx1, by1, bx2, by2 = bbox

    # If we found a button-like surrounding, expand the bbox to enclose the
    # full button background (the contiguous coloured/dark region) instead of
    # just the text.
    expanded = _expand_to_homogeneous_region(arr, bbox) if best_debug["button_score"] > 0.3 else None

    is_input = any(k in (kind or "").lower() for k in (
        "input", "search", "composer", "textarea", "field",
    ))

    if expanded is not None:
        refined = expanded
        logger.info("OCR refine: expanded text bbox %s -> button region %s", bbox, refined)
    elif is_input:
        # For input fields, only trust OCR text if BOTH:
        #   (a) the match is high-confidence (likely the placeholder we asked for)
        #   (b) the text Y falls within (or close to) Groq's approx Y range
        # Otherwise the OCR may have latched onto a label *outside* the field
        # (e.g. "Имя" written above the field at y=72 when the field is at y=92).
        ay1, ay2 = approx[1], approx[3]
        ax1, ax2 = approx[0], approx[2]
        text_cy = (by1 + by2) / 2
        approx_h = max(1, ay2 - ay1)
        # Allow Y to be 70% of approx height outside approx Y bounds
        y_tol = approx_h * 0.7
        within_y = (ay1 - y_tol) <= text_cy <= (ay2 + y_tol)
        high_conf = score >= 0.78
        if within_y and high_conf:
            bh = by2 - by1
            half = max(int(bh * 0.85), 12)
            nx1 = min(ax1, bx1 - 4)
            nx2 = max(ax2, bx2 + 4)
            ny1 = max(0, int(text_cy - half))
            ny2 = min(H - 1, int(text_cy + half))
            refined = (max(0, nx1), ny1, min(W - 1, nx2), ny2)
            logger.info(
                "OCR refine (input): text_y=%s approx_y=%s score=%.2f -> Y from text, X from approx, refined=%s",
                (by1, by2), (ay1, ay2), score, refined,
            )
        else:
            logger.info(
                "OCR refine (input): rejected text_y=%s approx_y=%s score=%.2f within_y=%s -> falling back to None",
                (by1, by2), (ay1, ay2), score, within_y,
            )
            return None  # Let caller fall back to CV polish on Groq's approx
    else:
        bw = bx2 - bx1
        bh = by2 - by1
        # The text we matched may be a fragment — pad generously by target word count
        target_words = max(1, len(_normalize(target_text).split()))
        text_words = max(1, len(_normalize(text).split()))
        # Each missing word ≈ another text width to the left
        word_factor = max(1.0, target_words / text_words)

        # For plain text / links, the clickable area is just the text itself.
        # We don't need to add half a width of padding — that produces a bbox
        # much wider than the link. Use small fixed padding instead.
        is_plain_text = (
            "link" in (kind or "").lower() or "other" in (kind or "").lower()
        )
        if is_plain_text:
            px = max(2, int(bh * 0.3))
            py = max(2, int(bh * 0.25))
        else:
            px = int(bw * (pad_x_ratio + 0.3 * (word_factor - 1)) + 8)
            py = int(bh * pad_y_ratio + 8)
        refined = (
            max(0, bx1 - px),
            max(0, by1 - py),
            min(W - 1, bx2 + px),
            min(H - 1, by2 + py),
        )
    logger.info(
        "OCR refine: matched %r -> text_bbox=%s refined=%s debug=%s",
        text, bbox, refined, best_debug,
    )
    return refined


def _expand_to_homogeneous_region(
    arr: np.ndarray,
    text_bbox: tuple[int, int, int, int],
    max_pad: int = 240,
    color_tol: int = 28,
    aa_skip: int = 2,
) -> tuple[int, int, int, int] | None:
    """Given a text bbox, find the contiguous coloured/dark region around it
    and return its bounding box. Useful for expanding from "Verify email" text
    to the full black button rectangle that contains it.

    `aa_skip` controls how many "non-matching" pixels of anti-aliased text
    fringe we will tolerate before giving up on expansion in a given direction.
    Bumping it from 0 → 3-4 lets us "skip past" the AA halo around glyphs and
    keep expanding through the real button background.
    """
    H, W = arr.shape[:2]
    x1, y1, x2, y2 = text_bbox
    # Sample background colour at a thin border just above & below the text
    # (avoids the text glyphs themselves which are usually a contrasting colour).
    border_samples = []
    if y1 - 2 > 0:
        border_samples.append(arr[max(0, y1 - 4):y1, x1:x2].reshape(-1, 3))
    if y2 + 2 < H:
        border_samples.append(arr[y2:min(H, y2 + 4), x1:x2].reshape(-1, 3))
    if not border_samples:
        return None
    s = np.concatenate(border_samples, axis=0)
    if s.size == 0:
        return None
    bg = np.median(s, axis=0).astype(np.int16)

    # Search outward from the text bbox, expanding bounds while the border
    # pixels remain close to `bg` colour.
    def _close(strip: np.ndarray) -> bool:
        if strip.size == 0:
            return False
        dist = np.linalg.norm(strip.reshape(-1, 3).astype(np.int16) - bg, axis=1)
        return float(np.median(dist)) < color_tol

    nx1, ny1, nx2, ny2 = x1, y1, x2, y2

    def _expand(direction: str) -> int:
        """Expand outward in `direction` ("left"|"right"|"top"|"bottom") and
        return the last confirmed-background coordinate.

        Algorithm: advance past up to `aa_skip` consecutive non-bg strips
        (handles anti-aliased text fringe). If a later strip matches bg,
        commit the position and reset the miss counter. If we hit
        `aa_skip + 1` misses in a row, ROLL BACK to the last confirmed-bg
        position (so we don't overshoot past the real edge into a
        different region).
        """
        if direction == "left":
            pos = nx1
            limit = 0
            step_dir = -1
        elif direction == "right":
            pos = nx2
            limit = W
            step_dir = 1
        elif direction == "top":
            pos = ny1
            limit = 0
            step_dir = -1
        elif direction == "bottom":
            pos = ny2
            limit = H
            step_dir = 1
        else:  # pragma: no cover
            raise ValueError(direction)

        def _strip(p: int):
            if direction == "left":
                col_x = p - 1
                if col_x < 0:
                    return None
                return arr[ny1:ny2, col_x:col_x + 1]
            if direction == "right":
                if p >= W:
                    return None
                return arr[ny1:ny2, p:p + 1]
            if direction == "top":
                row_y = p - 1
                if row_y < 0:
                    return None
                return arr[row_y:row_y + 1, nx1:nx2]
            # bottom
            if p >= H:
                return None
            return arr[p:p + 1, nx1:nx2]

        confirmed = pos
        misses = 0
        for _ in range(max_pad):
            next_pos = pos + step_dir
            if step_dir > 0 and next_pos > limit:
                break
            if step_dir < 0 and next_pos < limit:
                break
            strip = _strip(next_pos if direction in ("right", "bottom") else pos)
            if strip is None:
                break
            if _close(strip):
                pos = next_pos
                confirmed = pos
                misses = 0
            else:
                misses += 1
                if misses > aa_skip:
                    break
                pos = next_pos
        # Final position is the last position that was either bg or part of
        # the bounded look-ahead. If the last few steps were misses, we want
        # to ROLL BACK to the last confirmed-bg position.
        return confirmed

    nx1 = _expand("left")
    nx2 = _expand("right")
    ny1 = _expand("top")
    ny2 = _expand("bottom")

    if (nx2 - nx1) < (x2 - x1) + 4 and (ny2 - ny1) < (y2 - y1) + 4:
        return None  # didn't expand meaningfully
    return (nx1, ny1, nx2, ny2)


# ---------- Top-level helper ----------

def refine_bbox(image: Image.Image, approx: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Best-effort sync refinement using only OpenCV. Use this when async is awkward."""
    return refine_bbox_cv(image, approx)


async def refine_bbox_async(
    image: Image.Image,
    approx: tuple[int, int, int, int],
    *,
    label: str = "",
    kind: str = "",
    nearby_text: str = "",
    api_key: str = "",
    model: str = "",
    use_zoom: bool = False,
) -> tuple[int, int, int, int]:
    """Best-of-three refinement, preferring OCR when text is available.

    Order:
        1. OCR (most precise when target has visible text)
        2. Zoom-crop vision pass (off by default — Llama 4 Scout is biased)
        3. CV polish
    """
    # Decide which anchors to try, in what order, based on element kind.
    # For buttons / clickable elements, the LABEL is usually the on-button text
    # (most reliable). For inputs / fields, the LABEL is usually a placeholder
    # or external label, while `nearby_text` is whatever Groq saw next to it
    # (often a long instruction sentence — useless for snapping).
    is_input_like = any(k in (kind or "").lower() for k in (
        "input", "search", "composer", "textarea", "field",
    ))

    def _looks_like_real_text(s: str) -> bool:
        """Reject single special chars (→, •, ✓) and very short fragments."""
        if not s:
            return False
        clean = s.strip()
        # At least 3 alphanumeric (or letter) characters somewhere.
        letters = sum(1 for ch in clean if ch.isalnum())
        return letters >= 3

    anchors: list[str] = []
    if is_input_like:
        # For input fields the OCR-matched text is the placeholder inside
        # the box. We use it ONLY to pin the Y-axis (Groq's Y is unreliable
        # ±5%) — the X-extent is taken from Groq's approx (which is usually
        # wider and more correct on inputs). See refine_bbox_ocr's input
        # branch.
        if _looks_like_real_text(label):
            anchors.append(label)
    else:
        # Buttons / links: use ONLY the LABEL (the on-button text).
        # We deliberately do NOT fall back to `nearby_text` — that often
        # contains a section heading or instructional text some distance
        # away from the actual button (e.g. "Опасная зона" above a delete
        # button) and OCR would snap to the wrong place.
        if _looks_like_real_text(label):
            anchors.append(label)

    for anchor in anchors:
        ocr_box = refine_bbox_ocr(image, approx, anchor, kind=kind)
        if ocr_box is not None:
            return ocr_box

    refined = approx
    if use_zoom and api_key and model:
        zoomed = await refine_bbox_zoom(
            image,
            approx,
            label=label,
            kind=kind,
            nearby_text=nearby_text,
            api_key=api_key,
            model=model,
        )
        if zoomed is not None:
            refined = zoomed
    polished = refine_bbox_cv(image, refined)
    return polished
