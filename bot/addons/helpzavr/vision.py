"""Groq vision: looks at the screenshot, returns a structured description of likely fields.

We use the chat.completions endpoint with an image and ask the model to return JSON like:

    {
      "image_description": "...",
      "candidates": [
        {"id": "f1", "label": "Email field", "bbox": [x1, y1, x2, y2], "kind": "input"},
        ...
      ]
    }

The model returns coordinates in **percent of image** (0..100) — this is far more reliable
across different model sizes than asking for raw pixel values.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import re
from dataclasses import dataclass

import httpx
from PIL import Image

logger = logging.getLogger(__name__)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

VISION_PROMPT = """\
You are a UI analyst looking at a screenshot. Identify every region where a user \
could TYPE TEXT or click a primary action: text inputs, textareas, search boxes, \
message composers (the bottom bar of a chat where you write a reply), comment \
boxes, dialog fields, primary call-to-action buttons. Skip pure decoration, \
existing message bubbles, navigation, and labels.

Return STRICT JSON with this exact schema and NO extra text, NO markdown fences:
{
  "image_description": "<one short sentence describing what the screenshot shows>",
  "candidates": [
    {
      "id": "<short id like f1>",
      "label": "<EXACT verbatim text visible ON the element itself; for buttons this is the button text, for inputs this is the placeholder or label written next to the box; copy-paste it character-for-character in the ORIGINAL language of the screenshot — DO NOT translate, DO NOT paraphrase>",
      "kind": "<input|textarea|button|search|message_composer|other>",
      "bbox_pct": [x1, y1, x2, y2],
      "nearby_text": "<exact label/placeholder text adjacent to or inside this region, in original language, verbatim>",
      "context": "<short note in Russian: what this field seems to be for>"
    }
  ]
}

Coordinates `bbox_pct` are percentages of the image (0..100): x from the LEFT \
edge, y from the TOP edge. Each rectangle MUST tightly enclose the rectangular \
clickable/typeable area itself — the box outline you can actually click on — \
NOT the label text above it, NOT a wide row, NOT a surrounding panel.

Tips for accuracy:
- For chat composers: enclose the long horizontal text input bar at the bottom, \
  not the whole row of icons.
- For form inputs: enclose just the rectangle that holds the typed text, \
  excluding the visible label/title above it.
- Aspect ratio matters — input boxes are typically wider than tall.

If the screenshot has clear fields, list up to 6 of the most relevant. Always \
return at least one candidate if anything looks interactive. Respond with JSON only.
"""


@dataclass
class Candidate:
    id: str
    label: str
    kind: str
    bbox_pct: tuple[float, float, float, float]
    context: str
    nearby_text: str = ""

    def to_pixel_bbox(self, w: int, h: int) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = self.bbox_pct
        # The model can be inconsistent: sometimes percentages (0..100),
        # sometimes fractions (0..1), and *occasionally* a mix where X is
        # percent but Y is fractional. Detect each axis independently.
        if max(x1, x2) <= 1.5:
            x1, x2 = x1 * 100, x2 * 100
        if max(y1, y2) <= 1.5:
            y1, y2 = y1 * 100, y2 * 100
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        return (
            max(0, min(w - 1, int(round(x1 * w / 100)))),
            max(0, min(h - 1, int(round(y1 * h / 100)))),
            max(0, min(w - 1, int(round(x2 * w / 100)))),
            max(0, min(h - 1, int(round(y2 * h / 100)))),
        )


@dataclass
class VisionResult:
    image_description: str
    candidates: list[Candidate]


def _extract_json(text: str) -> dict:
    """Best-effort extraction of a JSON object from a model response."""
    text = text.strip()
    # remove markdown fences
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # find first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in response: {text[:200]!r}")
    return json.loads(match.group(0))


def _shrink_image_for_vision(image: Image.Image, max_dim: int = 1600) -> bytes:
    """Downscale very large images to keep request size small but preserve detail."""
    img = image.convert("RGB")
    w, h = img.size
    scale = min(1.0, max_dim / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


async def analyze_screenshot(
    image: Image.Image,
    user_question: str,
    api_key: str,
    model: str,
    timeout: float = 60.0,
) -> VisionResult:
    """Send the image to Groq and parse a structured list of candidate fields."""
    png_bytes = _shrink_image_for_vision(image)
    b64 = base64.b64encode(png_bytes).decode("ascii")
    data_url = f"data:image/png;base64,{b64}"

    body = {
        "model": model,
        "temperature": 0.1,
        "max_tokens": 1500,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VISION_PROMPT},
                    {
                        "type": "text",
                        "text": f"Вопрос пользователя: {user_question or '(не задан, опиши все потенциальные поля)'}",
                    },
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
    }

    import asyncio
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    max_retries = 2
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(max_retries + 1):
            resp = await client.post(GROQ_URL, headers=headers, json=body)
            if resp.status_code == 429 and attempt < max_retries:
                retry_after = float(resp.headers.get("retry-after") or 0)
                if not retry_after:
                    m = re.search(r"try again in ([\d.]+)s", resp.text)
                    if m:
                        retry_after = float(m.group(1))
                wait_s = max(retry_after + 0.5, 2.0)
                logger.warning("Groq vision 429 — waiting %.1fs (attempt %d/%d)", wait_s, attempt + 1, max_retries)
                await asyncio.sleep(wait_s)
                continue
            if resp.status_code >= 400:
                logger.error("Groq error %s: %s", resp.status_code, resp.text[:500])
                resp.raise_for_status()
            payload = resp.json()
            break
        else:
            raise RuntimeError("Groq retry loop exhausted")

    content = payload["choices"][0]["message"]["content"]
    data = _extract_json(content)

    cand: list[Candidate] = []
    for raw in data.get("candidates", []) or []:
        try:
            bb = raw.get("bbox_pct") or raw.get("bbox")
            if not bb or len(bb) != 4:
                continue
            cand.append(
                Candidate(
                    id=str(raw.get("id", f"f{len(cand)+1}")),
                    label=str(raw.get("label", "поле")),
                    kind=str(raw.get("kind", "other")),
                    bbox_pct=(float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])),
                    context=str(raw.get("context", "")),
                    nearby_text=str(raw.get("nearby_text", "")),
                )
            )
        except (TypeError, ValueError):
            logger.warning("Skipping malformed candidate: %r", raw)

    return VisionResult(
        image_description=str(data.get("image_description", "")),
        candidates=cand,
    )


GENERAL_PROMPT = """\
Ты внимательно смотришь на картинку и отвечаешь по-русски на вопрос пользователя \
о ней. Это может быть:
- скриншот сайта/приложения, где спрашивают где что-то находится («где X?»),
- фото объекта/места,
- головоломка или капча, которую нужно разобрать,
- что угодно ещё.

Если пользователь спрашивает «где X» — опиши словами где именно X находится на \
картинке (правый верхний угол, под кнопкой Y, рядом с заголовком Z, и т.п.). \
Если можно — добавь координаты в процентах от ширины/высоты картинки в виде \
JSON-блока в конце ответа.

Отвечай коротко, по делу, естественным языком. НЕ начинай словами «Конечно», \
«Хорошо», «Я вижу». Сразу к сути.

Если вопрос пустой — опиши кратко, что изображено на картинке и что с ней можно \
сделать.

В самом конце ответа, если ты явно можешь указать координаты пальцем (например, \
«Releases справа сверху»), добавь отдельной строкой:

```json
{"locate_pct": [x1, y1, x2, y2]}
```

где `x1,y1,x2,y2` — границы найденного элемента в процентах от ширины/высоты \
(0..100). ВАЖНО:
- bbox должен охватывать **весь объект целиком** (всю дверь, всю кнопку, весь \
блок Releases) — не «точку» в центре, а реальные границы. Минимум 5% ширины и \
5% высоты картинки.
- Координаты в порядке: левый край, верхний край, правый край, нижний край. \
Левый < правый, верхний < нижний.
- Если уверенно показать пальцем нельзя — НЕ добавляй этот блок.
"""


@dataclass
class GeneralAnswer:
    text: str
    locate_pct: tuple[float, float, float, float] | None = None


def _strip_locate_block(text: str) -> tuple[str, tuple[float, float, float, float] | None]:
    """Pull a trailing ```json {"locate_pct": [...]}``` block out of ``text``.

    Returns the cleaned text and the parsed bbox (or ``None`` if absent /
    malformed).
    """
    if not text:
        return "", None
    # Find the last fenced JSON block — that's where the prompt asks the model
    # to put the optional coordinates.
    m = list(re.finditer(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text))
    locate: tuple[float, float, float, float] | None = None
    if m:
        last = m[-1]
        try:
            data = json.loads(last.group(1))
            box = data.get("locate_pct") or data.get("bbox_pct") or data.get("bbox")
            if isinstance(box, list) and len(box) == 4:
                x1, y1, x2, y2 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
                # Reject zero-area or degenerate boxes — these mean the
                # model couldn't actually point.
                if abs(x2 - x1) > 0.5 and abs(y2 - y1) > 0.5:
                    locate = (x1, y1, x2, y2)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        if locate is not None:
            text = (text[:last.start()] + text[last.end():]).rstrip()
        else:
            # Strip the dud locate block from the visible text anyway.
            text = (text[:last.start()] + text[last.end():]).rstrip()
    return text.strip(), locate


async def answer_about_image(
    image: Image.Image,
    user_question: str,
    api_key: str,
    model: str,
    timeout: float = 60.0,
) -> GeneralAnswer:
    """General-purpose image Q&A (no candidates, just a free-text answer).

    Used as a fallback when :func:`analyze_screenshot` finds nothing
    actionable, or when the user's question is clearly analytical
    («где X», «опиши», «найди»).
    """
    png_bytes = _shrink_image_for_vision(image)
    b64 = base64.b64encode(png_bytes).decode("ascii")
    data_url = f"data:image/png;base64,{b64}"

    body = {
        "model": model,
        "temperature": 0.2,
        "max_tokens": 800,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": GENERAL_PROMPT},
                    {
                        "type": "text",
                        "text": f"Вопрос пользователя: {user_question or '(не задан, опиши кратко)'}",
                    },
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
    }

    import asyncio
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    max_retries = 2
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(max_retries + 1):
            resp = await client.post(GROQ_URL, headers=headers, json=body)
            if resp.status_code == 429 and attempt < max_retries:
                retry_after = float(resp.headers.get("retry-after") or 0)
                if not retry_after:
                    m = re.search(r"try again in ([\d.]+)s", resp.text)
                    if m:
                        retry_after = float(m.group(1))
                wait_s = max(retry_after + 0.5, 2.0)
                logger.warning("Groq general 429 — waiting %.1fs (attempt %d/%d)", wait_s, attempt + 1, max_retries)
                await asyncio.sleep(wait_s)
                continue
            if resp.status_code >= 400:
                logger.error("Groq general error %s: %s", resp.status_code, resp.text[:500])
                resp.raise_for_status()
            payload = resp.json()
            break
        else:
            raise RuntimeError("Groq general retry loop exhausted")

    content = payload["choices"][0]["message"].get("content") or ""
    text, locate = _strip_locate_block(content)
    return GeneralAnswer(text=text or "(пустой ответ модели)", locate_pct=locate)


VERIFY_LOCATE_PROMPT = """\
Я обвёл КРАСНОЙ рамкой место, которое должно соответствовать запросу \
пользователя. Посмотри на эту аннотированную картинку и реши:

- Если рамка стоит правильно (полностью охватывает нужный объект, без \
большого смещения) — ответь СТРОГО одним JSON-блоком:
```json
{"ok": true}
```

- Если рамка стоит неправильно (не на том объекте, смещена, слишком \
большая, слишком маленькая) — верни поправленные координаты в виде:
```json
{"ok": false, "locate_pct": [x1, y1, x2, y2]}
```
где `x1,y1,x2,y2` — проценты от ширины/высоты картинки (0..100), \
охватывающие нужный объект ЦЕЛИКОМ. Минимум 5% ширины и 5% высоты.

Никаких других слов — только JSON-блок.
"""


VERIFY_TIP_PROMPT = """\
На картинке нарисована КРАСНАЯ СТРЕЛКА. Её кончик (наконечник) должен \
указывать на объект из запроса пользователя и КАСАТЬСЯ его.

Оцени:
- Если кончик стрелки касается нужного объекта — ответь СТРОГО:
```json
{"ok": true}
```

- Если кончик стрелки промахнулся — верни **точку** куда надо сдвинуть \
кончик, в виде:
```json
{"ok": false, "tip_pct": [x, y]}
```
где `x, y` — проценты от ширины/высоты картинки (0..100), указывающие \
центр нужного объекта.

Никаких других слов — только JSON-блок.
"""


async def verify_tip(
    annotated_image: Image.Image,
    user_question: str,
    api_key: str,
    model: str,
    timeout: float = 60.0,
) -> tuple[bool, tuple[float, float] | None]:
    """Like :func:`verify_locate` but evaluates an arrow tip, not a rectangle.

    Returns ``(ok, tip_pct)``.
    """
    png_bytes = _shrink_image_for_vision(annotated_image)
    b64 = base64.b64encode(png_bytes).decode("ascii")
    data_url = f"data:image/png;base64,{b64}"
    body = {
        "model": model,
        "temperature": 0.1,
        "max_tokens": 200,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VERIFY_TIP_PROMPT},
                    {
                        "type": "text",
                        "text": f"Вопрос пользователя: {user_question or '(не задан)'}",
                    },
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
    }
    import asyncio
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    max_retries = 2
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(max_retries + 1):
            resp = await client.post(GROQ_URL, headers=headers, json=body)
            if resp.status_code == 429 and attempt < max_retries:
                retry_after = float(resp.headers.get("retry-after") or 0)
                if not retry_after:
                    m = re.search(r"try again in ([\d.]+)s", resp.text)
                    if m:
                        retry_after = float(m.group(1))
                wait_s = max(retry_after + 0.5, 2.0)
                await asyncio.sleep(wait_s)
                continue
            if resp.status_code >= 400:
                logger.error("Groq verify_tip error %s: %s", resp.status_code, resp.text[:500])
                resp.raise_for_status()
            payload = resp.json()
            break
        else:
            raise RuntimeError("Groq verify_tip retry loop exhausted")

    content = payload["choices"][0]["message"].get("content") or ""
    m = list(re.finditer(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", content))
    if not m:
        m = list(re.finditer(r"(\{[\s\S]*?\})", content))
    if not m:
        return False, None
    try:
        data = json.loads(m[-1].group(1))
    except json.JSONDecodeError:
        return False, None
    if bool(data.get("ok")):
        return True, None
    tip = data.get("tip_pct") or data.get("tip") or data.get("point")
    if isinstance(tip, list) and len(tip) == 2:
        try:
            x, y = float(tip[0]), float(tip[1])
            return False, (x, y)
        except (TypeError, ValueError):
            pass
    return False, None


PICK_BEST_PROMPT = """\
Тебе показано НЕСКОЛЬКО версий одной и той же картинки. На каждой я обвёл \
красной рамкой или указал красной стрелкой место, которое должно \
соответствовать запросу пользователя.

Сравни ВСЕ варианты и выбери ОДИН — который лучше всего показывает место \
из запроса. Учитывай:
- рамка/стрелка должны быть НА нужном объекте, а не рядом
- если выбор между рамкой и стрелкой — выбирай тот, что точнее
- если все плохие — выбери наименее плохой

Ответь СТРОГО одним JSON-блоком:
```json
{"best": N}
```
где N — номер варианта от 1 до количества картинок. Никаких других слов.
"""


async def pick_best_candidate(
    candidates: list[Image.Image],
    user_question: str,
    api_key: str,
    model: str,
    timeout: float = 90.0,
) -> int:
    """Show all candidates to Groq and ask which one best matches the query.

    Returns 0-based index of the picked candidate (defaults to 0 on failure).
    """
    if not candidates:
        return 0
    if len(candidates) == 1:
        return 0

    content: list[dict] = [
        {"type": "text", "text": PICK_BEST_PROMPT},
        {
            "type": "text",
            "text": f"Запрос пользователя: {user_question or '(не задан)'}",
        },
    ]
    for i, img in enumerate(candidates, 1):
        png_bytes = _shrink_image_for_vision(img, max_dim=900)
        b64 = base64.b64encode(png_bytes).decode("ascii")
        content.append({"type": "text", "text": f"Вариант {i}:"})
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
        )

    body = {
        "model": model,
        "temperature": 0.1,
        "max_tokens": 100,
        "messages": [{"role": "user", "content": content}],
    }
    import asyncio
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    max_retries = 2
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(max_retries + 1):
            resp = await client.post(GROQ_URL, headers=headers, json=body)
            if resp.status_code == 429 and attempt < max_retries:
                retry_after = float(resp.headers.get("retry-after") or 0)
                if not retry_after:
                    m = re.search(r"try again in ([\d.]+)s", resp.text)
                    if m:
                        retry_after = float(m.group(1))
                wait_s = max(retry_after + 0.5, 2.0)
                await asyncio.sleep(wait_s)
                continue
            if resp.status_code >= 400:
                logger.error("Groq pick_best error %s: %s", resp.status_code, resp.text[:500])
                resp.raise_for_status()
            payload = resp.json()
            break
        else:
            raise RuntimeError("Groq pick_best retry loop exhausted")

    content_text = payload["choices"][0]["message"].get("content") or ""
    matches = list(re.finditer(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", content_text))
    if not matches:
        matches = list(re.finditer(r"(\{[\s\S]*?\})", content_text))
    if not matches:
        logger.warning("pick_best: no JSON in response %r", content_text[:200])
        return 0
    try:
        data = json.loads(matches[-1].group(1))
    except json.JSONDecodeError:
        logger.warning("pick_best: bad JSON in response %r", content_text[:200])
        return 0
    n = data.get("best")
    if isinstance(n, int) and 1 <= n <= len(candidates):
        return n - 1
    if isinstance(n, str):
        try:
            i = int(n)
            if 1 <= i <= len(candidates):
                return i - 1
        except ValueError:
            pass
    logger.warning("pick_best: invalid best=%r, defaulting to 0", n)
    return 0


async def verify_locate(
    annotated_image: Image.Image,
    user_question: str,
    api_key: str,
    model: str,
    timeout: float = 60.0,
) -> tuple[bool, tuple[float, float, float, float] | None]:
    """Ask Groq to verify (or correct) a drawn red bbox on the image.

    Returns ``(ok, corrected_bbox_pct)``:
    - ``ok=True`` and ``corrected=None`` → the rectangle is fine.
    - ``ok=False`` and ``corrected=(...)`` → use the corrected coords.
    - ``ok=False`` and ``corrected=None`` → couldn't parse, keep original.
    """
    png_bytes = _shrink_image_for_vision(annotated_image)
    b64 = base64.b64encode(png_bytes).decode("ascii")
    data_url = f"data:image/png;base64,{b64}"
    body = {
        "model": model,
        "temperature": 0.1,
        "max_tokens": 200,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VERIFY_LOCATE_PROMPT},
                    {
                        "type": "text",
                        "text": f"Вопрос пользователя: {user_question or '(не задан)'}",
                    },
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
    }
    import asyncio
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    max_retries = 2
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(max_retries + 1):
            resp = await client.post(GROQ_URL, headers=headers, json=body)
            if resp.status_code == 429 and attempt < max_retries:
                retry_after = float(resp.headers.get("retry-after") or 0)
                if not retry_after:
                    m = re.search(r"try again in ([\d.]+)s", resp.text)
                    if m:
                        retry_after = float(m.group(1))
                wait_s = max(retry_after + 0.5, 2.0)
                logger.warning(
                    "Groq verify 429 — waiting %.1fs (attempt %d/%d)",
                    wait_s, attempt + 1, max_retries,
                )
                await asyncio.sleep(wait_s)
                continue
            if resp.status_code >= 400:
                logger.error("Groq verify error %s: %s", resp.status_code, resp.text[:500])
                resp.raise_for_status()
            payload = resp.json()
            break
        else:
            raise RuntimeError("Groq verify retry loop exhausted")

    content = payload["choices"][0]["message"].get("content") or ""
    # Same parser handles {"ok": ..., "locate_pct": [...]}
    m = list(re.finditer(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", content))
    if not m:
        m = list(re.finditer(r"(\{[\s\S]*?\})", content))
    if not m:
        logger.warning("verify_locate: no JSON in response %r", content[:200])
        return False, None
    try:
        data = json.loads(m[-1].group(1))
    except json.JSONDecodeError:
        logger.warning("verify_locate: bad JSON in response %r", content[:200])
        return False, None
    ok = bool(data.get("ok"))
    if ok:
        return True, None
    box = data.get("locate_pct") or data.get("bbox_pct") or data.get("bbox")
    if isinstance(box, list) and len(box) == 4:
        try:
            x1, y1, x2, y2 = (
                float(box[0]), float(box[1]), float(box[2]), float(box[3])
            )
            if abs(x2 - x1) > 0.5 and abs(y2 - y1) > 0.5:
                return False, (x1, y1, x2, y2)
        except (TypeError, ValueError):
            pass
    return False, None
