"""OpenRouter "thinking" model: given a description of fields and a user question,
chooses one target field and decides what to write, with a concrete example.

Returns:
    {
      "chosen_id": "f3",
      "instruction": "Введите ваш рабочий email, ...",
      "example": "ivan.petrov@company.com",
      "rationale": "..."
    }
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

import httpx

from .vision import VisionResult

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

THINKING_SYSTEM = """\
Ты помощник, который смотрит на список полей формы (с описанием) и решает, что \
именно написать в нужное поле, исходя из вопроса пользователя.

Тебе придёт:
- общий контекст изображения,
- список кандидатов-полей (id, label, kind, context),
- вопрос пользователя.

Тебе нужно:
1. Выбрать ОДИН id, который лучше всего отвечает на вопрос пользователя.
2. Сформулировать короткую инструкцию (1–2 предложения) ЧТО туда вписать.
3. Дать конкретный ПРИМЕР значения, которое можно скопировать.

ВАЖНО: ответ — строго один JSON-объект, без преамбулы, без markdown, без размышлений снаружи JSON.
Схема:
{
  "chosen_id": "<id из списка кандидатов>",
  "instruction": "<коротко по-русски, что туда написать>",
  "example": "<конкретный пример значения>",
  "rationale": "<очень коротко: почему именно это поле>"
}
"""


@dataclass
class ThinkingResult:
    chosen_id: str
    instruction: str
    example: str
    rationale: str


def _extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Some reasoning models put thinking before the JSON; grab the last {...} block
    matches = list(re.finditer(r"\{[\s\S]*\}", text))
    if not matches:
        raise ValueError(f"No JSON found in thinking response: {text[:300]!r}")
    return json.loads(matches[-1].group(0))


def _format_candidates(vision: VisionResult) -> str:
    lines = [f"Описание скриншота: {vision.image_description}"]
    lines.append("Кандидаты-поля:")
    for c in vision.candidates:
        lines.append(
            f"- id={c.id}; label={c.label}; kind={c.kind}; "
            f"bbox_pct={tuple(round(x, 1) for x in c.bbox_pct)}; "
            f"context={c.context}"
        )
    return "\n".join(lines)


GROQ_THINKING_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_THINKING_MODEL = "llama-3.3-70b-versatile"


async def _call_chat(
    url: str,
    api_key: str,
    model: str,
    user_msg: str,
    timeout: float,
    extra_headers: dict[str, str] | None = None,
    max_retries: int = 2,
) -> str:
    import asyncio
    body = {
        "model": model,
        "temperature": 0.2,
        "max_tokens": 4096,
        "messages": [
            {"role": "system", "content": THINKING_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(max_retries + 1):
            resp = await client.post(url, headers=headers, json=body)
            if resp.status_code == 429 and attempt < max_retries:
                # Honour Retry-After if present, else parse Groq's "try again in Ns"
                retry_after = float(resp.headers.get("retry-after") or 0)
                if not retry_after:
                    import re as _re
                    m = _re.search(r"try again in ([\d.]+)s", resp.text)
                    if m:
                        retry_after = float(m.group(1))
                wait_s = max(retry_after + 0.5, 2.0)
                logger.warning("429 from %s — waiting %.1fs (attempt %d/%d)", url, wait_s, attempt + 1, max_retries)
                await asyncio.sleep(wait_s)
                continue
            if resp.status_code >= 400:
                logger.error("Thinking %s error %s: %s", url, resp.status_code, resp.text[:500])
                resp.raise_for_status()
            payload = resp.json()
            msg = payload["choices"][0]["message"]
            content = msg.get("content") or ""
            if not content.strip():
                content = msg.get("reasoning") or ""
            return content
    raise RuntimeError("Unreachable")


async def decide_what_to_write(
    vision: VisionResult,
    user_question: str,
    api_key: str,
    model: str,
    timeout: float = 90.0,
    *,
    fallback_groq_key: str | None = None,
) -> ThinkingResult:
    """Pick a target field and decide what to write.

    Tries the configured OpenRouter model first. If it fails (e.g. free-tier
    rate limit 429) and `fallback_groq_key` is provided, retries via Groq's
    Llama 3.3 70B which has a much higher daily quota.
    """
    if not vision.candidates:
        raise ValueError("No candidate fields to choose from.")

    user_msg = (
        f"{_format_candidates(vision)}\n\n"
        f"Вопрос пользователя: {user_question or '(не задан — выбери самое важное пустое поле)'}"
    )

    content = ""
    last_error: Exception | None = None
    try:
        content = await _call_chat(
            OPENROUTER_URL, api_key, model, user_msg, timeout,
            extra_headers={"X-Title": "tg-screenshot-helper-bot"},
        )
    except httpx.HTTPStatusError as e:
        last_error = e
        if not fallback_groq_key:
            raise
        logger.warning(
            "OpenRouter failed (%s) — falling back to Groq Llama 3.3 for thinking",
            e.response.status_code,
        )
        content = await _call_chat(
            GROQ_THINKING_URL, fallback_groq_key, GROQ_THINKING_MODEL, user_msg, timeout,
        )
    except httpx.RequestError as e:
        last_error = e
        if not fallback_groq_key:
            raise
        logger.warning("OpenRouter network error — falling back to Groq: %r", e)
        content = await _call_chat(
            GROQ_THINKING_URL, fallback_groq_key, GROQ_THINKING_MODEL, user_msg, timeout,
        )

    if not content.strip():
        raise ValueError(f"Empty response from thinking model (last_error={last_error!r})")
    data = _extract_json(content)

    return ThinkingResult(
        chosen_id=str(data.get("chosen_id", "")),
        instruction=str(data.get("instruction", "")).strip(),
        example=str(data.get("example", "")).strip(),
        rationale=str(data.get("rationale", "")).strip(),
    )
