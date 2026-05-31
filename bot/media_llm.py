"""LLM helpers used by the media UI (prompt expansion + RU→EN translate).

Re-uses the same provider/key as the main chat brain
(``storage.get_provider_key("openrouter")`` + ``storage.get_model()``)
so users don't have to configure a second LLM.

Both helpers degrade gracefully:

* If no LLM key is configured, ``NoLLMError`` is raised — the caller
  surfaces a friendly message and falls back to using the raw keyword
  string as the prompt (no translation, no expansion).
* If the model returns malformed JSON, the helper tries to salvage as
  much as possible (best-effort line split) before giving up.
"""

from __future__ import annotations

import json
import logging
import re

from openai import AsyncOpenAI
from openai._exceptions import APIError

from .config import APP_TITLE, DEFAULT_MODEL, HTTP_REFERER, OPENROUTER_BASE_URL
from .storage import storage

logger = logging.getLogger(__name__)


class NoLLMError(RuntimeError):
    """Raised when no LLM key is configured anywhere."""


def _build_llm_client() -> tuple[AsyncOpenAI, str]:
    """Build an OpenAI-compatible client + active model name.

    Mirrors ``bot.agent._build_client`` but with a simpler fallback: we
    always read the same ``openrouter`` slot (custom endpoints reuse it
    via ``base_url``).
    """
    api_key = storage.get_provider_key("openrouter")
    if not api_key:
        raise NoLLMError(
            "LLM-мозг не настроен — /setup → выбери мозг и пришли API-ключ. "
            "Без LLM генерация промптов недоступна; пришли промпт сообщением."
        )
    provider = storage.get_provider()
    if provider == "custom":
        base_url = storage.get_base_url() or OPENROUTER_BASE_URL
    else:
        base_url = OPENROUTER_BASE_URL
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        default_headers={"HTTP-Referer": HTTP_REFERER, "X-Title": APP_TITLE},
    )
    model = storage.get_model() or DEFAULT_MODEL
    return client, model


async def _oneshot(messages: list[dict], *, max_tokens: int = 600) -> str:
    """Single non-streamed chat call. Returns the text content."""
    client, model = _build_llm_client()
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=max_tokens,
        )
    except APIError as exc:
        raise NoLLMError(f"LLM API ошибка: {exc}") from exc
    choices = getattr(resp, "choices", None)
    if not choices:
        raise NoLLMError(f"LLM вернул пустой ответ: {resp!r}")
    msg = choices[0].message
    text = (getattr(msg, "content", None) or "").strip()
    if not text:
        raise NoLLMError("LLM вернул пустое сообщение.")
    return text


async def generate_prompt_variants(
    keywords: str,
    *,
    kind: str,  # "photo" or "video"
    count: int = 3,
) -> list[str]:
    """Expand short user keywords into ``count`` polished RU prompts.

    Prompts are designed for image/video diffusion models — descriptive,
    visual, with mood + lighting + composition. Returned strings are
    plain Russian (no quotes, no numbering).
    """
    kind_desc = {
        "photo": "редактирования изображения (img2img — что должно быть на финальной картинке)",
        "video": "анимации одного кадра в видео (img2video — какое движение, ракурс, эмоция)",
    }.get(kind, "генерации картинки")
    system = (
        "Ты помощник, генерирующий промпты для AI-моделей. "
        "Из коротких ключевых слов пользователя выдай несколько разных "
        "развёрнутых промптов на русском языке. "
        "Каждый промпт — одно-два предложения, визуально-описательный, "
        "с указанием стиля, освещения, эмоции/движения, без нумерации и кавычек. "
        "Верни строго JSON-массив строк, без пояснений вокруг."
    )
    user_msg = (
        f"Задача: {kind_desc}.\n"
        f"Ключевые слова пользователя: «{keywords.strip()}»\n"
        f"Сгенерируй {count} разных вариантов промпта."
    )
    text = await _oneshot(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=700,
    )
    return _parse_variants(text, expected=count)


def _parse_variants(text: str, *, expected: int) -> list[str]:
    """Try hard to extract a list of variant strings from LLM output."""
    text = text.strip()
    # Strip markdown code fences if any.
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    # 1) Try strict JSON.
    try:
        data = json.loads(text)
        if isinstance(data, list):
            variants = [str(x).strip() for x in data if str(x).strip()]
            if variants:
                return variants[:expected]
    except json.JSONDecodeError:
        pass
    # 2) Fallback — split by newlines, strip leading numbers/bullets.
    lines = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        s = re.sub(r"^[-*•\d)\.\s]+", "", s).strip()
        s = s.strip('"').strip("«»")
        if s:
            lines.append(s)
    if lines:
        return lines[:expected]
    # 3) Last resort — whole text as a single variant.
    return [text]


async def translate_to_english(text: str) -> str:
    """Translate ``text`` to natural English for diffusion-model prompts.

    Stripped of trailing punctuation/quotes for cleanliness. Falls back
    to the original text on any LLM error.
    """
    text = text.strip()
    if not text:
        return text
    try:
        out = await _oneshot(
            [
                {
                    "role": "system",
                    "content": (
                        "You translate Russian prompts for image/video AI "
                        "models into natural, vivid English. Output ONLY the "
                        "translation, one sentence preferred, no quotes, no "
                        "explanation."
                    ),
                },
                {"role": "user", "content": text},
            ],
            max_tokens=300,
        )
    except NoLLMError as exc:
        logger.warning("translate_to_english failed, using original: %s", exc)
        return text
    out = out.strip().strip('"').strip("'").strip("«»")
    return out or text
