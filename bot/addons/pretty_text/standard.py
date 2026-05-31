"""The "📰 Стандарт" style — LLM adds formatting WITHOUT touching words.

The original ``standard`` style used to fully rewrite the user's text in
a "Telegram-post" voice (see :data:`_LEGACY_REWRITE_SYSTEM_PROMPT` for
the old prompt). The user complained: «pretty text» should only
decorate, not paraphrase.

This module replaces that behavior with a strict "decoration-only" pass:

1. Send the original text to the LLM with a prompt that says
   *only* insert ``**bold**``, ``*italic*``, line-breaks and emojis at
   the start of paragraphs / lines. Don't add, remove, or change any
   word.
2. After the LLM responds, strip every Markdown marker and emoji from
   its output and from the original text, then normalise whitespace and
   case. If the two strings don't match, the LLM violated the rule —
   we fall back to plain mechanical formatting and log a warning so it
   can be investigated.

That guarantees the user-visible text is never paraphrased even if the
model is sloppy.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import unicodedata

logger = logging.getLogger(__name__)

__all__ = ["rewrite_standard_decor_only", "verify_words_unchanged"]

# Hard ceiling on the LLM call so the user never sits more than ~30s
# waiting for the standard-style decoration. If the model is slow or
# rate-limited we fall back to mechanical formatting instead of
# hanging the chat for minutes.
_STANDARD_LLM_TIMEOUT_S = 30.0


_STANDARD_DECOR_SYSTEM_PROMPT = """\
Ты — оформитель текста для Telegram. Ты получаешь произвольный текст и \
возвращаешь его обратно, добавив ТОЛЬКО оформление. Правила строгие:

ЧТО МОЖНО:
- Добавлять **жирный** (Markdown) на отдельные слова и фразы.
- Добавлять *курсив* (одна звёздочка) на отдельные слова и фразы.
- Добавлять эмодзи в начале абзацев или строк (по одной штуке).
- Добавлять переносы строк между смысловыми блоками.
- Добавлять пустые строки для разделения абзацев.

ЧТО НЕЛЬЗЯ:
- НЕЛЬЗЯ менять слова.
- НЕЛЬЗЯ переписывать предложения «другими словами».
- НЕЛЬЗЯ добавлять новые слова, фразы, заголовки, выводы, призывы к \
действию.
- НЕЛЬЗЯ удалять слова или предложения.
- НЕЛЬЗЯ менять порядок слов или предложений.
- НЕЛЬЗЯ исправлять опечатки.
- НЕЛЬЗЯ переводить на другой язык.

ПРОВЕРКА: после твоего ответа я уберу всё оформление и сравню текст с \
оригиналом посимвольно. Если хотя бы одно слово отличается — твой \
ответ будет отброшен и пользователь увидит текст без оформления. \
Поэтому ничего не сочиняй.

Верни ТОЛЬКО оформленный текст, без преамбулы.
"""


_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF"
    "\U0001F900-\U0001F9FF\U0001FA70-\U0001FAFF\u2700-\u27BF\u2300-\u23FF"
    "\u2B00-\u2BFF\uFE0F]"
)

_MD_MARKERS_RE = re.compile(r"\*+|_+|`+|~+|#+|>+|\[|\]\(|\)|\\")


def _strip_decoration(text: str) -> str:
    """Remove emoji, Markdown markers, and collapse whitespace so the
    remainder is comparable letter-for-letter to the original.
    """
    s = _EMOJI_RE.sub("", text)
    s = _MD_MARKERS_RE.sub("", s)
    # Drop punctuation/spacing/zero-width — keep alphanumerics & dashes.
    s = "".join(
        ch
        for ch in unicodedata.normalize("NFKC", s)
        if not unicodedata.category(ch).startswith(("P", "Z", "C"))
    )
    return s.casefold()


def verify_words_unchanged(original: str, decorated: str) -> bool:
    """Return True iff stripping decoration from ``decorated`` yields
    exactly the same character sequence as ``original`` (case-folded,
    punctuation/space-normalised). The caller falls back to mechanical
    formatting when this returns False.
    """
    return _strip_decoration(original) == _strip_decoration(decorated)


async def rewrite_standard_decor_only(raw_text: str) -> str:
    """Ask the LLM to decorate ``raw_text`` without changing any word.

    On success returns the LLM's output. On any failure (no API key,
    model refusal, network error, timeout, or word-mismatch verification)
    raises a ``RuntimeError`` — the caller then falls back to mechanical
    formatting and surfaces a short user-facing warning.

    The LLM call is bounded by ``_STANDARD_LLM_TIMEOUT_S`` (30s) so the
    chat doesn't freeze on slow upstream models.
    """
    from ...agent import NoApiKeyError, _build_client
    from ...config import DEFAULT_MODEL
    from ...storage import storage

    try:
        client = _build_client()
    except NoApiKeyError as exc:
        raise RuntimeError(str(exc)) from exc

    model = storage.get_model() or DEFAULT_MODEL
    t0 = time.monotonic()
    logger.info(
        "pretty-text standard: calling %s (len=%d, timeout=%.0fs)",
        model,
        len(raw_text),
        _STANDARD_LLM_TIMEOUT_S,
    )
    try:
        resp = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": _STANDARD_DECOR_SYSTEM_PROMPT},
                    {"role": "user", "content": raw_text},
                ],
            ),
            timeout=_STANDARD_LLM_TIMEOUT_S,
        )
    except TimeoutError as exc:
        logger.warning(
            "pretty-text standard: LLM call exceeded %.0fs (model=%s)",
            _STANDARD_LLM_TIMEOUT_S,
            model,
        )
        raise RuntimeError(
            f"Модель не ответила за {_STANDARD_LLM_TIMEOUT_S:.0f} секунд."
        ) from exc
    logger.info(
        "pretty-text standard: LLM responded in %.2fs", time.monotonic() - t0
    )

    decorated = (resp.choices[0].message.content or "").strip()
    if not decorated:
        raise RuntimeError("LLM вернул пустой ответ")

    if not verify_words_unchanged(raw_text, decorated):
        # The model violated the rule — refuse and let the caller
        # fall back to mechanical formatting.
        logger.warning(
            "pretty-text standard: LLM changed words; falling back. "
            "orig=%r decorated=%r",
            raw_text[:200],
            decorated[:200],
        )
        raise RuntimeError(
            "Модель попыталась переписать текст. "
            "Возвращаю без оформления — попробуй ещё раз или другой стиль."
        )

    return decorated
