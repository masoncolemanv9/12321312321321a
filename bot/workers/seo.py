"""SEO stage: title/description/tags for a finished short.

For each ``edit`` result we:

1. Read the transcript JSON (if present) for context.
2. Pull a handful of currently-trending keywords for the clip's
   language via ``pytrends`` (best-effort — gracefully degrades if
   Google rate-limits us or the package isn't available).
3. Ask the LLM (OpenRouter) to write a hooky title, a description
   ending in 4-7 hashtags, and a list of search tags. If no key is
   set or the call fails, fall back to a deterministic template.

The output goes to the ``publish`` stage which stamps it onto the
final upload metadata blob.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from .. import config as _config
from ..jobs import Job
from .base import Worker

logger = logging.getLogger(__name__)


class SeoWorker(Worker):
    kind = "seo"
    next_kind = "publish"

    async def process(self, job: Job) -> dict[str, Any]:
        clip_path = str(job.payload.get("clip_path") or "")
        hook = str(job.payload.get("hook") or "")
        original_title = str(job.payload.get("original_title") or "")
        uploader = str(job.payload.get("uploader") or "")
        language = (job.payload.get("language") or "").lower() or None
        transcript_path = job.payload.get("transcript_path")

        return await asyncio.to_thread(
            self._build_blocking,
            clip_path=clip_path,
            hook=hook,
            original_title=original_title,
            uploader=uploader,
            language=language,
            transcript_path=transcript_path,
            payload=job.payload,
        )

    @staticmethod
    def _build_blocking(
        *,
        clip_path: str,
        hook: str,
        original_title: str,
        uploader: str,
        language: str | None,
        transcript_path: Any,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        excerpt, transcript_lang = _read_transcript(transcript_path)
        final_lang = language or transcript_lang or "en"
        trends = fetch_trends(language=final_lang)

        seo, method = generate_seo(
            hook=hook,
            original_title=original_title,
            uploader=uploader,
            transcript_excerpt=excerpt,
            language=final_lang,
            trends=trends,
        )

        return {
            "clip_path": clip_path,
            "source_path": payload.get("source_path"),
            "clip_index": payload.get("clip_index"),
            "hook": hook,
            "language": final_lang,
            "trends_used": trends,
            "seo_method": method,
            "seo_title": seo["title"],
            "seo_description": seo["description"],
            "tags": seo["tags"],
        }


# ---- transcript ----------------------------------------------------------


def _read_transcript(transcript_path: Any) -> tuple[str, str | None]:
    """Return ``(excerpt, language)`` from the analyzer's transcript.json."""
    if not transcript_path:
        return "", None
    path = Path(str(transcript_path))
    if not path.exists():
        return "", None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return "", None
    segments = data.get("segments") or []
    text = " ".join(str(s.get("text", "")).strip() for s in segments).strip()
    return text[:1500], data.get("language")


# ---- trends --------------------------------------------------------------


def fetch_trends(*, language: str) -> list[str]:
    """Best-effort trending-keyword fetch via pytrends.

    Returns ``[]`` on any failure — Google rate-limits aggressively.
    """
    try:
        from pytrends.request import TrendReq
    except ImportError:
        logger.info("pytrends not available; skipping trends")
        return []

    geo, hl = _geo_for_language(language)
    try:
        client = TrendReq(hl=hl, tz=0)
        df = client.trending_searches(pn=_pytrends_country(geo))
        # Return the top-15 strings from column 0.
        values = list(df.iloc[:15, 0])
        return [str(v).strip() for v in values if str(v).strip()]
    except Exception as exc:  # noqa: BLE001 — pytrends raises a zoo of types
        logger.info("pytrends fetch failed (%s); using empty trends", exc)
        return []


def _geo_for_language(language: str) -> tuple[str, str]:
    table = {
        "ru": ("RU", "ru-RU"),
        "en": ("US", "en-US"),
        "es": ("ES", "es-ES"),
        "pt": ("BR", "pt-BR"),
        "de": ("DE", "de-DE"),
        "fr": ("FR", "fr-FR"),
        "ja": ("JP", "ja"),
        "uk": ("UA", "uk"),
    }
    return table.get(language[:2], ("", "en-US"))


def _pytrends_country(geo: str) -> str:
    """pytrends ``trending_searches`` takes a country slug (e.g. 'united_states')."""
    table = {
        "RU": "russia",
        "US": "united_states",
        "ES": "spain",
        "BR": "brazil",
        "DE": "germany",
        "FR": "france",
        "JP": "japan",
        "UA": "ukraine",
    }
    return table.get(geo, "united_states")


# ---- LLM / template ------------------------------------------------------


def generate_seo(
    *,
    hook: str,
    original_title: str,
    uploader: str,
    transcript_excerpt: str,
    language: str,
    trends: list[str],
) -> tuple[dict[str, Any], str]:
    """Try the LLM ranker; fall back to template on any failure."""
    if _config.OPENROUTER_API_KEY:
        try:
            seo = _generate_with_llm(
                hook=hook,
                original_title=original_title,
                uploader=uploader,
                transcript_excerpt=transcript_excerpt,
                language=language,
                trends=trends,
            )
            return _normalize(seo, language=language, trends=trends), "llm"
        except Exception as exc:  # noqa: BLE001 — fallback path
            logger.warning("SEO LLM failed, using template: %s", exc)
    return (
        _normalize(
            _template(
                hook=hook,
                original_title=original_title,
                language=language,
                trends=trends,
            ),
            language=language,
            trends=trends,
        ),
        "template",
    )


def _normalize(
    seo: dict[str, Any], *, language: str, trends: list[str]
) -> dict[str, Any]:
    """Trim title/description/tags to platform-friendly bounds."""
    title = str(seo.get("title") or "").strip()
    description = str(seo.get("description") or "").strip()
    tags_raw = seo.get("tags") or []
    if isinstance(tags_raw, str):
        tags_raw = [t.strip() for t in tags_raw.split(",")]

    tags: list[str] = []
    seen: set[str] = set()
    for entry in tags_raw:
        clean = str(entry).lstrip("#").strip()
        if not clean:
            continue
        key = clean.lower()
        if key in seen:
            continue
        seen.add(key)
        tags.append(clean[:30])
        if len(tags) >= _config.SEO_MAX_TAGS:
            break

    title = title[: _config.SEO_TITLE_MAX_LEN]
    description = description[: _config.SEO_DESCRIPTION_MAX_LEN]
    return {
        "title": title or "untitled clip",
        "description": description,
        "tags": tags,
    }


def _template(
    *,
    hook: str,
    original_title: str,
    language: str,
    trends: list[str],
) -> dict[str, Any]:
    title = (hook or original_title or "untitled clip").strip()
    base_tags = ["shorts", "viral", language, "fyp", "reels"]
    tags = base_tags + [t for t in trends[:10]]
    hashtags = " ".join(f"#{_to_hashtag(t)}" for t in tags[:5])
    description_parts: list[str] = []
    if hook:
        description_parts.append(hook)
    if original_title:
        description_parts.append(f"From: {original_title}")
    description_parts.append(hashtags)
    return {
        "title": title,
        "description": "\n\n".join(description_parts),
        "tags": tags,
    }


def _to_hashtag(s: str) -> str:
    """Strip punctuation/spaces so the result is a valid hashtag."""
    return "".join(ch for ch in s if ch.isalnum())


def _generate_with_llm(
    *,
    hook: str,
    original_title: str,
    uploader: str,
    transcript_excerpt: str,
    language: str,
    trends: list[str],
) -> dict[str, Any]:
    import httpx

    system = (
        "You are an SEO/marketing copywriter for short-form video on YouTube "
        "Shorts, TikTok, and Instagram Reels. Output STRICT JSON with three "
        f"fields: title (≤{_config.SEO_TITLE_MAX_LEN} chars, "
        f"language={language}), description (≤{_config.SEO_DESCRIPTION_MAX_LEN} "
        "chars, ends with 4-7 hashtags), and tags (list of search keywords, "
        f"max {_config.SEO_MAX_TAGS} items, each ≤30 chars, language={language}). "
        "The title MUST grab attention in 2 seconds. Do not include markdown."
    )
    user = json.dumps(
        {
            "clip_hook": hook,
            "transcript_excerpt": transcript_excerpt,
            "original_title": original_title,
            "uploader": uploader,
            "trending_now": trends[:15],
        },
        ensure_ascii=False,
    )
    payload = {
        "model": _config.SEO_LLM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.6,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {_config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    resp = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        json=payload,
        headers=headers,
        timeout=60.0,
    )
    resp.raise_for_status()
    body = resp.json()
    try:
        content = body["choices"][0]["message"]["content"]
        parsed = json.loads(content)
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"SEO LLM returned unparseable response: {exc}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("SEO LLM JSON is not an object")
    return parsed
