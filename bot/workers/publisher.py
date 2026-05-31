"""Publisher stage — DRY-RUN release packager.

In this release the publisher does NOT call any upload API. Instead it
stages a *release directory* per clip:

```
data/releases/<job_id>/
├── clip.mp4
├── youtube.json
├── tiktok.json
├── instagram.json
└── upload-instructions.md
```

Real uploaders (``publisher-youtube-uploader``, ``publisher-tiktok-uploader``,
``publisher-instagrapi``) will live as separate tentacles and consume
those release dirs directly. Keeping the metadata format stable now lets
us iterate on the upload layer later without breaking the rest of the
pipeline.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

from .. import config as _config
from ..jobs import Job
from .base import Worker

logger = logging.getLogger(__name__)


class PublisherWorker(Worker):
    kind = "publish"
    next_kind = None  # terminal stage

    async def process(self, job: Job) -> dict[str, Any]:
        clip_path_str = job.payload.get("clip_path")
        if not clip_path_str:
            raise ValueError("publish payload missing 'clip_path'")
        clip_path = Path(clip_path_str)
        if not clip_path.exists():
            raise FileNotFoundError(f"publish clip not found: {clip_path}")

        seo_title = str(job.payload.get("seo_title") or "").strip()
        seo_description = str(job.payload.get("seo_description") or "").strip()
        tags_raw = job.payload.get("tags") or []
        tags = [str(t).strip() for t in tags_raw if str(t).strip()]
        language = str(job.payload.get("language") or "en").lower()
        hook = str(job.payload.get("hook") or "")
        clip_index = int(job.payload.get("clip_index") or 0)
        original_title = str(job.payload.get("original_title") or "")

        return await asyncio.to_thread(
            self._stage_release,
            job_id=job.id,
            clip_path=clip_path,
            seo_title=seo_title,
            seo_description=seo_description,
            tags=tags,
            language=language,
            hook=hook,
            clip_index=clip_index,
            original_title=original_title,
        )

    @staticmethod
    def _stage_release(
        *,
        job_id: int,
        clip_path: Path,
        seo_title: str,
        seo_description: str,
        tags: list[str],
        language: str,
        hook: str,
        clip_index: int,
        original_title: str,
    ) -> dict[str, Any]:
        release_dir = _config.RELEASES_DIR / str(job_id)
        release_dir.mkdir(parents=True, exist_ok=True)

        staged_clip = release_dir / "clip.mp4"
        _link_or_copy(clip_path, staged_clip)

        yt_meta = build_youtube_metadata(
            title=seo_title, description=seo_description, tags=tags, language=language
        )
        tiktok_meta = build_tiktok_metadata(
            title=seo_title, description=seo_description, tags=tags, language=language
        )
        ig_meta = build_instagram_metadata(
            title=seo_title, description=seo_description, tags=tags, language=language
        )

        yt_path = release_dir / "youtube.json"
        tiktok_path = release_dir / "tiktok.json"
        ig_path = release_dir / "instagram.json"
        instructions_path = release_dir / "upload-instructions.md"

        yt_path.write_text(json.dumps(yt_meta, ensure_ascii=False, indent=2))
        tiktok_path.write_text(json.dumps(tiktok_meta, ensure_ascii=False, indent=2))
        ig_path.write_text(json.dumps(ig_meta, ensure_ascii=False, indent=2))
        instructions_path.write_text(
            _build_instructions(
                clip_path=staged_clip,
                hook=hook,
                clip_index=clip_index,
                original_title=original_title,
                seo_title=seo_title,
            )
        )

        return {
            "dry_run": _config.PUBLISHER_DRY_RUN,
            "release_dir": str(release_dir),
            "clip_path": str(staged_clip),
            "metadata_paths": {
                "youtube": str(yt_path),
                "tiktok": str(tiktok_path),
                "instagram": str(ig_path),
            },
            "instructions_path": str(instructions_path),
            "platforms": ["youtube", "tiktok", "instagram"],
            "title": seo_title,
            "post_url": None,
        }

    async def enqueue_next(self, job: Job, result: dict[str, Any]) -> None:
        """Terminal stage — notify owner via Telegram if a chat is set."""
        if job.chat_id is None:
            return
        try:
            from aiogram import Bot
            from aiogram.client.default import DefaultBotProperties
            from aiogram.enums import ParseMode

            from ..config import BOT_TOKEN
        except Exception:
            logger.exception("[publisher] could not import aiogram for chat report")
            return

        title = result.get("title", "")
        release_dir = result.get("release_dir", "")
        dry_run = result.get("dry_run")
        text = (
            "🚀 <b>Релиз готов</b>\n\n"
            + (f"<i>{title}</i>\n\n" if title else "")
            + (f"📁 <code>{release_dir}</code>\n" if release_dir else "")
            + (
                "🧪 Тестовый прогон (DRY-RUN) — реальной публикации не было."
                if dry_run
                else "✅ Опубликовано."
            )
        )
        bot = Bot(BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
        try:
            await bot.send_message(job.chat_id, text)
        finally:
            await bot.session.close()


# ---- staging helpers -----------------------------------------------------


def _link_or_copy(src: Path, dst: Path) -> None:
    """Hard-link the clip into the release dir; fall back to copy."""
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _build_instructions(
    *,
    clip_path: Path,
    hook: str,
    clip_index: int,
    original_title: str,
    seo_title: str,
) -> str:
    return (
        "# Upload instructions (DRY-RUN)\n\n"
        f"Clip: `{clip_path}`\n"
        f"Hook: {hook}\n"
        f"Clip index: {clip_index}\n"
        f"Original source title: {original_title}\n"
        f"Generated SEO title: {seo_title}\n\n"
        "## YouTube Shorts\n"
        "1. https://studio.youtube.com → CREATE → Upload Video.\n"
        "2. Drag `clip.mp4`. Use fields from `youtube.json`.\n"
        "3. After review, switch privacy from `private` to `public`.\n\n"
        "## TikTok\n"
        "1. https://www.tiktok.com/upload\n"
        "2. Use caption + hashtags from `tiktok.json`.\n"
        "3. Set 'Who can watch' = Everyone after review.\n\n"
        "## Instagram Reels\n"
        "1. Open Instagram mobile app → Reels → +.\n"
        "2. Use caption from `instagram.json` (already includes hashtags).\n"
    )


# ---- metadata builders ---------------------------------------------------


def build_youtube_metadata(
    *,
    title: str,
    description: str,
    tags: list[str],
    language: str,
) -> dict[str, Any]:
    """Format compatible with the YouTube Data API ``videos.insert`` body."""
    return {
        "snippet": {
            "title": title or "Untitled clip",
            "description": description,
            "tags": tags,
            "categoryId": _config.PUBLISHER_YT_CATEGORY_ID,
            "defaultLanguage": language,
            "defaultAudioLanguage": language,
        },
        "status": {
            "privacyStatus": _config.PUBLISHER_YT_PRIVACY,
            "selfDeclaredMadeForKids": False,
            "embeddable": True,
        },
    }


def build_tiktok_metadata(
    *,
    title: str,
    description: str,
    tags: list[str],
    language: str,
) -> dict[str, Any]:
    hashtags = [_to_hashtag(t) for t in tags if _to_hashtag(t)]
    caption = title
    if description:
        caption = f"{title}\n\n{description}" if title else description
    # TikTok caption limit is currently 2200 chars; trim defensively.
    caption = caption[:2200]
    return {
        "caption": caption,
        "hashtags": hashtags[:30],
        "language": language,
        "schedule_at": None,
        "disable_duet": False,
        "disable_stitch": False,
        "disable_comment": False,
        "private": True,
    }


def build_instagram_metadata(
    *,
    title: str,
    description: str,
    tags: list[str],
    language: str,
) -> dict[str, Any]:
    hashtag_block = " ".join(f"#{_to_hashtag(t)}" for t in tags if _to_hashtag(t))
    pieces: list[str] = []
    if title:
        pieces.append(title)
    if description:
        pieces.append(description)
    if hashtag_block and hashtag_block not in description:
        pieces.append(hashtag_block)
    caption = "\n\n".join(pieces)[:2200]
    return {
        "caption": caption,
        "share_to_feed": True,
        "language": language,
    }


def _to_hashtag(s: str) -> str:
    """Strip punctuation/spaces so the result is a valid hashtag."""
    return "".join(ch for ch in s.lstrip("#") if ch.isalnum())
