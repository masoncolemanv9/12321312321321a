"""Real ``yt-dlp`` based downloader worker.

Lifecycle:

1. ``extract_info(download=False)`` — pre-flight: check the source exists,
   read its size and duration. Reject with a clear error if the file would
   exceed :data:`bot.config.DOWNLOAD_MAX_FILESIZE_MB`.
2. ``YoutubeDL.download([url])`` — actually fetch the video, merging
   bestvideo + bestaudio into a single MP4 via ffmpeg.
3. Persist the yt-dlp ``info_dict`` to ``metadata.json`` and emit a
   structured ``result`` dict the analyzer stage can consume.

All filesystem writes go under ``DOWNLOADS_DIR / <job_id>/`` so concurrent
downloads can't collide. Progress is streamed to ``download.log`` in the
same dir; bot-side progress messages are wired up by a follow-up tentacle.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

from ..config import (
    DOWNLOAD_MAX_FILESIZE_MB,
    DOWNLOAD_MAX_HEIGHT,
    DOWNLOADS_DIR,
    YT_COOKIES_FILE,
)
from ..jobs import Job
from .base import Worker

logger = logging.getLogger(__name__)


class DownloaderWorker(Worker):
    kind = "download"
    next_kind = "analyze"

    async def _handle(self, job: Job) -> None:
        """Override base ``_handle`` to surface failures to chat.

        Base ``Worker._handle`` silently marks the job ``failed`` and
        logs the traceback — fine for /dl pipeline jobs (the user
        polls ``/jobs``) but it leaves the «скачай <url>» (simple-only)
        flow looking dead. The user just sees «⬇️ Скачиваю» forever.
        For those jobs we additionally send a short, human-readable
        error message to the chat so the user knows what happened.
        """
        import traceback as _tb

        logger.info(
            "worker[%s] processing job=%s parent=%s",
            self.kind,
            job.id,
            job.parent_id,
        )
        try:
            result = await self.process(job)
        except Exception as exc:  # noqa: BLE001
            tb = _tb.format_exc()
            logger.exception(
                "worker[%s] job=%s failed: %s", self.kind, job.id, exc
            )
            await self.queue.mark_failed(job.id, tb)
            if job.payload.get("simple_only") and job.chat_id is not None:
                try:
                    await _notify_simple_download_failed(
                        job.chat_id, job.id, exc
                    )
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "[download] simple-only error notify failed "
                        "for job=%s",
                        job.id,
                    )
            return

        await self.queue.mark_done(job.id, result)
        try:
            await self.enqueue_next(job, result)
        except Exception:
            logger.exception(
                "worker[%s] job=%s succeeded but enqueue_next failed",
                self.kind,
                job.id,
            )

    async def enqueue_next(self, job: Job, result: dict[str, Any]) -> None:
        """Skip auto-chain to «analyze» when ``simple_only`` is set.

        The natural-language «скачай <url>» flow downloads a single file
        and pings the user with a «🎞 Монтажёр» button instead of
        running the full 5-stage publish pipeline. Setting
        ``simple_only=True`` on the original download payload (the only
        place we can stash a hint without a schema migration) is how the
        text handler tells the worker to stop after stage 1.

        Notification is fire-and-forget: any aiogram import / send
        failure is logged but doesn't fail the job — the file is still
        on disk and the user can re-trigger via ``/jobs`` if needed.
        """
        if not job.payload.get("simple_only"):
            await super().enqueue_next(job, result)
            return

        if job.chat_id is None:
            return
        try:
            await _notify_simple_download(job.chat_id, job.id, result)
        except Exception:  # noqa: BLE001
            logger.exception(
                "[download] simple-only notify failed for job=%s", job.id
            )

    async def process(self, job: Job) -> dict[str, Any]:
        url = (job.payload.get("url") or "").strip()
        if not url:
            raise ValueError("download payload missing 'url'")
        max_height = int(job.payload.get("max_height") or DOWNLOAD_MAX_HEIGHT)
        max_filesize_mb = int(
            job.payload.get("max_filesize_mb") or DOWNLOAD_MAX_FILESIZE_MB
        )
        max_filesize_bytes = max_filesize_mb * 1024 * 1024

        job_dir = DOWNLOADS_DIR / str(job.id)
        job_dir.mkdir(parents=True, exist_ok=True)
        log_path = job_dir / "download.log"

        # yt-dlp is sync (blocking); run it in the default executor so the
        # async worker loop stays responsive.
        return await asyncio.to_thread(
            self._download_blocking,
            url=url,
            job_dir=job_dir,
            log_path=log_path,
            max_height=max_height,
            max_filesize_bytes=max_filesize_bytes,
        )

    @staticmethod
    def _download_blocking(
        *,
        url: str,
        job_dir: Path,
        log_path: Path,
        max_height: int,
        max_filesize_bytes: int,
    ) -> dict[str, Any]:
        # Pre-flight: extract metadata only. Cheaper than downloading and
        # gives us a chance to reject oversized sources up front.
        info: dict[str, Any] | None = None
        last_exc: DownloadError | None = None
        winning_client: str | None = None
        # Try default client first; fall back to mobile/TV clients
        # whenever YouTube greets us with «Sign in to confirm you're
        # not a bot». Most ordinary URLs work on the first attempt and
        # never enter the fallback loop.
        for player_client in (None, *_YT_PLAYER_CLIENT_FALLBACKS):
            info_opts = _ydl_options(
                job_dir=job_dir,
                log_path=log_path,
                max_height=max_height,
                quiet=True,
                player_client=player_client,
            )
            with YoutubeDL(info_opts) as ydl:
                try:
                    info = ydl.extract_info(url, download=False)
                except DownloadError as exc:
                    last_exc = exc
                    info = None
                    if not _is_yt_bot_check(exc):
                        # Non-bot failure (private video, geo, 404, …) —
                        # changing the player client won't help; bail
                        # out and surface the original error.
                        break
                    logger.info(
                        "[download] yt-dlp pre-flight tripped bot-check "
                        "(client=%s); retrying with next fallback",
                        player_client or "default",
                    )
                    continue
            if info is not None:
                winning_client = player_client
                if player_client:
                    logger.info(
                        "[download] yt-dlp succeeded via player_client=%s",
                        player_client,
                    )
                break
        if info is None:
            if last_exc is not None:
                raise RuntimeError(
                    f"yt-dlp pre-flight failed: {last_exc}"
                ) from last_exc
            raise RuntimeError("yt-dlp pre-flight returned no metadata")
        # Some extractors return playlists; we only handle a single entry.
        if info.get("_type") == "playlist":
            entries = info.get("entries") or []
            if not entries:
                raise RuntimeError("yt-dlp returned an empty playlist")
            info = entries[0]

        approx_bytes = (
            info.get("filesize")
            or info.get("filesize_approx")
            or 0
        )
        if approx_bytes and approx_bytes > max_filesize_bytes:
            raise RuntimeError(
                f"source too big: ~{approx_bytes // (1024 * 1024)} MB exceeds "
                f"cap of {max_filesize_bytes // (1024 * 1024)} MB"
            )

        # Real download. Reuse the player_client that survived the
        # pre-flight (if any) so we don't trip the bot-check on the
        # actual fetch.
        download_opts = _ydl_options(
            job_dir=job_dir,
            log_path=log_path,
            max_height=max_height,
            quiet=False,
            player_client=winning_client,
        )
        with YoutubeDL(download_opts) as ydl:
            try:
                final_info = ydl.extract_info(url, download=True)
            except DownloadError as exc:
                raise RuntimeError(f"yt-dlp download failed: {exc}") from exc
        if final_info is None:
            raise RuntimeError("yt-dlp returned no info after download")
        if final_info.get("_type") == "playlist":
            entries = final_info.get("entries") or []
            if not entries:
                raise RuntimeError("yt-dlp playlist had no entries")
            final_info = entries[0]

        # Persist the info_dict next to the source. ``default=str`` handles
        # the few non-JSON-able fields yt-dlp occasionally returns (paths,
        # datetime objects).
        metadata_path = job_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(final_info, ensure_ascii=False, indent=2, default=str)
        )

        # Locate the actual on-disk file. yt-dlp may have transmuxed it.
        source_path = _resolve_source_path(final_info, job_dir)
        if source_path is None or not source_path.exists():
            raise RuntimeError(
                "downloaded file not found on disk; check yt-dlp output template"
            )

        return {
            "url": url,
            "source_path": str(source_path),
            "metadata_path": str(metadata_path),
            "duration_s": float(final_info.get("duration") or 0.0),
            "title": final_info.get("title") or "",
            "uploader": final_info.get("uploader") or "",
            "original_url": final_info.get("webpage_url") or url,
            "width": final_info.get("width"),
            "height": final_info.get("height"),
            "fps": final_info.get("fps"),
            "language": final_info.get("language"),
        }


def _ydl_options(
    *,
    job_dir: Path,
    log_path: Path,
    max_height: int,
    quiet: bool,
    player_client: str | None = None,
) -> dict[str, Any]:
    """Build a ``YoutubeDL`` option dict shared by pre-flight and download.

    ``player_client`` lets the caller force a specific YouTube player
    client (``android`` / ``ios`` / ``tv_embedded`` / etc.). These are
    the canonical workaround for the «Sign in to confirm you're not
    a bot» 403 — YouTube's web player gates aggressively, but the
    mobile / TV clients still hand out streams without auth.
    """
    outtmpl = str(job_dir / "source.%(ext)s")
    fmt = (
        f"bestvideo[height<={max_height}]+bestaudio/"
        f"best[height<={max_height}]/best"
    )
    opts: dict[str, Any] = {
        "outtmpl": outtmpl,
        "format": fmt,
        "merge_output_format": "mp4",
        "noprogress": False,
        "quiet": quiet,
        "no_warnings": quiet,
        "logger": _FileLogger(log_path),
        "progress_hooks": [_make_progress_hook(log_path)],
        "retries": 3,
        "fragment_retries": 3,
        # Stay polite — single connection, no aria2c speedups by default.
        "concurrent_fragment_downloads": 1,
        # Skip livestreams; they don't make sense as a video source for
        # the pipeline.
        "match_filter": _no_live_filter,
    }
    # Resolve cookies dynamically so an upload via the wizard takes
    # effect on the very next «скачай» without restarting the bot.
    # Falls back to the static ``YT_COOKIES_FILE`` resolved at config
    # import time (which itself prefers the env var over the default
    # data/yt_cookies.txt).
    cookies_path: str | None = None
    from ..config import DATA_DIR as _DATA_DIR

    user_cookies = _DATA_DIR / "yt_cookies.txt"
    if user_cookies.is_file() and user_cookies.stat().st_size > 0:
        cookies_path = str(user_cookies)
    elif YT_COOKIES_FILE:
        cookies_path = YT_COOKIES_FILE
    if cookies_path:
        opts["cookiefile"] = cookies_path
    if player_client:
        opts["extractor_args"] = {"youtube": {"player_client": [player_client]}}
    return opts


# YouTube player clients in decreasing order of bot-detection resistance.
# Order chosen from yt-dlp issue tracker observations (May 2026):
# the TV client rarely needs PO tokens; android/ios usually work for
# stream URLs even when web is gated.
_YT_PLAYER_CLIENT_FALLBACKS: tuple[str, ...] = (
    "tv_embedded",
    "android",
    "ios",
    "web_creator",
)


def _is_yt_bot_check(exc: Exception) -> bool:
    """Heuristic: did the failure look like YouTube's anti-bot gate?"""
    msg = str(exc).lower()
    return (
        "sign in to confirm you" in msg
        or "this video is age-restricted" in msg
        or "cookies" in msg
        and "youtube" in msg
    )


def _no_live_filter(info_dict: dict[str, Any]) -> str | None:
    """yt-dlp ``match_filter``: skip live broadcasts."""
    if info_dict.get("is_live"):
        return "live broadcast not supported"
    return None


def _make_progress_hook(log_path: Path):
    """Append yt-dlp progress events to ``log_path`` for later inspection."""

    def hook(d: dict[str, Any]) -> None:
        status = d.get("status", "?")
        line: str
        if status == "downloading":
            pct = d.get("_percent_str", "??")
            speed = d.get("_speed_str", "??")
            eta = d.get("_eta_str", "??")
            line = f"[downloading] {pct} speed={speed} eta={eta}"
        elif status == "finished":
            filename = d.get("filename", "?")
            line = f"[finished] {filename}"
        else:
            line = f"[{status}] {d.get('filename', '')}"
        try:
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError as exc:
            # Logging failure must not abort a download.
            logger.warning("progress log write failed: %s", exc)

    return hook


class _FileLogger:
    """yt-dlp logger adapter that mirrors output into a job-local file."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path

    def _emit(self, level: str, msg: str) -> None:
        try:
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(f"[{level}] {msg}\n")
        except OSError:
            pass
        getattr(logger, level, logger.info)(msg)

    def debug(self, msg: str) -> None:  # noqa: D401 — yt-dlp interface
        # yt-dlp prefixes "real" debug output with "[debug] ".
        if msg.startswith("[debug] "):
            return
        self._emit("debug", msg)

    def info(self, msg: str) -> None:
        self._emit("info", msg)

    def warning(self, msg: str) -> None:
        self._emit("warning", msg)

    def error(self, msg: str) -> None:
        self._emit("error", msg)


def _resolve_source_path(info: dict[str, Any], job_dir: Path) -> Path | None:
    """Find the concrete file yt-dlp wrote, regardless of extension."""
    # Newer yt-dlp records the final filename in ``requested_downloads``.
    for entry in info.get("requested_downloads") or []:
        path = entry.get("filepath") or entry.get("_filename")
        if path:
            p = Path(path)
            if p.exists():
                return p
    # Older releases just expose ``_filename``.
    legacy = info.get("_filename")
    if legacy and Path(legacy).exists():
        return Path(legacy)
    # Fallback: glob for the merged output template.
    for candidate in job_dir.glob("source.*"):
        if candidate.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}:
            return candidate
    return None


# ---- simple-only chat notification --------------------------------------


async def _notify_simple_download(
    chat_id: int, job_id: int, result: dict[str, Any]
) -> None:
    """Send the freshly-downloaded file to ``chat_id`` + «Монтажёр» button.

    Used by the natural-language «скачай <url>» flow (set via
    ``simple_only=True`` on the download payload). The button enqueues
    an analyze job whose payload points back to ``source_path`` — so the
    full edit→seo→publish chain runs on demand, not automatically.

    Telegram limits documents to 50 MB on bot uploads. If the rendered
    file exceeds that, we fall back to a text message with the on-disk
    path so the user can pull it via ``/work`` or another channel.
    """
    from aiogram import Bot
    from aiogram.client.default import DefaultBotProperties
    from aiogram.enums import ParseMode
    from aiogram.types import (
        FSInputFile,
        InlineKeyboardButton,
        InlineKeyboardMarkup,
    )

    from ..config import BOT_TOKEN

    source = Path(result.get("source_path") or "")
    title = (result.get("title") or "").strip()
    url = result.get("original_url") or result.get("url") or ""

    kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="🎞 Монтажёр",
                    callback_data=f"dl:edit:{job_id}",
                )
            ]
        ]
    )

    # Render a clickable, paragraph-spaced caption. HTML keeps the
    # source URL as a real link rather than raw text, and the blank
    # line between "Готово" and the URL line gives the message air to
    # breathe (the user explicitly asked for this style).
    caption_lines: list[str] = ["✅ <b>Видео скачано</b>"]
    if title:
        caption_lines.append("")
        caption_lines.append(f"<i>{_html_escape_short(title)}</i>")
    if url:
        caption_lines.append("")
        caption_lines.append(f"🔗 <a href=\"{url}\">источник</a>")
    caption_lines.append("")
    caption_lines.append(
        "Нажми <b>🎞 Монтажёр</b> чтобы запустить нарезку и SEO."
    )
    caption = "\n".join(caption_lines)

    bot = Bot(
        BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    try:
        # 50 MB hard cap for bot file uploads on Telegram. Pad slightly
        # below to avoid edge-case rejects on the network.
        try:
            size_mb = source.stat().st_size / (1024 * 1024) if source.exists() else 0
        except OSError:
            size_mb = 0

        sent = False
        if source.exists() and size_mb <= 48:
            try:
                await bot.send_video(
                    chat_id=chat_id,
                    video=FSInputFile(str(source)),
                    caption=caption,
                    reply_markup=kb,
                    supports_streaming=True,
                )
                sent = True
            except Exception:  # noqa: BLE001
                # Some sources are .webm / .mkv etc. — Telegram rejects
                # those as ``send_video``; retry as document.
                logger.exception(
                    "[download] send_video failed for job=%s, retrying as document",
                    job_id,
                )
        if not sent and source.exists() and size_mb <= 48:
            await bot.send_document(
                chat_id=chat_id,
                document=FSInputFile(str(source)),
                caption=caption,
                reply_markup=kb,
            )
            sent = True
        if not sent:
            # Either the file is missing or it's bigger than Telegram
            # allows. Tell the user where to find it so they can pull
            # it manually.
            await bot.send_message(
                chat_id=chat_id,
                text=(
                    caption
                    + "\n\n⚠️ Файл слишком большой для Telegram "
                    f"({size_mb:.1f} MB). Лежит на сервере: "
                    f"<code>{_html_escape_short(str(source))}</code>"
                ),
                reply_markup=kb,
                disable_web_page_preview=True,
            )
    finally:
        await bot.session.close()


def _html_escape_short(s: str) -> str:
    """Tiny HTML-escape — keeps the worker free of html-stdlib dependency.

    Telegram's HTML parse mode only needs ``< > &`` escaped. Quotes are
    safe inside ``<i>`` / ``<code>`` content.
    """
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


async def _notify_simple_download_failed(
    chat_id: int, job_id: int, exc: Exception
) -> None:
    """Send a short human-readable error to chat when simple_only fails.

    The user sees something like:
        ❌ Не получилось скачать.
        YouTube требует cookies / залогинен браузер. Попробуй из инстаграма / TikTok.

    Special-cases the two most common yt-dlp failures so the user
    knows what to try next instead of seeing the raw traceback.
    """
    from aiogram import Bot
    from aiogram.client.default import DefaultBotProperties
    from aiogram.enums import ParseMode

    import os

    token = os.environ.get("BOT_TOKEN", "")
    if not token:
        return

    msg = str(exc) or exc.__class__.__name__
    msg_lower = msg.lower()
    hint = ""
    if "sign in to confirm" in msg_lower or "cookies" in msg_lower:
        hint = (
            "\n\n<b>YouTube</b> требует <i>cookies</i> авторизованного "
            "браузера (свежий anti-bot).\n\n"
            "Сделай так:\n"
            "1. Открой главное меню → <b>📥 Скачать видео</b> → "
            "<b>🍪 Загрузить cookies</b>\n"
            "2. Пришли файл <code>cookies.txt</code>, "
            "экспортированный из браузера (расширение «Get cookies.txt LOCALLY»)\n\n"
            "Или просто пришли ссылку с <b>Instagram</b>, <b>TikTok</b>, <b>X</b> — "
            "там обычно работает без cookies."
        )
    elif "private video" in msg_lower or "members-only" in msg_lower:
        hint = (
            "\n\nВидео <b>приватное</b> или для подписчиков канала. "
            "Без аккаунта-владельца не возьмёшь."
        )
    elif "geographic" in msg_lower or "geo" in msg_lower:
        hint = "\n\nВидео <b>заблокировано в этом регионе</b>. Нужен VPN."
    elif "filesize" in msg_lower or "too large" in msg_lower:
        hint = (
            "\n\nВидео <b>слишком большое</b>. Попробуй короче или "
            "снизь качество в настройках."
        )

    safe_msg = _html_escape_short(msg[:700])

    bot = Bot(
        token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    try:
        await bot.send_message(
            chat_id=chat_id,
            text=(
                "❌ <b>Не получилось скачать.</b>\n\n"
                f"<code>{safe_msg}</code>"
                + hint
            ),
            disable_web_page_preview=True,
        )
    finally:
        await bot.session.close()
