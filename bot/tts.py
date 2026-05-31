"""Voice-reply ("голосовой ответчик") wiring.

Uses Supertonic — a lightning-fast on-device TTS — to narrate text the
bot already sends as a regular message. Code blocks and raw URLs are
stripped from the spoken version so it stays listenable. GitHub repo
links become "репо от создателей <name>".

The Supertonic model is loaded lazily in a worker thread so the bot's
event loop is never blocked.

Public surface used by handlers:

* ``sanitize_for_tts(text)`` — produce a clean string for narration.
* ``synthesize_voice_ogg(text)`` — async, returns OGG/Opus bytes ready
  for ``bot.send_voice``. ``None`` when nothing was synthesisable
  (empty text after sanitising, model still loading, etc.).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Supertonic peaks ~470 MB RSS during the first synthesis (model load +
# ONNX runtime + ~3s of audio). Below this much *available* RAM at the
# moment of synthesis, attempting TTS will OOM-kill the entire bot on
# small hosting boxes (Render Free / Starter = 512 MB total). We skip
# synthesis with a clear log line instead. Override via env if you've
# upgraded the box.
_TTS_MIN_AVAILABLE_MB = int(os.environ.get("TTS_MIN_AVAILABLE_MB", "450"))


# ---- text sanitiser -------------------------------------------------------

_FENCED_CODE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`]+`")
_HTML_PRE = re.compile(r"<pre[\s\S]*?</pre>", re.IGNORECASE)
_HTML_CODE = re.compile(r"<code[\s\S]*?</code>", re.IGNORECASE)
_HTML_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"https?://([^\s<>]+)", re.IGNORECASE)
_WHITESPACE = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_MAX_TTS_CHARS = 1500  # don't try to read essays out loud


def _shorten_github_url(host: str, path: str) -> str:
    """Format a github.com URL as 'репо от создателей <repo>'."""
    parts = [p for p in path.split("/") if p]
    if len(parts) >= 2:
        repo = parts[1].rstrip(".,)")
        return f"репо от создателей {repo}"
    if len(parts) == 1:
        return f"репо {parts[0]}"
    return "репо на гитхабе"


def _shorten_url(match: re.Match) -> str:
    """Replace a URL with its domain (or repo-name for github)."""
    rest = match.group(1)
    host, _, path = rest.partition("/")
    host = host.lower().rstrip(".,)")
    if host.startswith("www."):
        host = host[4:]
    if host == "github.com" or host.endswith(".github.com"):
        return _shorten_github_url(host, path)
    return host or "ссылка"


def sanitize_for_tts(text: str) -> str:
    """Strip code/HTML, shorten URLs, normalise whitespace.

    Returns possibly the empty string when nothing speakable is left
    (e.g. the bot answered with only a code block).
    """
    if not text:
        return ""
    cleaned = text
    cleaned = _FENCED_CODE.sub(" ", cleaned)
    cleaned = _HTML_PRE.sub(" ", cleaned)
    cleaned = _HTML_CODE.sub(" ", cleaned)
    cleaned = _INLINE_CODE.sub(" ", cleaned)
    cleaned = _HTML_TAG.sub(" ", cleaned)
    cleaned = _URL.sub(_shorten_url, cleaned)
    cleaned = cleaned.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    cleaned = _WHITESPACE.sub(" ", cleaned)
    cleaned = _MULTI_NEWLINE.sub("\n\n", cleaned)
    cleaned = cleaned.strip()
    if len(cleaned) > _MAX_TTS_CHARS:
        cleaned = cleaned[:_MAX_TTS_CHARS].rsplit(" ", 1)[0] + " …"
    return cleaned


# ---- Supertonic wrapper ---------------------------------------------------

# Supertonic-3 ships ten built-in voices: M1–M5 (male) and F1–F5 (female).
# These are the only valid ``voice_name`` values for ``get_voice_style``.
BUILTIN_MALE_VOICES: tuple[str, ...] = ("M1", "M2", "M3", "M4", "M5")
BUILTIN_FEMALE_VOICES: tuple[str, ...] = ("F1", "F2", "F3", "F4", "F5")
BUILTIN_VOICES: tuple[str, ...] = BUILTIN_MALE_VOICES + BUILTIN_FEMALE_VOICES

_DEFAULT_VOICE = "M1"
_DEFAULT_LANG = "ru"
_DEFAULT_STEPS = 8
_DEFAULT_SPEED = 1.05

_tts_lock = asyncio.Lock()
_tts_obj: Any | None = None
# Cached voice_style objects per (voice_name | path) so switching voices
# at runtime doesn't pay the build cost every time.
_voice_style_cache: dict[str, Any] = {}
_tts_unavailable_reason: str | None = None


def _read_cgroup_int(*paths: str) -> int | None:
    """Read the first existing cgroup file as int. Handles 'max'."""
    for path in paths:
        try:
            with open(path, "r", encoding="ascii") as f:
                raw = f.read().strip()
        except OSError:
            continue
        if raw == "max" or not raw:
            return None
        try:
            return int(raw)
        except ValueError:
            continue
    return None


def _available_ram_mb() -> int | None:
    """Best-effort read of available memory in MB.

    In a container the kernel's ``/proc/meminfo`` reports the *host*
    figures, which are useless when the container is cgroup-capped
    (Render Free → 512 MB host-wide). So we cross-check with the
    cgroup memory accounting:

    * ``memory.max``        — the limit (cgroup v2)
    * ``memory.current``    — current usage (cgroup v2)
    * ``memory.limit_in_bytes`` / ``memory.usage_in_bytes`` — v1

    When both are readable, we return ``min(meminfo, cgroup_limit -
    cgroup_current)``. Falls back to ``MemAvailable`` from
    ``/proc/meminfo`` outside containers / on non-Linux.
    """
    meminfo_mb: int | None = None
    try:
        with open("/proc/meminfo", "r", encoding="ascii") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        meminfo_mb = int(parts[1]) // 1024
                        break
    except (OSError, ValueError):
        meminfo_mb = None

    limit = _read_cgroup_int(
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    )
    current = _read_cgroup_int(
        "/sys/fs/cgroup/memory.current",
        "/sys/fs/cgroup/memory/memory.usage_in_bytes",
    )
    # cgroup v1 sometimes uses a sentinel like 9223372036854771712
    # (~8 EiB) to mean "unlimited"; treat anything that wouldn't fit in
    # a typical host as "unlimited".
    if limit is not None and limit > (1 << 50):
        limit = None
    if limit is not None and current is not None:
        cgroup_mb = max(0, (limit - current) // (1024 * 1024))
        if meminfo_mb is None:
            return cgroup_mb
        return min(meminfo_mb, cgroup_mb)
    return meminfo_mb


def _ram_blocks_tts() -> str | None:
    """Return a human-readable reason when current RAM is too low for TTS."""
    avail = _available_ram_mb()
    if avail is None:
        return None
    if avail < _TTS_MIN_AVAILABLE_MB:
        return (
            f"для озвучки нужно ~{_TTS_MIN_AVAILABLE_MB} MB свободной RAM, "
            f"сейчас доступно {avail} MB. На Render Free/Starter (0.5 GB) "
            f"\u00ab\U0001f50a Голосовой ответчик\u00bb не работает — нужен "
            f"тариф от Standard (2 GB) и выше."
        )
    return None


def _load_supertonic_sync() -> Any | None:
    """Load Supertonic in this thread. Returns the TTS object or None."""
    global _tts_unavailable_reason
    try:
        from supertonic import TTS
    except Exception as exc:  # noqa: BLE001 — surface to caller
        _tts_unavailable_reason = f"supertonic не установлен: {exc}"
        logger.warning("supertonic import failed: %s", exc)
        return None
    try:
        tts = TTS(auto_download=True)
    except Exception as exc:  # noqa: BLE001
        _tts_unavailable_reason = f"supertonic не загрузился: {exc}"
        logger.warning("supertonic init failed: %s", exc)
        return None
    return tts


async def _ensure_loaded() -> Any | None:
    """Lazy-load TTS, cached for the lifetime of the process."""
    global _tts_obj
    if _tts_obj is not None:
        return _tts_obj
    async with _tts_lock:
        if _tts_obj is not None:
            return _tts_obj
        result = await asyncio.to_thread(_load_supertonic_sync)
        if result is None:
            return None
        _tts_obj = result
        logger.info("supertonic loaded")
        return result


def _resolve_voice_style_sync(
    tts_obj: Any, voice: str, custom_path: str | None
) -> Any | None:
    """Return a Supertonic voice_style for the requested voice/path.

    Priority: a custom Voice-Builder JSON path (when supplied) wins
    over a built-in voice name. Cached so flipping voices in the UI
    doesn't repeatedly re-parse the same JSON. Returns ``None`` when
    the underlying call fails so the caller can fall back to the
    default voice.
    """
    if custom_path:
        cache_key = f"path::{custom_path}"
        if cache_key in _voice_style_cache:
            return _voice_style_cache[cache_key]
        try:
            style = tts_obj.get_voice_style_from_path(custom_path)
        except Exception:  # noqa: BLE001
            logger.warning("failed to load custom voice JSON: %s", custom_path)
            return None
        _voice_style_cache[cache_key] = style
        return style
    name = (voice or _DEFAULT_VOICE).upper()
    if name not in BUILTIN_VOICES:
        name = _DEFAULT_VOICE
    if name in _voice_style_cache:
        return _voice_style_cache[name]
    try:
        style = tts_obj.get_voice_style(voice_name=name)
    except Exception:  # noqa: BLE001
        logger.warning("failed to build voice_style for %s", name)
        if name != _DEFAULT_VOICE:
            try:
                style = tts_obj.get_voice_style(voice_name=_DEFAULT_VOICE)
            except Exception:  # noqa: BLE001
                return None
        else:
            return None
    _voice_style_cache[name] = style
    return style


def get_unavailable_reason() -> str | None:
    """Return human-readable reason if TTS isn't usable (or None when fine).

    Checks (in order):
    1. Persistent reason recorded during a previous import/load failure.
    2. Live RAM check — if there isn't enough free RAM right now to
       safely load Supertonic, surface that to the settings UI so users
       on Render Free understand why voice replies aren't arriving.
    """
    if _tts_unavailable_reason:
        return _tts_unavailable_reason
    return _ram_blocks_tts()


def _synthesize_to_wav_sync(
    text: str, wav_path: Path, *, voice_style: Any
) -> bool:
    """Run Supertonic synchronously and write a WAV to disk. Returns success."""
    if _tts_obj is None or voice_style is None:
        return False
    try:
        wav, _ = _tts_obj.synthesize(
            text=text,
            lang=_DEFAULT_LANG,
            voice_style=voice_style,
            total_steps=_DEFAULT_STEPS,
            speed=_DEFAULT_SPEED,
        )
        _tts_obj.save_audio(wav, str(wav_path))
    except Exception:  # noqa: BLE001
        logger.exception("supertonic synthesise failed")
        return False
    return wav_path.exists() and wav_path.stat().st_size > 0


def _wav_to_ogg_sync(wav_path: Path, ogg_path: Path) -> bool:
    """ffmpeg WAV → OGG/Opus (Telegram voice-compatible)."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(wav_path),
        "-c:a", "libopus", "-b:a", "32k",
        "-ar", "48000", "-ac", "1",
        str(ogg_path),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30
        )
    except FileNotFoundError:
        logger.warning("ffmpeg not installed — cannot convert WAV to OGG")
        return False
    except subprocess.TimeoutExpired:
        logger.warning("ffmpeg WAV→OGG conversion timed out")
        return False
    if result.returncode != 0:
        logger.warning("ffmpeg failed: %s", result.stderr.strip()[:200])
        return False
    return ogg_path.exists() and ogg_path.stat().st_size > 0


def _selected_voice() -> tuple[str, str | None]:
    """Read the user's currently selected voice + optional custom path.

    Local import of storage to avoid the ``bot.tts`` → ``bot.storage`` →
    ``bot.tts`` cycle at module-load time (tests import tts directly).
    """
    from .storage import storage

    voice = storage.get_tts_voice() or _DEFAULT_VOICE
    custom = storage.get_tts_custom_voice_path() or None
    return voice, custom


async def _synthesize_elevenlabs(spoken: str) -> bytes | None:
    """Synthesise via ElevenLabs HTTP API. Returns OGG/Opus bytes or None.

    Telegram accepts OGG/Opus for ``answer_voice``. ElevenLabs supports
    ``output_format=opus_48000_*`` which produces a raw Opus stream
    Telegram won't play as a voice note. We ask for ``ulaw_8000`` or
    ``mp3_44100_32`` (mp3) and remux to OGG/Opus with ffmpeg — the same
    helper Supertonic already uses for its WAV output.
    """
    from .storage import storage

    api_key = storage.get_elevenlabs_api_key()
    voice_id = storage.get_elevenlabs_voice_id()
    if not api_key or not voice_id:
        logger.info(
            "elevenlabs TTS skipped: missing %s%s%s",
            "api key" if not api_key else "",
            " and " if not api_key and not voice_id else "",
            "voice_id" if not voice_id else "",
        )
        return None
    base = storage.ELEVENLABS_BASE_URL.rstrip("/")
    url = f"{base}/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": api_key,
        "accept": "audio/mpeg",
        "content-type": "application/json",
    }
    payload = {
        "text": spoken,
        "model_id": "eleven_multilingual_v2",
        "output_format": "mp3_44100_64",
    }
    try:
        import httpx
    except ImportError:
        logger.warning("elevenlabs TTS skipped: httpx not installed")
        return None
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
        if resp.status_code != 200:
            body = resp.text[:200] if resp.text else ""
            logger.warning(
                "elevenlabs TTS http %s: %s", resp.status_code, body
            )
            return None
        audio_bytes = resp.content
    except Exception as exc:  # noqa: BLE001 — keep the chat path alive
        logger.warning("elevenlabs TTS request failed: %s", exc)
        return None
    if not audio_bytes:
        return None
    # Remux mp3 → OGG/Opus so Telegram renders it as a voice note.
    with tempfile.TemporaryDirectory(prefix="el-tts-") as td:
        mp3_path = Path(td) / "in.mp3"
        ogg_path = Path(td) / "out.ogg"
        try:
            mp3_path.write_bytes(audio_bytes)
        except OSError as exc:
            logger.warning("could not write mp3 buffer: %s", exc)
            return None
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", str(mp3_path),
                    "-c:a", "libopus", "-b:a", "32k", "-application", "voip",
                    str(ogg_path),
                ],
                check=True, capture_output=True, timeout=30,
            )
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
            logger.warning("ffmpeg remux failed: %s", exc)
            return None
        try:
            return ogg_path.read_bytes()
        except OSError:
            return None


async def synthesize_voice_ogg(
    text: str, *, voice: str | None = None, custom_path: str | None = None
) -> bytes | None:
    """Sanitise + synthesise + convert. Returns OGG/Opus bytes or None.

    Returns None for any failure (empty text, model unavailable, ffmpeg
    missing, etc.) — caller must treat None as "no voice this time" and
    proceed without it. Errors are logged, never raised.

    ``voice`` / ``custom_path`` let callers («Прослушать» in the voice
    menu, tests) override the storage-backed defaults.

    Provider dispatch: ``storage.get_tts_provider()`` decides whether to
    call ElevenLabs (cloud) or Supertonic (local). On free Render with
    only 512 MB RAM, ElevenLabs is the recommended pick.
    """
    spoken = sanitize_for_tts(text)
    if not spoken:
        return None
    # Cloud provider path: no RAM check needed (synthesis is server-side).
    from .storage import storage as _storage_mod

    provider = _storage_mod.get_tts_provider()
    if provider == "elevenlabs":
        ogg = await _synthesize_elevenlabs(spoken)
        if ogg is not None:
            return ogg
        # Fall through to local synthesis below so a temporary ElevenLabs
        # outage doesn't silently strand the user without voice. The next
        # call will retry ElevenLabs first again.
        logger.info("elevenlabs synth failed — falling back to local Supertonic")
    # RAM guard. Supertonic peaks ~470 MB on first synthesis; on a 512 MB
    # box (Render Free / Starter) this OOM-kills the whole bot. Skip
    # gracefully when the host can't fit it instead of restart-looping.
    ram_block = _ram_blocks_tts()
    if ram_block:
        logger.warning("skipping TTS: %s", ram_block)
        return None
    tts_obj = await _ensure_loaded()
    if tts_obj is None:
        return None
    if voice is None and custom_path is None:
        voice, custom_path = _selected_voice()
    style = await asyncio.to_thread(
        _resolve_voice_style_sync, tts_obj, voice or _DEFAULT_VOICE, custom_path
    )
    if style is None:
        return None
    with tempfile.TemporaryDirectory(prefix="tts-") as td:
        wav_path = Path(td) / "out.wav"
        ogg_path = Path(td) / "out.ogg"
        ok = await asyncio.to_thread(
            _synthesize_to_wav_sync, spoken, wav_path, voice_style=style
        )
        if not ok:
            return None
        ok = await asyncio.to_thread(_wav_to_ogg_sync, wav_path, ogg_path)
        if not ok:
            return None
        try:
            return ogg_path.read_bytes()
        except OSError as exc:
            logger.warning("could not read produced OGG: %s", exc)
            return None


async def maybe_send_voice_reply(message: Any, text: str) -> None:
    """Best-effort wrapper: if TTS is enabled, send a voice reply.

    Pulls the user's preference from storage so callers don't have to.
    Never raises — the caller is in a hot path and shouldn't fail just
    because the model is sulking.
    """
    from .storage import storage

    if not storage.get_tts_enabled():
        return
    try:
        ogg = await synthesize_voice_ogg(text)
    except Exception:  # noqa: BLE001
        logger.exception("TTS pipeline raised — skipping voice reply")
        return
    if not ogg:
        return
    try:
        from aiogram.types import BufferedInputFile  # local import to avoid cycle
    except Exception:  # noqa: BLE001 — aiogram missing? bail quietly
        return
    voice = BufferedInputFile(ogg, filename="reply.ogg")
    with contextlib.suppress(Exception):
        await message.answer_voice(voice)
