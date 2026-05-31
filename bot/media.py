"""Image-to-video / image-to-image / background-removal via deAPI.ai
(or any OpenAPI-compatible provider stored in a ``media_slot``).

Storage model
-------------
The bot keeps up to 3 *media slots* in ``state.json``. Each slot is a
self-contained provider config::

    {
        "url": "https://api.deapi.ai",
        "api_key": "...",
        "name": "deapi",                   # auto-derived from URL
        "video_model": "Ltx2_19B_Dist_FP8",
        "photo_model": "Flux_2_Klein_4B_BF16",
        "rmbg_model":  "Ben2",
    }

A slot is *active for a task* when ``storage.set_active_media_slot(task,
slot_id)`` points at it. Tasks: ``"video"`` (img2video), ``"photo"``
(img2img), ``"rmbg"`` (background removal).

API shape
---------
All deAPI endpoints follow the same pattern:

  1. POST ``/api/v1/client/<op>`` (multipart) → ``{data: {request_id}}``.
  2. Poll GET ``/api/v1/client/request-status/{request_id}`` until
     ``status`` is ``done`` (returns ``result_url``) or ``error``.

Auth: ``Authorization: Bearer <api_key>``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)


# ---- provider catalogue --------------------------------------------------

# Known providers — used to auto-name a slot once its URL is filled and
# to surface model-slug hints in the wizard. Adding a new provider here
# is enough; per-provider request shapes are still handled by code below
# (currently only deAPI; fal/replicate are stubs documented in PR text).

KNOWN_MEDIA_PROVIDERS: dict[str, dict] = {
    "deapi": {
        "url": "https://api.deapi.ai",
        "label": "deAPI.ai",
        "match_url": "deapi.ai",
        "hint": (
            "Decentralized AI inference. $5 free signup. Endpoints:\n"
            "• img2video — LTX-Video\n"
            "• img2img — Qwen-Image-Edit / Flux-2-Klein\n"
            "• img-rmbg — Ben2"
        ),
        "video_models": [
            ("Ltxv_13B_0_9_8_Distilled_FP8", "LTX-Video 0.9.8 13B — fast/cheap, 512×512, 30 fps"),
            ("Ltx2_19B_Dist_FP8",            "LTX-2 19B Distilled — up to 1024×1024, 24 fps"),
            ("Ltx2_3_22B_Dist_INT8",         "LTX-2.3 22B Distilled — best quality, also audio2video"),
        ],
        "photo_models": [
            ("QwenImageEdit_Plus_NF4", "Qwen-Image-Edit Plus — best for prompted edits"),
            ("Flux_2_Klein_4B_BF16",  "Flux 2 Klein 4B — faster, smaller (txt2img+img2img)"),
        ],
        "rmbg_models": [
            ("Ben2", "Ben2 — boundary-aware background removal"),
        ],
    },
    "fal": {
        "url": "https://fal.run",
        "label": "fal.ai",
        "match_url": "fal.run",
        "hint": "fal.ai adapter is stubbed; needs adapter implementation (see media.py).",
        "video_models": [
            ("fal-ai/kling-video/v3/standard/image-to-video", "Kling v3 Std — premium quality"),
            ("fal-ai/wan/v2.2-a14b/image-to-video",           "Wan 2.2 — cheaper alternative"),
        ],
        "photo_models": [],
        "rmbg_models": [],
    },
    "replicate": {
        "url": "https://api.replicate.com",
        "label": "Replicate",
        "match_url": "replicate.com",
        "hint": "Replicate adapter is stubbed; needs adapter implementation.",
        "video_models": [],
        "photo_models": [],
        "rmbg_models": [],
    },
}


def detect_provider_name(url: str) -> str:
    """Auto-derive a slot name from its URL. Falls back to ''."""
    if not url:
        return ""
    url_l = url.lower()
    for key, meta in KNOWN_MEDIA_PROVIDERS.items():
        if meta["match_url"] in url_l:
            return key
    return ""


# ---- defaults / quality mapping -----------------------------------------

# LTXv-0.9.8 is single-step (must use steps=1, fps=30). LTX-2 (19B/22B)
# wants multi-step and 24 fps. Anything outside the model's range will
# be 422'd by deAPI, so callers should pick a quality-friendly model
# (i.e. send 480p to LTXv-0.9.8, 1080p to LTX-2).
VIDEO_QUALITY_MAP: dict[str, tuple[int, int, int, int, int]] = {
    # quality -> (width, height, frames, fps, steps)
    "480":  (512, 512, 60, 30, 1),    # LTXv-0.9.8 friendly
    "720":  (768, 768, 60, 30, 1),    # LTXv-0.9.8 friendly
    "1080": (1024, 1024, 120, 24, 8), # LTX-2 friendly (max 1024)
    "2k":   (1024, 1024, 120, 24, 8), # cap at model max — 2K is aspirational
    "auto": (512, 512, 60, 30, 1),    # safe default
}


# ---- data types ----------------------------------------------------------


@dataclass(frozen=True)
class MediaSlot:
    slot_id: str  # "1" | "2" | "3"
    url: str
    api_key: str
    name: str
    video_model: str
    photo_model: str
    rmbg_model: str

    @property
    def configured(self) -> bool:
        return bool(self.url and self.api_key)


class MediaError(RuntimeError):
    pass


ProgressCb = Callable[[str, float], Awaitable[None]]


# ---- slot loader ---------------------------------------------------------


def _load_slot(task: str) -> MediaSlot:
    """Pull the active slot for ``task`` (``"video"``/``"photo"``/``"rmbg"``).

    Falls back to env vars when no slot is configured (back-compat with
    the simpler initial integration).
    """
    from .storage import storage  # local import — avoids circular load

    slot = storage.get_active_media_slot(task)
    if slot is None:
        # Back-compat: synthesise a slot from env so the existing
        # DEAPI_API_KEY-only deploy keeps working.
        env_key = os.environ.get("DEAPI_API_KEY") or os.environ.get("DEAPI_KEY")
        if not env_key:
            raise MediaError(
                "Media-провайдер не настроен. /setup → 🎬 Мозг для видео и фото"
            )
        defaults = KNOWN_MEDIA_PROVIDERS["deapi"]
        return MediaSlot(
            slot_id="env",
            url=defaults["url"],
            api_key=env_key,
            name="deapi",
            video_model="Ltxv_13B_0_9_8_Distilled_FP8",
            photo_model="QwenImageEdit_Plus_NF4",
            rmbg_model="Ben2",
        )
    if not slot.configured:
        raise MediaError(
            f"Media-слот {slot.slot_id} не настроен (нет URL или API key). "
            "Открой /setup → 🎬 Мозг для видео и фото и дозаполни."
        )
    return slot


# ---- deAPI calls ---------------------------------------------------------


async def _submit_and_poll(
    slot: MediaSlot,
    op: str,
    data: dict[str, str],
    files: dict,
    *,
    on_progress: ProgressCb | None = None,
    poll_timeout: float = 600.0,
    poll_interval: float = 4.0,
) -> dict:
    """Submit a deAPI request and poll until ``status`` is ``done``.

    Returns the final ``data`` payload from /request-status (i.e. the
    one containing ``result_url`` and ``results_alt_formats``).
    """
    if "deapi.ai" not in slot.url.lower():
        # Stub for non-deAPI providers — they have different shapes.
        raise MediaError(
            f"Provider {slot.name!r} (URL={slot.url}) не поддерживается "
            "в этой сборке. Сейчас работает только deAPI."
        )
    headers = {
        "Authorization": f"Bearer {slot.api_key}",
        "Accept": "application/json",
    }
    submit_url = f"{slot.url.rstrip('/')}/api/v1/client/{op}"
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(submit_url, headers=headers, data=data, files=files)
        if resp.status_code >= 400:
            raise MediaError(f"{op} submit HTTP {resp.status_code}: {resp.text[:500]}")
        try:
            request_id = resp.json()["data"]["request_id"]
        except Exception as exc:  # noqa: BLE001
            raise MediaError(f"{op}: unexpected submit response {resp.text[:500]}") from exc
        logger.info("media: %s submitted id=%s slot=%s", op, request_id, slot.slot_id)
        return await _poll(client, slot, request_id, headers, op,
                           on_progress=on_progress,
                           poll_timeout=poll_timeout,
                           poll_interval=poll_interval)


async def _poll(
    client: httpx.AsyncClient,
    slot: MediaSlot,
    request_id: str,
    headers: dict[str, str],
    op: str,
    *,
    on_progress: ProgressCb | None,
    poll_timeout: float,
    poll_interval: float,
) -> dict:
    deadline = asyncio.get_event_loop().time() + poll_timeout
    last_reported = -10.0
    while asyncio.get_event_loop().time() < deadline:
        resp = await client.get(
            f"{slot.url.rstrip('/')}/api/v1/client/request-status/{request_id}",
            headers=headers,
        )
        if resp.status_code >= 400:
            raise MediaError(f"{op} status HTTP {resp.status_code}: {resp.text[:500]}")
        payload = resp.json().get("data") or {}
        status = payload.get("status") or "processing"
        progress = float(payload.get("progress") or 0.0)
        if status == "done":
            return payload
        if status == "error":
            raise MediaError(f"{op} failed: {payload!r}")
        if on_progress is not None and (progress - last_reported >= 5.0 or status != "processing"):
            last_reported = progress
            try:
                await on_progress(status, progress)
            except Exception:  # noqa: BLE001
                logger.debug("on_progress failed", exc_info=True)
        await asyncio.sleep(poll_interval)
    raise MediaError(f"{op} timeout after {poll_timeout}s (request_id={request_id})")


# ---- public API ----------------------------------------------------------


async def image_to_video(
    image_bytes: bytes,
    prompt: str,
    *,
    quality: str = "auto",
    negative_prompt: str | None = None,
    on_progress: ProgressCb | None = None,
) -> str:
    """Animate ``image_bytes`` with ``prompt`` at ``quality``. Returns MP4 URL."""
    slot = _load_slot("video")
    w, h, frames, fps, steps = VIDEO_QUALITY_MAP.get(quality, VIDEO_QUALITY_MAP["auto"])
    seed = random.randint(1, 2**31 - 1)
    data = {
        "prompt": prompt or "Cinematic gentle motion, soft natural light",
        "model": slot.video_model,
        "width": str(w),
        "height": str(h),
        "frames": str(frames),
        "fps": str(fps),
        "steps": str(steps),
        "guidance": "7.5",
        "seed": str(seed),
    }
    if negative_prompt:
        data["negative_prompt"] = negative_prompt
    files = {"first_frame_image": ("input.jpg", image_bytes, "image/jpeg")}
    payload = await _submit_and_poll(
        slot, "img2video", data, files, on_progress=on_progress,
    )
    url = payload.get("result_url")
    if not url:
        raise MediaError(f"img2video done but no result_url: {payload!r}")
    return url


async def image_to_image(
    image_bytes: bytes,
    prompt: str,
    *,
    fmt: str = "png",
    on_progress: ProgressCb | None = None,
) -> tuple[str, str]:
    """Transform ``image_bytes`` per ``prompt``.

    Returns ``(png_url, jpg_url_or_empty)`` — both URLs from one
    submission. The PNG is the native output (uncompressed, can be sent
    as document); the JPG is deAPI's auto-converted alt format suitable
    for inline ``answer_photo`` preview.

    ``fmt`` is currently accepted for back-compat but ignored: we always
    request the native PNG and let the caller decide what to deliver.
    Setting ``fmt="jpeg"`` swaps which URL goes first, but both are
    always returned.
    """
    slot = _load_slot("photo")
    seed = random.randint(1, 2**31 - 1)
    data = {
        "prompt": prompt,
        "model": slot.photo_model,
        "steps": "20",
        "seed": str(seed),
        "guidance": "7.5",
    }
    files = {"image": ("input.jpg", image_bytes, "image/jpeg")}
    payload = await _submit_and_poll(
        slot, "img2img", data, files, on_progress=on_progress,
    )
    png_url = payload.get("result_url") or ""
    jpg_url = (payload.get("results_alt_formats") or {}).get("jpg") or ""
    if not png_url and not jpg_url:
        raise MediaError(f"img2img done but no result_url: {payload!r}")
    fmt_l = (fmt or "png").lower()
    if fmt_l in ("jpeg", "jpg") and jpg_url:
        # Caller prefers JPEG as the "primary"; still return PNG too.
        return jpg_url, png_url
    return png_url, jpg_url


async def remove_background(
    image_bytes: bytes,
    *,
    on_progress: ProgressCb | None = None,
) -> tuple[str, str]:
    """Remove background from ``image_bytes``.

    Returns ``(png_url, jpg_url_or_empty)`` — the PNG keeps transparency
    (Photoshop-ready), the JPG is a flat preview for Telegram inline.
    """
    slot = _load_slot("rmbg")
    data = {"model": slot.rmbg_model}
    files = {"image": ("input.jpg", image_bytes, "image/jpeg")}
    payload = await _submit_and_poll(
        slot, "img-rmbg", data, files, on_progress=on_progress,
    )
    png_url = payload.get("result_url")
    if not png_url:
        raise MediaError(f"img-rmbg done but no result_url: {payload!r}")
    jpg_url = (payload.get("results_alt_formats") or {}).get("jpg", "")
    return png_url, jpg_url


# ---- diagnostics ---------------------------------------------------------


async def get_balance(slot: MediaSlot | None = None) -> float:
    """Return current credit balance for the slot (defaults to video slot)."""
    if slot is None:
        slot = _load_slot("video")
    if "deapi.ai" not in slot.url.lower():
        raise MediaError(f"balance check supports only deAPI (got {slot.url})")
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            f"{slot.url.rstrip('/')}/api/v1/client/balance",
            headers={
                "Authorization": f"Bearer {slot.api_key}",
                "Accept": "application/json",
            },
        )
        if resp.status_code >= 400:
            raise MediaError(f"balance HTTP {resp.status_code}: {resp.text[:200]}")
        return float(resp.json().get("data", {}).get("balance", 0.0))


async def list_models(slot: MediaSlot | None = None) -> list[dict]:
    if slot is None:
        slot = _load_slot("video")
    if "deapi.ai" not in slot.url.lower():
        raise MediaError(f"list_models supports only deAPI (got {slot.url})")
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"{slot.url.rstrip('/')}/api/v1/client/models",
            headers={
                "Authorization": f"Bearer {slot.api_key}",
                "Accept": "application/json",
            },
        )
        if resp.status_code >= 400:
            raise MediaError(f"models HTTP {resp.status_code}: {resp.text[:200]}")
        return resp.json().get("data") or []
