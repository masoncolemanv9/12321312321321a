"""Groq-specific helpers wired into the dual-brain flow.

Two features live here:

* **Voice → text (Whisper).** When the user records a voice or sends an
  audio file from the main menu (outside any FSM state), we feed it to
  Groq's Whisper endpoint and pipe the transcript back into the normal
  chat handler.
* **Photo + caption → Vision.** When the user sends a photo with a
  caption from the main menu, we feed both to a Groq vision model and
  return the answer.

Both flows are *overrides* on top of the regular chat: they only kick
in when Brain 2 is configured against ``api.groq.com`` AND Brain 1 is
NOT the «Другое» (custom-URL) endpoint. The latter is the user's
contract — when «Другое» is set, it has top priority and Groq must
not interfere.

If the Groq Whisper/Vision call fails for any reason (network, bad
key, unsupported model), the override gracefully bows out: the audio
or photo falls back to whatever handler would have caught it in the
absence of these helpers (transcript: a "не получилось" note; photo:
the photo chooser).
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass

from openai import AsyncOpenAI

from .storage import storage

logger = logging.getLogger(__name__)


# Default models. The user can override per-slot via the brain config UI;
# these are what we try first if the slot's model is empty or doesn't
# look like an audio/vision model.
DEFAULT_GROQ_WHISPER_MODEL = "whisper-large-v3"
DEFAULT_GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# Substrings we use to spot a Groq endpoint in a slot's ``base_url``.
_GROQ_URL_MARKERS = ("groq.com",)


@dataclass(frozen=True)
class GroqSlotCfg:
    """Resolved Groq credentials for one of the brain slots."""

    slot: str  # "1" or "2"
    api_key: str
    base_url: str  # always non-empty when this dataclass is returned
    model: str  # slot's configured model (may be empty for whisper/vision)


def _slot_cfg(slot: str) -> tuple[str, str, str, str]:
    """Mirror of :func:`bot.agent._slot_cfg` to avoid a cycle.

    Returns ``(provider, api_key, base_url, model)`` for slot 1 (legacy
    top-level fields) or slot 2 (``brain_slot2`` nested dict).
    """
    if slot == "2":
        s = storage.get_brain_slot2()
        return (
            s.get("provider") or "custom",
            s.get("api_key") or "",
            s.get("base_url") or "",
            s.get("model") or "",
        )
    return (
        storage.get_provider() or "openrouter",
        storage.get_provider_key("openrouter") or "",
        storage.get_base_url() or "",
        storage.get_model() or "",
    )


def _is_groq_url(url: str) -> bool:
    """True when ``url`` points at a Groq-compatible endpoint."""
    u = (url or "").lower()
    return any(marker in u for marker in _GROQ_URL_MARKERS)


def find_groq_slot() -> GroqSlotCfg | None:
    """Return whichever brain slot is wired to Groq, or ``None``.

    Slot 2 wins ties — that's the slot dedicated to the Groq override in
    the UX. The slot must have both an api_key and a base_url pointing
    at groq.com, otherwise it isn't usable.
    """
    for slot in ("2", "1"):
        provider, api_key, base_url, model = _slot_cfg(slot)
        if not api_key:
            continue
        if not _is_groq_url(base_url):
            continue
        return GroqSlotCfg(slot=slot, api_key=api_key, base_url=base_url, model=model)
    return None


def is_other_brain_active() -> bool:
    """True when slot 1 is the «Другое» custom endpoint (has top priority).

    The contract: when «Другое» is configured, Brain 2 / Groq overrides
    must not interfere with chat. This is detected from slot 1's
    provider=custom + a non-empty base_url.
    """
    provider, api_key, base_url, _ = _slot_cfg("1")
    return provider == "custom" and bool(api_key) and bool(base_url)


def should_use_groq_override() -> bool:
    """Gate that decides whether the Groq voice/vision hook fires.

    True when *all* of the following hold:

    * A brain slot is wired to Groq with an api_key.
    * «Другое» (slot 1 custom + base_url) is NOT active. The user
      explicitly asked that «Другое» wins over Brain 2 overrides.
    """
    if is_other_brain_active():
        return False
    return find_groq_slot() is not None


def _build_groq_client(slot_cfg: GroqSlotCfg) -> AsyncOpenAI:
    """Build an OpenAI-compatible client pointed at Groq for this slot."""
    return AsyncOpenAI(api_key=slot_cfg.api_key, base_url=slot_cfg.base_url)


async def transcribe_audio_with_groq(audio_path: str) -> str:
    """Run ``audio_path`` through Groq Whisper and return the transcript.

    Raises ``RuntimeError`` when no Groq slot is configured. Any
    network/API error from Groq is propagated unchanged — the caller
    decides what to show the user.
    """
    slot_cfg = find_groq_slot()
    if slot_cfg is None:
        raise RuntimeError("Groq slot is not configured")

    model = slot_cfg.model if "whisper" in slot_cfg.model.lower() else DEFAULT_GROQ_WHISPER_MODEL
    client = _build_groq_client(slot_cfg)
    with open(audio_path, "rb") as fh:
        # The OpenAI SDK accepts a file-like object here; it'll send the
        # right multipart/form-data shape to Groq.
        resp = await client.audio.transcriptions.create(
            file=fh,
            model=model,
        )
    text = getattr(resp, "text", None)
    if not text:
        # Some endpoints return ``{"text": "..."}`` shape rather than the
        # SDK's typed object — extract defensively.
        text = str(resp).strip()
    return (text or "").strip()


async def describe_image_with_groq(image_bytes: bytes, caption: str) -> str:
    """Send ``image_bytes`` + ``caption`` to a Groq vision model.

    The image is base64-encoded inline (no need to upload it
    anywhere). We use whatever vision-capable model the slot has
    configured, or fall back to ``DEFAULT_GROQ_VISION_MODEL`` if the
    slot's model doesn't look like a vision model.
    """
    slot_cfg = find_groq_slot()
    if slot_cfg is None:
        raise RuntimeError("Groq slot is not configured")

    model = slot_cfg.model
    # Pick a vision-capable model. Heuristic: model name contains
    # "vision" or "scout" (llama-4 scout is vision) or "llama-3.2"
    # which on Groq is the vision family.
    looks_visiony = any(
        marker in model.lower()
        for marker in ("vision", "scout", "llama-3.2", "llava")
    )
    if not looks_visiony:
        model = DEFAULT_GROQ_VISION_MODEL

    encoded = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:image/jpeg;base64,{encoded}"
    user_text = caption.strip() or "Опиши, что на изображении, и ответь по-русски."
    client = _build_groq_client(slot_cfg)
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        temperature=0.3,
        max_tokens=1024,
    )
    choices = getattr(resp, "choices", None)
    if not choices:
        return "(Groq не вернул choices)"
    msg = choices[0].message
    return (getattr(msg, "content", None) or "").strip() or "(пустой ответ Groq)"
