"""Shared addon-settings helper — piggybacks on lilush's storage.

Stores addon flags & secrets under ``storage._settings()["addons"]`` so
the existing ``state.json`` file keeps everything (no extra files for
ops to remember to back up on Render Free).

API kept tiny on purpose; each addon does its own reads/writes through
these primitives instead of editing lilush.storage directly.
"""

from __future__ import annotations

from typing import Any

from ..storage import storage


def _root() -> dict:
    """Return the mutable ``addons`` sub-dict, creating it lazily."""
    settings = storage._settings()  # noqa: SLF001 — intentional helper
    return settings.setdefault("addons", {})


def get(addon: str, key: str, default: Any = None) -> Any:
    bucket = _root().setdefault(addon, {})
    return bucket.get(key, default)


def set_(addon: str, key: str, value: Any) -> None:
    bucket = _root().setdefault(addon, {})
    bucket[key] = value
    storage._save()  # noqa: SLF001


def delete(addon: str, key: str) -> bool:
    bucket = _root().get(addon, {})
    if key in bucket:
        del bucket[key]
        storage._save()  # noqa: SLF001
        return True
    return False


def chat_get(addon: str, chat_id: int, key: str, default: Any = None) -> Any:
    """Per-chat-scoped value (e.g. pretty-text toggle is per chat)."""
    bucket = _root().setdefault(addon, {}).setdefault("by_chat", {})
    return bucket.setdefault(str(chat_id), {}).get(key, default)


def chat_set(addon: str, chat_id: int, key: str, value: Any) -> None:
    bucket = _root().setdefault(addon, {}).setdefault("by_chat", {})
    bucket.setdefault(str(chat_id), {})[key] = value
    storage._save()  # noqa: SLF001


def role_is_chosen() -> bool:
    """True iff the user has explicitly picked a role via /role.

    Used to gate project / clone / exec prompts until the bot has been
    given an identity — that's what the user asked for: 'don't tell me
    to choose or create a project until I've picked a role'.
    """
    return storage.get_persona_override() is not None
