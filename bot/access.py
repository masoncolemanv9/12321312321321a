"""Centralised "who is allowed to do what" helpers.

The bot has three access modes (see ``Storage.get_access_mode``):

* ``private`` — only the owner + explicit co-owners can use the bot at
  all and change settings (this is the original lilush behaviour).
* ``public`` — anyone can USE the bot's features (chat, photo flows,
  Helpzavr, pretty-text, mailbox, media-generation). Settings —
  API keys, models, statistics, persona switching — stay owner-only.
* ``full_public`` — anyone can use AND change settings (no gating).

``can_use`` is the broad gate; ``can_admin`` is the strict gate. UI
code chooses which one to call depending on whether the action is a
feature or a settings change.

Co-owners (see :func:`Storage.add_co_owner`) are tracked in storage and
always count as admins regardless of the current access mode.
"""

from __future__ import annotations


def _storage():
    """Resolve the current :class:`Storage` singleton lazily.

    Tests sometimes pop ``bot.storage`` from ``sys.modules`` and
    re-import it to get a fresh data directory. Calling :func:`_storage`
    every time means we always read the most-recent module-level
    singleton instead of caching a stale reference from import time.
    """
    from .storage import storage

    return storage


def can_use(user_id: int | None) -> bool:
    """``True`` when ``user_id`` is allowed to use the bot's features."""
    return _storage().can_use_bot(user_id)


def can_admin(user_id: int | None) -> bool:
    """``True`` when ``user_id`` is allowed to change bot settings.

    Used to gate API-key entry, model selection, URL changes, persona
    switching, token statistics, the access-mode picker itself, and the
    co-owner list.
    """
    return _storage().can_admin_bot(user_id)


def is_owner(user_id: int | None) -> bool:
    """``True`` only for the single original owner (not co-owners).

    Used by the ``/start`` ownership-claim flow, where co-owners must
    not be able to "re-claim" the slot. Most other callers want
    :func:`can_admin` instead.
    """
    if user_id is None:
        return False
    try:
        uid = int(user_id)
    except (TypeError, ValueError):
        return False
    return _storage().get_owner_id() == uid
