"""Slot read/write helpers for the algorithm addon.

Storage layout under ``storage._settings()`` (via ``bot.addons.state``):

::

    addons:
      algorithm:
        by_chat:
          <chat_id>:
            slot:
              "1":
                name: "гугл"
                plan: "1. ...\n2. ..."
                interval_minutes: 0.5
                last_run_at: 1726512345.6
                is_running: false
              "2": {...}
              ...
              "10": {...}

The schema is intentionally per-chat — mirroring how
``bot.addons.mailbox.handlers`` scopes its ``awaiting_field`` — so a
guest chat in public mode cannot read or overwrite the owner's saved
algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from .. import state as addon_state

ADDON = "algorithm"
SLOT_COUNT = 10


@dataclass
class Slot:
    """One saved algorithm slot. ``index`` is 1..10."""

    index: int
    name: str = ""
    plan: str = ""
    interval_minutes: float = 0.0
    last_run_at: float = 0.0
    is_running: bool = False

    @property
    def is_empty(self) -> bool:
        return not self.plan.strip()

    @property
    def has_interval(self) -> bool:
        return self.interval_minutes > 0

    def label(self) -> str:
        """Button label for the slot list — number prefix + dynamic name."""
        if self.is_empty:
            return f"Слот {self.index}"
        # Always prefix with the index so duplicates ("гугл" / "гугл 2")
        # plus their position are both visible at a glance.
        return f"{self.index}. {self.name or 'без имени'}"


def _slots_root(chat_id: int) -> dict:
    """Return the mutable ``slot`` sub-dict for one chat, creating
    intermediate keys as needed."""
    by_chat = addon_state._root().setdefault(ADDON, {}).setdefault("by_chat", {})  # noqa: SLF001
    chat_bucket = by_chat.setdefault(str(chat_id), {})
    return chat_bucket.setdefault("slot", {})


def _coerce_slot(index: int, raw: dict[str, Any] | None) -> Slot:
    if not raw:
        return Slot(index=index)
    return Slot(
        index=index,
        name=str(raw.get("name", "") or ""),
        plan=str(raw.get("plan", "") or ""),
        interval_minutes=float(raw.get("interval_minutes", 0) or 0),
        last_run_at=float(raw.get("last_run_at", 0) or 0),
        is_running=bool(raw.get("is_running", False)),
    )


def get_slot(chat_id: int, index: int) -> Slot:
    if not (1 <= index <= SLOT_COUNT):
        raise ValueError(f"slot index out of range: {index}")
    raw = _slots_root(chat_id).get(str(index))
    return _coerce_slot(index, raw)


def list_slots(chat_id: int) -> list[Slot]:
    """All 10 slots in order, empty placeholders included."""
    bucket = _slots_root(chat_id)
    return [_coerce_slot(i, bucket.get(str(i))) for i in range(1, SLOT_COUNT + 1)]


def list_chat_ids() -> list[int]:
    """Every chat that has at least one stored slot — used by the
    scheduler to know which chats to poll."""
    by_chat = addon_state._root().get(ADDON, {}).get("by_chat", {})  # noqa: SLF001
    out: list[int] = []
    for cid in by_chat:
        try:
            out.append(int(cid))
        except (TypeError, ValueError):
            continue
    return out


def save_slot(chat_id: int, slot: Slot) -> None:
    """Overwrite the slot at ``slot.index`` for ``chat_id``."""
    if not (1 <= slot.index <= SLOT_COUNT):
        raise ValueError(f"slot index out of range: {slot.index}")
    _slots_root(chat_id)[str(slot.index)] = {
        k: v for k, v in asdict(slot).items() if k != "index"
    }
    from ...storage import storage as _st

    _st._save()  # noqa: SLF001


def clear_slot(chat_id: int, index: int) -> None:
    """Wipe a slot back to the empty placeholder."""
    bucket = _slots_root(chat_id)
    bucket.pop(str(index), None)
    from ...storage import storage as _st

    _st._save()  # noqa: SLF001


def derive_unique_name(chat_id: int, raw_name: str, *, exclude_index: int | None = None) -> str:
    """Pick a slot name that doesn't collide with another non-empty
    slot in the same chat.

    Args:
        chat_id: chat scope.
        raw_name: the base name the planner / user proposed
            (lowercased, 1-3 short words). Empty string → ``"без имени"``.
        exclude_index: if set, the slot at this index is ignored when
            checking collisions (so renaming an existing slot to its
            current name doesn't bump it to ``" 2"``).

    Returns:
        ``raw_name`` if unique, else ``f"{raw_name} {n}"`` with
        the smallest ``n >= 2`` that disambiguates.
    """
    base = (raw_name or "").strip().lower()
    if not base:
        base = "без имени"
    existing = {
        s.name.lower()
        for s in list_slots(chat_id)
        if not s.is_empty and (exclude_index is None or s.index != exclude_index)
    }
    if base not in existing:
        return base
    n = 2
    while f"{base} {n}" in existing:
        n += 1
    return f"{base} {n}"


# ---- transient per-chat UI state ----------------------------------------
#
# When the user is mid-flow (typing a plan, picking an interval,
# choosing a slot for AI planning, etc.) we need to remember which slot
# they're editing without leaning on aiogram FSM — handlers below DO
# also set an FSM state so that ``pretty_text``'s catch-all
# ``StateFilter(None)`` correctly bails out, but the FSM payload is
# opaque from outside the wizard and tests want to peek.
#
# The transient flag lives at addons.algorithm.by_chat.<id>.editing.{slot,mode}.


def set_editing(chat_id: int, slot_index: int | None, mode: str | None) -> None:
    by_chat = addon_state._root().setdefault(ADDON, {}).setdefault("by_chat", {})  # noqa: SLF001
    bucket = by_chat.setdefault(str(chat_id), {})
    if slot_index is None and mode is None:
        bucket.pop("editing", None)
    else:
        bucket["editing"] = {"slot": slot_index, "mode": mode}
    from ...storage import storage as _st

    _st._save()  # noqa: SLF001


def get_editing(chat_id: int) -> tuple[int | None, str | None]:
    by_chat = addon_state._root().get(ADDON, {}).get("by_chat", {})  # noqa: SLF001
    bucket = by_chat.get(str(chat_id), {})
    edit = bucket.get("editing") or {}
    return edit.get("slot"), edit.get("mode")
