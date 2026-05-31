"""Tests for the access-mode + co-owner machinery.

Three modes (``private`` / ``public`` / ``full_public``) plus an
explicit co-owner list control who can use vs. who can admin the bot.
This file exercises the storage primitives and the central access
helpers so future refactors don't regress.
"""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture()
def fresh_modules(tmp_path, monkeypatch):
    """A clean Storage + access view bound to a tmp data dir."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BOT_PERSONA", "boss")
    import sys

    for mod in ("bot.config", "bot.storage", "bot.persona", "bot.access"):
        sys.modules.pop(mod, None)
    config = importlib.import_module("bot.config")
    storage_mod = importlib.import_module("bot.storage")
    storage_mod.storage = storage_mod.Storage(data_dir=config.DATA_DIR)
    access = importlib.import_module("bot.access")
    return storage_mod.storage, access


def test_default_mode_is_private(fresh_modules) -> None:
    storage, _ = fresh_modules
    assert storage.get_access_mode() == "private"
    assert storage.get_co_owners() == []


def test_set_and_get_access_mode(fresh_modules) -> None:
    storage, _ = fresh_modules
    storage.set_access_mode("public")
    assert storage.get_access_mode() == "public"
    storage.set_access_mode("full_public")
    assert storage.get_access_mode() == "full_public"
    storage.set_access_mode("private")
    assert storage.get_access_mode() == "private"


def test_set_access_mode_rejects_unknown(fresh_modules) -> None:
    storage, _ = fresh_modules
    with pytest.raises(ValueError):
        storage.set_access_mode("god_mode")


def test_get_access_mode_falls_back_to_private_on_bad_value(fresh_modules) -> None:
    storage, _ = fresh_modules
    # Inject a garbage value bypassing the setter — getter must coerce.
    storage._state.setdefault("_settings", {})["access_mode"] = "weird"
    assert storage.get_access_mode() == "private"


def test_private_mode_only_owner_and_co_owners_can_use(fresh_modules) -> None:
    storage, access = fresh_modules
    storage.set_owner_id(111)
    storage.set_access_mode("private")
    assert access.can_use(111) is True
    assert access.can_admin(111) is True
    # Strangers are blocked from both use and admin.
    assert access.can_use(222) is False
    assert access.can_admin(222) is False
    # Co-owner gets full rights.
    assert storage.add_co_owner(222) is True
    assert access.can_use(222) is True
    assert access.can_admin(222) is True


def test_public_mode_everyone_can_use_only_admins_can_admin(fresh_modules) -> None:
    storage, access = fresh_modules
    storage.set_owner_id(111)
    storage.set_access_mode("public")
    # Stranger: use yes, admin no.
    assert access.can_use(222) is True
    assert access.can_admin(222) is False
    # Owner: both.
    assert access.can_use(111) is True
    assert access.can_admin(111) is True
    # Co-owner: both even in public mode.
    storage.add_co_owner(333)
    assert access.can_admin(333) is True


def test_full_public_mode_everyone_admins(fresh_modules) -> None:
    storage, access = fresh_modules
    storage.set_owner_id(111)
    storage.set_access_mode("full_public")
    assert access.can_use(222) is True
    assert access.can_admin(222) is True
    assert access.can_use(None) is False  # Anonymous still rejected.


def test_co_owner_idempotent_add_and_remove(fresh_modules) -> None:
    storage, _ = fresh_modules
    storage.set_owner_id(111)
    assert storage.add_co_owner(222) is True
    assert storage.add_co_owner(222) is False  # already there
    assert storage.get_co_owners() == [222]
    # Cannot add the owner themselves as a co-owner.
    assert storage.add_co_owner(111) is False
    # Remove flow.
    assert storage.remove_co_owner(222) is True
    assert storage.remove_co_owner(222) is False  # not present anymore
    assert storage.get_co_owners() == []


def test_co_owner_persists_across_storage_reload(fresh_modules, tmp_path) -> None:
    storage, _ = fresh_modules
    storage.set_owner_id(111)
    storage.add_co_owner(222)
    storage.set_access_mode("public")
    # Reload the storage from disk.
    import importlib

    storage_mod = importlib.import_module("bot.storage")
    reloaded = storage_mod.Storage(data_dir=tmp_path)
    assert reloaded.get_owner_id() == 111
    assert reloaded.get_co_owners() == [222]
    assert reloaded.get_access_mode() == "public"


def test_is_owner_only_for_real_owner(fresh_modules) -> None:
    storage, access = fresh_modules
    storage.set_owner_id(111)
    storage.add_co_owner(222)
    assert access.is_owner(111) is True
    assert access.is_owner(222) is False
    assert access.is_owner(None) is False
    assert access.is_owner("not-a-number") is False
