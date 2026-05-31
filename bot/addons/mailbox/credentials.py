"""Mailbox credentials resolver.

Lookup order:
1. Lilush state.json (set via in-Telegram wizard) — most flexible.
2. Env vars ``MAILBOX_EMAIL`` / ``MAILBOX_PASSWORD`` (Render dashboard).
3. Optional ``mailbox_credentials.json`` next to lilush's state.json
   (legacy: imported from freelance-mailbox-mcp).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from .. import state as addon_state
from .config import get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MailboxCredentials:
    email: str
    password: str

    @property
    def normalized_password(self) -> str:
        return self.password.replace(" ", "")


def load_credentials() -> MailboxCredentials | None:
    # 1) addon_state (wizard)
    email = (addon_state.get("mailbox", "email", "") or "").strip()
    password = (addon_state.get("mailbox", "password", "") or "").strip()
    if email and password:
        return MailboxCredentials(email=email, password=password)

    # 2) env vars
    settings = get_settings()
    if settings.mailbox_email and settings.mailbox_password:
        return MailboxCredentials(
            email=settings.mailbox_email,
            password=settings.mailbox_password,
        )

    # 3) legacy json file
    path = Path(settings.credentials_file)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read mailbox_credentials.json: %s", exc)
        return None
    e = data.get("mailbox_email") or ""
    p = data.get("mailbox_password") or ""
    if not e or not p:
        return None
    return MailboxCredentials(email=e, password=p)


def save_credentials(email: str, password: str) -> None:
    addon_state.set_("mailbox", "email", email.strip())
    addon_state.set_("mailbox", "password", password.replace(" ", ""))


def is_configured() -> bool:
    return load_credentials() is not None
