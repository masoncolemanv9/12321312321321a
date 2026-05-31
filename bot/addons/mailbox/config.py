"""Mailbox-addon settings (lightweight, no Pydantic).

Originally a Pydantic-Settings model in freelance-mailbox-mcp; replaced
with plain ``dataclasses`` here so the addon doesn't pull
pydantic-settings into lilush's tight dependency tree. Values come from
env vars; runtime overrides (set via the Telegram wizard) live in
lilush's ``state.json`` via :mod:`bot.addons.state`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ...config import DATA_DIR


def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    if v is None:
        return default
    v = v.strip()
    return v if v else default


@dataclass(frozen=True)
class Settings:
    mailbox_email: str | None
    mailbox_password: str | None
    mailbox_imap_host: str
    mailbox_imap_port: int
    mailbox_imap_folder: str
    mailbox_imap_timeout: float
    default_scan_folders: tuple[str, ...]
    credentials_file: str
    bot_poll_interval_seconds: int


def get_settings() -> Settings:
    data_dir: Path = DATA_DIR
    return Settings(
        mailbox_email=_env("MAILBOX_EMAIL"),
        mailbox_password=_env("MAILBOX_PASSWORD"),
        mailbox_imap_host=_env("MAILBOX_IMAP_HOST", "imap.yandex.com") or "imap.yandex.com",
        mailbox_imap_port=int(_env("MAILBOX_IMAP_PORT", "993") or "993"),
        mailbox_imap_folder=_env("MAILBOX_IMAP_FOLDER", "INBOX") or "INBOX",
        mailbox_imap_timeout=float(_env("MAILBOX_IMAP_TIMEOUT", "20.0") or "20.0"),
        default_scan_folders=(
            "INBOX",
            "Рассылки",
            "Уведомления",
            "Социальные сети",
        ),
        credentials_file=str(data_dir / "mailbox_credentials.json"),
        bot_poll_interval_seconds=int(
            _env("MAILBOX_POLL_INTERVAL", "300") or "300"
        ),
    )
