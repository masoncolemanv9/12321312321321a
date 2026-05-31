"""Thin wrapper around imap-tools that yields normalised :class:`EmailContent`.

This module is **read-only by design**:
- The IMAP folder is always re-selected with ``readonly=True`` before any fetch,
  which makes the IMAP server reject any STORE/COPY/EXPUNGE/APPEND request.
- We never import or instantiate any SMTP client; no email-sending code path
  exists in the project.
- ``mark_task_seen`` (in :mod:`mcp_tools`) only writes to a local JSON state
  file, it does NOT modify message flags on the server.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from imap_tools import AND, MailBox

from .config import get_settings
from .credentials import load_credentials
from .parsers.base import EmailContent, html_to_text


class MailboxNotConfiguredError(RuntimeError):
    """Raised when an IMAP operation is attempted before credentials are provisioned."""


@contextmanager
def open_mailbox(folder: str | None = None) -> Iterator[MailBox]:
    """Open the IMAP mailbox in **read-only** mode, optionally on a specific folder.

    Re-selecting the configured folder with ``readonly=True`` after login
    forces the IMAP server into EXAMINE mode, where any write command
    (STORE, COPY, EXPUNGE, APPEND, …) is refused.

    If ``folder`` is None, the default folder from settings is used.
    """
    settings = get_settings()
    creds = load_credentials()
    if creds is None:
        raise MailboxNotConfiguredError(
            "Mailbox credentials are not configured yet. "
            "Open /setup in a browser and submit the email + app-password form."
        )
    target_folder = folder or settings.mailbox_imap_folder
    # Bound the IMAP socket so a hung connection does not stall the MCP tool
    # request indefinitely (Yandex IMAP can be slow or refuse non-RU IPs).
    mailbox = MailBox(
        settings.mailbox_imap_host,
        settings.mailbox_imap_port,
        timeout=settings.mailbox_imap_timeout,
    )
    with mailbox.login(
        creds.email,
        creds.normalized_password,
        initial_folder=target_folder,
    ):
        mailbox.folder.set(target_folder, readonly=True)
        yield mailbox


def list_imap_folders() -> list[str]:
    """Return all IMAP folder names visible to the configured account.

    Useful to discover Yandex / Gmail category folders such as ``Рассылки``,
    ``Уведомления`` etc. Login uses the default folder; the listing itself
    does not modify state.
    """
    with open_mailbox() as mailbox:
        return [f.name for f in mailbox.folder.list()]


def fetch_emails(
    *,
    sender_substr: str | None = None,
    limit: int = 30,
    only_unseen: bool = False,
    folder: str | None = None,
) -> list[EmailContent]:
    """Fetch up to ``limit`` recent emails from a single IMAP folder."""
    with open_mailbox(folder=folder) as mailbox:
        if sender_substr:
            criteria = (
                AND(from_=sender_substr, seen=False) if only_unseen else AND(from_=sender_substr)
            )
        else:
            criteria = AND(seen=False) if only_unseen else AND(all=True)

        out: list[EmailContent] = []
        for msg in mailbox.fetch(criteria, reverse=True, limit=limit, mark_seen=False, bulk=True):
            text = msg.text or ""
            links: list[str] = []
            if msg.html:
                html_text, html_links = html_to_text(msg.html)
                if not text or len(html_text) > len(text):
                    text = html_text
                links = html_links

            out.append(
                EmailContent(
                    uid=str(msg.uid or ""),
                    from_address=msg.from_ or "",
                    subject=msg.subject or "",
                    received_at=msg.date,
                    text=text,
                    html=msg.html or "",
                    links=links,
                    folder=mailbox.folder.get(),
                )
            )
        return out


def fetch_emails_multi_folder(
    *,
    folders: list[str],
    sender_substr: str | None = None,
    limit_per_folder: int = 30,
) -> list[EmailContent]:
    """Fetch recent emails from multiple folders, silently skipping missing ones.

    Yandex's category tabs (``Рассылки``, ``Уведомления``, ``Социальные сети``)
    are exposed as separate IMAP folders, so to surface all freelance task
    notifications we need to walk each candidate folder. Folders that don't
    exist on the account (e.g. English-only Yandex layout) are skipped.
    """
    out: list[EmailContent] = []
    for folder in folders:
        try:
            out.extend(
                fetch_emails(
                    sender_substr=sender_substr,
                    limit=limit_per_folder,
                    folder=folder,
                )
            )
        except Exception as exc:  # noqa: BLE001 - log + continue to next folder
            import logging

            logging.getLogger(__name__).warning(
                "Skipping folder %r during multi-folder fetch: %s", folder, exc
            )
    return out


def fetch_email_by_uid(uid: str, folder: str | None = None) -> EmailContent | None:
    """Fetch a single message by UID from the given folder (default: settings folder)."""
    with open_mailbox(folder=folder) as mailbox:
        for msg in mailbox.fetch(AND(uid=uid), mark_seen=False, bulk=True):
            text = msg.text or ""
            links: list[str] = []
            if msg.html:
                html_text, html_links = html_to_text(msg.html)
                if not text or len(html_text) > len(text):
                    text = html_text
                links = html_links

            return EmailContent(
                uid=str(msg.uid or uid),
                from_address=msg.from_ or "",
                subject=msg.subject or "",
                received_at=msg.date,
                text=text,
                html=msg.html or "",
                links=links,
                folder=mailbox.folder.get(),
            )
    return None
