"""Append-only log of every incoming Telegram message.

Used as a chat backup AND as the inbox a real Devin session reads when the
bot is in ``brain=devin`` mode (see :func:`bot.storage.Storage.get_brain`).
"""

import logging
from datetime import UTC, datetime
from pathlib import Path

from .config import DATA_DIR

INBOX_LOG: Path = DATA_DIR / "inbox.log"

logger = logging.getLogger(__name__)


def log_inbox(
    *,
    user_id: int | None,
    chat_id: int | None,
    text: str,
    kind: str = "text",
) -> None:
    """Append one line to ``data/inbox.log``.

    Best-effort: if the disk write fails (e.g. read-only fs) we log a warning
    but never raise — message handling must not be blocked by logging.
    """
    timestamp = datetime.now(UTC).isoformat(timespec="seconds")
    safe = (text or "").replace("\n", "\\n")
    line = f"{timestamp} [{kind}] from={user_id} chat={chat_id} text={safe!r}\n"
    try:
        INBOX_LOG.parent.mkdir(parents=True, exist_ok=True)
        with INBOX_LOG.open("a", encoding="utf-8") as fp:
            fp.write(line)
    except OSError as exc:
        logger.warning("failed to append to inbox.log: %s", exc)
