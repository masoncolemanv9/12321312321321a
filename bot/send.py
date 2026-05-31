"""CLI to send a Telegram message *through the bot* from a shell.

Used by a real Devin session that has shell access to the bot's VM so it can
reply to the user when the bot is in ``brain=devin`` mode.

Usage:
    python -m bot.send <chat_id> <text...>
    echo "long reply" | python -m bot.send <chat_id> -

Both "raw" (no parse_mode) and HTML modes are supported via ``--html``.
"""

import argparse
import asyncio
import sys

from aiogram import Bot
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from .config import BOT_TOKEN

# Telegram hard limit; messages longer than this are split.
_TG_LIMIT = 4096


async def _send(chat_id: int, text: str, html: bool) -> None:
    if not text.strip():
        raise SystemExit("empty message; refusing to send")
    default = (
        DefaultBotProperties(parse_mode=ParseMode.HTML)
        if html
        else DefaultBotProperties()
    )
    bot = Bot(BOT_TOKEN, default=default)
    try:
        # Naive split on TG_LIMIT — caller can pre-format if they want better cuts.
        for i in range(0, len(text), _TG_LIMIT):
            await bot.send_message(chat_id, text[i : i + _TG_LIMIT])
    finally:
        await bot.session.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Send a Telegram message through the bot.")
    parser.add_argument("chat_id", type=int, help="Telegram chat id (usually the user id)")
    parser.add_argument(
        "text",
        nargs="*",
        help="Message text. Use '-' to read from stdin.",
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Render as HTML (default: plain text)",
    )
    args = parser.parse_args(argv)

    text = sys.stdin.read() if not args.text or args.text == ["-"] else " ".join(args.text)
    asyncio.run(_send(args.chat_id, text, args.html))


if __name__ == "__main__":
    main()
