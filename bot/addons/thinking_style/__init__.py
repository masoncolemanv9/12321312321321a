"""Thinking-style addon — let the user pick how the bot shows that it
is currently working on something.

Three render modes are available, plus a "back" button:

* ``white`` — a single white status bubble that edits in place
  ("Думаю…" → "Размышляю…" → "Готово"). This matches what lilush did
  out of the box.
* ``white_model`` — same bubble, but each line is suffixed with the
  active model name so the user can see *who* is thinking.
* ``typing`` — only Telegram's native ``send_chat_action("typing")``
  indicator is shown (the small "… печатает" hint at the top of the
  conversation). No status message is sent or edited. This is the
  minimal style from the original lilush before tg_bot+(13) was merged.

The chosen style is read by ``handlers._make_status_updater`` when it
builds the per-request ``on_status`` callback, and by the addon-side
status helpers in Helpzavr / Pretty Text via :func:`make_status_runner`.
"""

from .handlers import (
    build_thinking_style_router,
    get_style,
    is_typing_only,
    make_status_runner,
    show_screen,
)

__all__ = [
    "build_thinking_style_router",
    "get_style",
    "is_typing_only",
    "make_status_runner",
    "show_screen",
]
