"""Bot memory addon.

Surfaces the agent's conversation history via the main-menu screen
"🧠 Память бота" so the owner can see *how many messages the bot is
keeping in context for /chat*, and clear them on demand.

The actual history storage already lives in :mod:`bot.storage`
(``storage.get_history`` / ``storage.append_history``) and runs with
``HISTORY_LIMIT=20`` by default — so the bot already remembers ~40
turns. This addon just makes that visible and clearable from the UI
rather than only via the ``/reset`` slash-command.
"""

from .handlers import build_memory_router, show_screen  # noqa: F401

__all__ = ["build_memory_router", "show_screen"]
