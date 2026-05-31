"""RAM guard addon.

UI for the settings sub-screen "💻 RAM". Lets the owner set a soft RAM
limit (so the agent loop can compress history / refuse to continue
when the bot's RSS gets too close to the Render Free 512MB cap),
toggle whether RSS is reported live in the status bubble during a
task, and pick the over-limit behaviour.

The actual enforcement lives in :mod:`bot.agent` — this addon is
purely the settings UI surface.
"""

from .handlers import build_ram_router, show_screen, current_rss_mb  # noqa: F401

__all__ = ["build_ram_router", "show_screen", "current_rss_mb"]
