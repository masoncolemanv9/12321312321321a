"""Telegram UI for the Editor Agent (v1 / v2 / v2.1 / v6 dispatch).

This addon adds the **🎞 Монтажёр** main-menu screen — a status & runtime
configuration panel for the editor pipeline that Parts 1-11 built. It
does not invoke the pipeline directly (the worker queue does that
asynchronously for ``edit`` jobs); instead it exposes the three knobs
that change what the next ``edit`` job will do:

* **Editor version** — ``v1`` (legacy default) / ``v2`` (face-aware
  baseline) / ``v2.1`` (cuts + mirror + subtitles) / ``v6`` (creative
  planner orchestration).
* **Intensity profile** — ``light`` / ``medium`` / ``heavy`` — caps how
  aggressive the creative planner is allowed to be.
* **v6 creative planner** — master ON/OFF for the v6 dispatch inside
  ``EditorV2Worker``.

State is stored in lilush's settings dict under
``_settings.addons.editor_agent.*``. When a key is missing, the live
``bot.config`` value (which itself defaults to the matching env var)
is used — so a fresh deploy keeps env-var behaviour until the user
touches the toggle.
"""

from .handlers import (
    build_editor_agent_router,
    current_profile,
    current_version,
    is_v6_enabled,
    show_screen,
)

__all__ = [
    "build_editor_agent_router",
    "current_profile",
    "current_version",
    "is_v6_enabled",
    "show_screen",
]
