"""Master photo/video toggle — silences lilush's media_ui prompt and the
helpzavr pipeline when the user just wants to share a picture casually
without the bot asking "что делаем с фото?".

The toggle is stored per-chat in lilush's addon state. Default = ON so
fresh deploys keep the lilush behavior the user is used to.
"""

from .handlers import build_media_toggle_router, is_media_enabled, show_screen

__all__ = ["build_media_toggle_router", "is_media_enabled", "show_screen"]
