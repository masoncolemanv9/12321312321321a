"""Photo-chooser — when the user sends a photo, the bot first asks them
which subsystem should handle it: Helpzavr (screenshot helper) or the
media-generation flow (img2img / img2video / rmbg).

Previously both Helpzavr and media_ui had their own ``F.photo`` handlers
and competed over every incoming photo. Routing photos through this
single chooser keeps the two subsystems isolated and lets the user
explicitly opt-in to one of them per-photo.
"""

from .handlers import build_photo_chooser_router

__all__ = ["build_photo_chooser_router"]
