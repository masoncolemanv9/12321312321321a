"""Workers that pull jobs from the SQLite queue and chain pipeline stages.

Each worker subclasses :class:`bot.workers.base.Worker` and overrides
``process()``. The base class handles polling, status transitions and
failure paths so concrete workers stay focused on their stage logic.

For the ``edit`` stage two implementations live side-by-side:

* :class:`bot.workers.editor.EditorWorker` — the legacy v1 path
  (single crop+scale + optional logo overlay). **Default.**
* :class:`bot.workers.editor_v2.EditorV2Worker` — the v2.0 baseline
  pipeline (face-aware zoom, mirror, colorgrade, loudnorm, manifest).
  Opt-in via ``EDITOR_VERSION=v2`` (or ``v2.1`` / ``v6`` once their
  layers land in later Parts).

Use :func:`build_editor_worker` to instantiate the right one at boot.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

from .analyzer import AnalyzerWorker
from .base import Worker
from .downloader import DownloaderWorker
from .editor import EditorWorker
from .editor_v2 import EditorV2Worker
from .publisher import PublisherWorker
from .seo import SeoWorker

if TYPE_CHECKING:
    from ..jobs import JobQueue


def build_editor_worker(
    queue: JobQueue, *, poll_interval_s: float = 1.0
) -> Worker:
    """Return the right ``edit`` worker for the current ``EDITOR_VERSION``.

    Resolution: prefer the live ``bot.config.EDITOR_VERSION`` module
    attribute (so tests / the Telegram UI can update it at runtime),
    then fall back to ``os.environ["EDITOR_VERSION"]``, then ``"v1"``.
    Looking up ``sys.modules`` on every call keeps the factory honest
    when other tests reload ``bot.config`` (``sys.modules.pop`` +
    ``importlib.import_module``).
    """
    config_mod = sys.modules.get("bot.config")
    raw = getattr(config_mod, "EDITOR_VERSION", None) if config_mod else None
    if not raw:
        raw = os.environ.get("EDITOR_VERSION", "")
    version = (raw or "v1").lower().strip()
    if version in {"v2", "v2.0", "v2.1", "v6"}:
        return EditorV2Worker(queue, poll_interval_s=poll_interval_s)
    if version != "v1":
        import logging  # local — keep module-import side effects minimal
        logging.getLogger(__name__).warning(
            "EDITOR_VERSION=%r is not recognised; falling back to v1", version
        )
    return EditorWorker(queue, poll_interval_s=poll_interval_s)


__all__ = [
    "AnalyzerWorker",
    "DownloaderWorker",
    "EditorV2Worker",
    "EditorWorker",
    "PublisherWorker",
    "SeoWorker",
    "Worker",
    "build_editor_worker",
]
