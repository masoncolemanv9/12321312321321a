"""Base worker class shared by every pipeline stage.

A worker runs an asyncio polling loop:

1. ``pop_one(kind)`` claims the oldest queued job of its stage.
2. ``process(job)`` produces a result dict (or raises).
3. On success, the worker marks the job ``done`` and enqueues the next
   stage via ``next_kind`` (if set) with ``parent_id=job.id``.
4. On error, the job goes to ``failed`` with the traceback in ``error``.

Multiple workers of the same kind are safe to run concurrently — the
queue's atomic ``UPDATE ... RETURNING`` guarantees each job is claimed
by exactly one worker.
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from typing import Any

from ..jobs import Job, JobQueue

logger = logging.getLogger(__name__)


class Worker:
    """Subclass and override :meth:`process` (and optionally :attr:`next_kind`).

    Workers are owned by the application's main event loop. To stop a
    running worker call :meth:`stop` — it sets a flag, the polling loop
    finishes the in-flight job and exits cleanly.
    """

    #: Stage this worker handles. Subclasses must set this.
    kind: str = ""

    #: Stage that should run after this one. ``None`` means terminal stage.
    #: Workers can override :meth:`enqueue_next` for fan-out (multiple
    #: child jobs from one parent — e.g. analyzer → many editor jobs).
    next_kind: str | None = None

    def __init__(self, queue: JobQueue, *, poll_interval_s: float = 1.0) -> None:
        if not self.kind:
            raise ValueError(f"{type(self).__name__}.kind must be set")
        self.queue = queue
        self.poll_interval_s = poll_interval_s
        self._stopping = asyncio.Event()

    # ---- subclass surface ------------------------------------------------

    async def process(self, job: Job) -> dict[str, Any]:
        """Do the actual work for ``job``. Return result dict to persist.

        Raise any exception to mark the job failed. The traceback is
        captured into ``jobs.error``.
        """
        raise NotImplementedError

    async def enqueue_next(self, job: Job, result: dict[str, Any]) -> None:
        """Enqueue the next-stage job(s) after ``job`` succeeds.

        Default behaviour: if :attr:`next_kind` is set, enqueue exactly
        one child job whose payload is the just-produced ``result``.
        Workers that need fan-out (e.g. analyzer creating one edit job
        per clip) override this method.
        """
        if not self.next_kind:
            return
        await self.queue.enqueue(
            self.next_kind,
            result,
            chat_id=job.chat_id,
            parent_id=job.id,
        )

    # ---- lifecycle -------------------------------------------------------

    def stop(self) -> None:
        self._stopping.set()

    async def run(self) -> None:
        logger.info("worker[%s] started", self.kind)
        try:
            while not self._stopping.is_set():
                try:
                    job = await self.queue.pop_one(self.kind)
                except Exception as exc:  # noqa: BLE001 — never let the loop die
                    logger.exception("worker[%s] pop_one failed: %s", self.kind, exc)
                    await self._sleep_or_stop()
                    continue
                if job is None:
                    await self._sleep_or_stop()
                    continue
                try:
                    await self._handle(job)
                except Exception as exc:  # noqa: BLE001 — never let the loop die
                    # _handle already protects ``process``; this catches
                    # failures from mark_done / mark_failed / enqueue_next
                    # (e.g. SQLite BUSY, full disk) so a transient DB
                    # error doesn't kill the worker permanently.
                    logger.exception(
                        "worker[%s] _handle crashed on job=%s: %s",
                        self.kind,
                        job.id,
                        exc,
                    )
                    await self._sleep_or_stop()
        finally:
            logger.info("worker[%s] stopped", self.kind)

    async def _sleep_or_stop(self) -> None:
        try:
            await asyncio.wait_for(self._stopping.wait(), timeout=self.poll_interval_s)
        except TimeoutError:
            return

    async def _handle(self, job: Job) -> None:
        logger.info("worker[%s] processing job=%s parent=%s", self.kind, job.id, job.parent_id)
        try:
            result = await self.process(job)
        except Exception as exc:  # noqa: BLE001 — every failure is logged
            tb = traceback.format_exc()
            logger.exception("worker[%s] job=%s failed: %s", self.kind, job.id, exc)
            await self.queue.mark_failed(job.id, tb)
            return

        await self.queue.mark_done(job.id, result)
        try:
            await self.enqueue_next(job, result)
        except Exception:
            # Failure to chain shouldn't unmark the parent — we log loudly
            # and leave a human or a re-runner to recover.
            logger.exception(
                "worker[%s] job=%s succeeded but enqueue_next failed", self.kind, job.id
            )
