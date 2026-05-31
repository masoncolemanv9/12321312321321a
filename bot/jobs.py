"""Async SQLite-backed job queue for the Lilush pipeline.

A *job* represents one stage of work for a given video — download, analyse,
edit, seo, publish. Workers poll the queue for jobs of their own ``kind``
and chain to the next stage by enqueuing a new job with ``parent_id``
pointing back at the completed one.

The queue is intentionally small and dependency-free: aiosqlite is the only
runtime dep added on top of stdlib. SQLite ``UPDATE ... RETURNING`` is used
for atomic ``pop_one``, so multiple workers of the same kind can poll
without stepping on each other.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)


JobKind = str  # 'download' | 'analyze' | 'edit' | 'seo' | 'publish'
JobStatus = str  # 'queued' | 'running' | 'done' | 'failed'


_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT    NOT NULL,
    status      TEXT    NOT NULL DEFAULT 'queued',
    parent_id   INTEGER,
    payload     TEXT    NOT NULL,
    result      TEXT,
    error       TEXT,
    retries     INTEGER NOT NULL DEFAULT 0,
    chat_id     INTEGER,
    created_at  TEXT    NOT NULL,
    updated_at  TEXT    NOT NULL,
    FOREIGN KEY (parent_id) REFERENCES jobs(id)
);

CREATE INDEX IF NOT EXISTS idx_jobs_kind_status ON jobs(kind, status);
CREATE INDEX IF NOT EXISTS idx_jobs_chat ON jobs(chat_id);
"""


@dataclass
class Job:
    """One row from the ``jobs`` table, decoded for callers.

    ``payload`` and ``result`` are stored as JSON in SQLite but exposed
    as ``dict`` so workers don't repeat the parse/dump dance.
    """

    id: int
    kind: JobKind
    status: JobStatus
    parent_id: int | None
    payload: dict[str, Any]
    result: dict[str, Any] | None
    error: str | None
    retries: int
    chat_id: int | None
    created_at: str
    updated_at: str

    @classmethod
    def from_row(cls, row: aiosqlite.Row | tuple) -> Job:
        # aiosqlite.Row supports both index and key access; we use indexes
        # so this works whether or not row_factory is set.
        return cls(
            id=row[0],
            kind=row[1],
            status=row[2],
            parent_id=row[3],
            payload=json.loads(row[4]),
            result=json.loads(row[5]) if row[5] else None,
            error=row[6],
            retries=row[7],
            chat_id=row[8],
            created_at=row[9],
            updated_at=row[10],
        )


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class JobQueue:
    """Thin async wrapper over a single ``jobs.db``.

    Each method opens its own connection — aiosqlite uses a worker thread
    per connection, and the per-call cost is microseconds. Keeps the API
    simple and avoids long-lived connection lifecycle bugs.
    """

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    async def init(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(_SCHEMA)
            await db.commit()

    async def enqueue(
        self,
        kind: JobKind,
        payload: dict[str, Any],
        *,
        chat_id: int | None = None,
        parent_id: int | None = None,
    ) -> int:
        now = _now()
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute(
                "INSERT INTO jobs(kind, status, parent_id, payload, chat_id, created_at, updated_at) "
                "VALUES (?, 'queued', ?, ?, ?, ?, ?)",
                (kind, parent_id, json.dumps(payload, ensure_ascii=False), chat_id, now, now),
            )
            await db.commit()
            job_id = cur.lastrowid
        logger.info("enqueued job %s kind=%s chat=%s parent=%s", job_id, kind, chat_id, parent_id)
        assert job_id is not None  # SQLite always returns lastrowid for AUTOINCREMENT
        return job_id

    async def pop_one(self, kind: JobKind) -> Job | None:
        """Atomically claim the oldest queued job of ``kind``.

        Uses ``UPDATE ... RETURNING`` so two workers polling simultaneously
        will get different rows or one will get nothing. Available since
        SQLite 3.35 (Python 3.12 ships with 3.45+).
        """
        now = _now()
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute(
                "UPDATE jobs SET status='running', updated_at=? "
                "WHERE id = (SELECT id FROM jobs WHERE kind=? AND status='queued' ORDER BY id LIMIT 1) "
                "RETURNING id, kind, status, parent_id, payload, result, error, retries, chat_id, created_at, updated_at",
                (now, kind),
            )
            row = await cur.fetchone()
            await db.commit()
        if row is None:
            return None
        return Job.from_row(row)

    async def mark_done(self, job_id: int, result: dict[str, Any]) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE jobs SET status='done', result=?, updated_at=? WHERE id=?",
                (json.dumps(result, ensure_ascii=False), _now(), job_id),
            )
            await db.commit()

    async def mark_failed(self, job_id: int, error: str) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE jobs SET status='failed', error=?, retries=retries+1, updated_at=? WHERE id=?",
                (error, _now(), job_id),
            )
            await db.commit()

    async def get(self, job_id: int) -> Job | None:
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute(
                "SELECT id, kind, status, parent_id, payload, result, error, retries, chat_id, created_at, updated_at "
                "FROM jobs WHERE id=?",
                (job_id,),
            )
            row = await cur.fetchone()
        if row is None:
            return None
        return Job.from_row(row)

    async def list_active(self, chat_id: int | None = None) -> list[Job]:
        """Return all jobs that haven't reached a terminal state.

        ``chat_id=None`` returns every active job in the system; pass an id
        to scope to one user's pipeline.
        """
        sql = (
            "SELECT id, kind, status, parent_id, payload, result, error, retries, chat_id, created_at, updated_at "
            "FROM jobs WHERE status IN ('queued','running')"
        )
        params: tuple[Any, ...] = ()
        if chat_id is not None:
            sql += " AND chat_id=?"
            params = (chat_id,)
        sql += " ORDER BY id"
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute(sql, params)
            rows = await cur.fetchall()
        return [Job.from_row(r) for r in rows]

    async def list_by_chat(self, chat_id: int, limit: int = 50) -> list[Job]:
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute(
                "SELECT id, kind, status, parent_id, payload, result, error, retries, chat_id, created_at, updated_at "
                "FROM jobs WHERE chat_id=? ORDER BY id DESC LIMIT ?",
                (chat_id, limit),
            )
            rows = await cur.fetchall()
        return [Job.from_row(r) for r in rows]


# ---- module-level singleton ---------------------------------------------

_default_queue: JobQueue | None = None


def get_default_queue() -> JobQueue:
    """Return the process-wide :class:`JobQueue` bound to ``JOBS_DB_PATH``.

    Both handlers and worker setup use this so they share one queue file
    without callers needing to pass it around explicitly.
    """
    global _default_queue
    if _default_queue is None:
        from .config import JOBS_DB_PATH

        _default_queue = JobQueue(JOBS_DB_PATH)
    return _default_queue
