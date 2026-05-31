"""Tests for the SQLite-backed job queue."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from bot.jobs import Job, JobQueue


@pytest.fixture
async def queue(tmp_path: Path) -> JobQueue:
    q = JobQueue(tmp_path / "jobs.db")
    await q.init()
    return q


async def test_enqueue_and_get(queue: JobQueue) -> None:
    job_id = await queue.enqueue("download", {"url": "https://example.com/x"}, chat_id=42)
    job = await queue.get(job_id)
    assert job is not None
    assert job.kind == "download"
    assert job.status == "queued"
    assert job.payload == {"url": "https://example.com/x"}
    assert job.chat_id == 42
    assert job.parent_id is None


async def test_pop_one_transitions_status(queue: JobQueue) -> None:
    await queue.enqueue("download", {"url": "https://x"}, chat_id=1)
    job = await queue.pop_one("download")
    assert job is not None
    assert job.status == "running"
    fetched = await queue.get(job.id)
    assert fetched is not None
    assert fetched.status == "running"


async def test_pop_one_kind_filter(queue: JobQueue) -> None:
    await queue.enqueue("download", {"u": "1"}, chat_id=1)
    await queue.enqueue("edit", {"u": "2"}, chat_id=1)

    # Asking for "analyze" gets nothing even though queue is non-empty.
    assert await queue.pop_one("analyze") is None

    # Asking for the right kind returns its job.
    edit_job = await queue.pop_one("edit")
    assert edit_job is not None
    assert edit_job.kind == "edit"

    download_job = await queue.pop_one("download")
    assert download_job is not None
    assert download_job.kind == "download"


async def test_pop_one_atomic_under_concurrency(queue: JobQueue) -> None:
    """Concurrent ``pop_one`` calls each return distinct rows or None."""
    for i in range(20):
        await queue.enqueue("download", {"i": i}, chat_id=1)

    # 30 concurrent poppers fight over 20 queued jobs → 20 winners + 10 None.
    results = await asyncio.gather(*[queue.pop_one("download") for _ in range(30)])
    winners = [j for j in results if j is not None]
    losers = [j for j in results if j is None]
    assert len(winners) == 20
    assert len(losers) == 10
    assert len({j.id for j in winners}) == 20  # no duplicates


async def test_mark_done_persists_result(queue: JobQueue) -> None:
    job_id = await queue.enqueue("download", {"url": "https://x"}, chat_id=1)
    await queue.pop_one("download")
    await queue.mark_done(job_id, {"file_path": "/tmp/x.mp4"})
    fetched = await queue.get(job_id)
    assert fetched is not None
    assert fetched.status == "done"
    assert fetched.result == {"file_path": "/tmp/x.mp4"}


async def test_mark_failed_records_error(queue: JobQueue) -> None:
    job_id = await queue.enqueue("download", {"url": "https://x"}, chat_id=1)
    await queue.pop_one("download")
    await queue.mark_failed(job_id, "boom\nTraceback ...")
    fetched = await queue.get(job_id)
    assert fetched is not None
    assert fetched.status == "failed"
    assert fetched.error is not None
    assert "boom" in fetched.error
    assert fetched.retries == 1


async def test_parent_id_chains(queue: JobQueue) -> None:
    parent_id = await queue.enqueue("download", {"url": "https://x"}, chat_id=1)
    child_id = await queue.enqueue(
        "analyze", {"file_path": "/tmp/x.mp4"}, chat_id=1, parent_id=parent_id
    )
    child = await queue.get(child_id)
    assert child is not None
    assert child.parent_id == parent_id


async def test_list_active_filters_terminal_states(queue: JobQueue) -> None:
    a = await queue.enqueue("download", {"u": "1"}, chat_id=1)
    b = await queue.enqueue("download", {"u": "2"}, chat_id=1)
    c = await queue.enqueue("download", {"u": "3"}, chat_id=2)

    # Terminate one of them.
    await queue.pop_one("download")  # a → running
    await queue.mark_done(a, {"file_path": "/x"})  # a → done

    active_all = await queue.list_active()
    active_ids = {j.id for j in active_all}
    assert a not in active_ids
    assert b in active_ids
    assert c in active_ids

    # Scope to chat=1 → should not include c.
    active_chat1 = await queue.list_active(chat_id=1)
    active_ids_chat1 = {j.id for j in active_chat1}
    assert active_ids_chat1 == {b}


async def test_list_by_chat_descending(queue: JobQueue) -> None:
    ids = []
    for i in range(5):
        ids.append(await queue.enqueue("download", {"i": i}, chat_id=7))
    listed = await queue.list_by_chat(7, limit=10)
    assert [j.id for j in listed] == list(reversed(ids))


async def test_job_from_row_decodes_json(tmp_path: Path) -> None:
    """Sanity check: Job.from_row understands SQLite tuples directly."""
    q = JobQueue(tmp_path / "jobs.db")
    await q.init()
    job_id = await q.enqueue("seo", {"deep": {"nested": [1, 2, 3]}}, chat_id=99)
    job = await q.get(job_id)
    assert isinstance(job, Job)
    assert job.payload == {"deep": {"nested": [1, 2, 3]}}
