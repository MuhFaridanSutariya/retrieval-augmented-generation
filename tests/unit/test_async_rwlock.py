import asyncio

import pytest

from app.utils.async_rwlock import AsyncRWLock


@pytest.mark.asyncio
async def test_multiple_readers_proceed_concurrently() -> None:
    lock = AsyncRWLock()
    counter = {"in_flight": 0, "max_concurrent": 0}

    async def reader() -> None:
        async with lock.read():
            counter["in_flight"] += 1
            counter["max_concurrent"] = max(counter["max_concurrent"], counter["in_flight"])
            await asyncio.sleep(0.02)
            counter["in_flight"] -= 1

    await asyncio.gather(*(reader() for _ in range(5)))
    assert counter["max_concurrent"] >= 2


@pytest.mark.asyncio
async def test_writer_is_exclusive() -> None:
    lock = AsyncRWLock()
    active_writers = 0
    max_active = 0

    async def writer() -> None:
        nonlocal active_writers, max_active
        async with lock.write():
            active_writers += 1
            max_active = max(max_active, active_writers)
            await asyncio.sleep(0.01)
            active_writers -= 1

    await asyncio.gather(*(writer() for _ in range(3)))
    assert max_active == 1


@pytest.mark.asyncio
async def test_writer_blocks_readers() -> None:
    lock = AsyncRWLock()
    order: list[str] = []

    async def writer() -> None:
        async with lock.write():
            order.append("writer_start")
            await asyncio.sleep(0.05)
            order.append("writer_end")

    async def reader() -> None:
        await asyncio.sleep(0.01)
        async with lock.read():
            order.append("reader")

    await asyncio.gather(writer(), reader())
    assert order == ["writer_start", "writer_end", "reader"]


@pytest.mark.asyncio
async def test_reader_blocks_writer_until_drained() -> None:
    lock = AsyncRWLock()
    order: list[str] = []

    async def reader() -> None:
        async with lock.read():
            order.append("reader_start")
            await asyncio.sleep(0.05)
            order.append("reader_end")

    async def writer() -> None:
        await asyncio.sleep(0.01)
        async with lock.write():
            order.append("writer")

    await asyncio.gather(reader(), writer())
    assert order == ["reader_start", "reader_end", "writer"]
