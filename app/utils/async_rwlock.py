import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager


class AsyncRWLock:
    # Writer-preference readers-writer lock built on asyncio.Condition.
    # Many readers can hold the lock concurrently; a writer has exclusive access.
    # Readers yield to queued writers so writes never starve under steady read load.

    def __init__(self) -> None:
        self._cond = asyncio.Condition()
        self._readers: int = 0
        self._writer_active: bool = False
        self._writers_waiting: int = 0

    @asynccontextmanager
    async def read(self) -> AsyncIterator[None]:
        async with self._cond:
            while self._writer_active or self._writers_waiting > 0:
                await self._cond.wait()
            self._readers += 1
        try:
            yield
        finally:
            async with self._cond:
                self._readers -= 1
                if self._readers == 0:
                    self._cond.notify_all()

    @asynccontextmanager
    async def write(self) -> AsyncIterator[None]:
        async with self._cond:
            self._writers_waiting += 1
            try:
                while self._writer_active or self._readers > 0:
                    await self._cond.wait()
                self._writer_active = True
            finally:
                self._writers_waiting -= 1
        try:
            yield
        finally:
            async with self._cond:
                self._writer_active = False
                self._cond.notify_all()
