import asyncio
import re
from uuid import UUID

from rank_bm25 import BM25Okapi

from app.core.config import Settings
from app.models.domain.chunk import RetrievedChunk
from app.storages.faiss_store import FaissStore
from app.utils.async_rwlock import AsyncRWLock

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN_PATTERN.finditer(text or "")]


class BM25Retriever:
    def __init__(self, *, faiss_store: FaissStore, settings: Settings) -> None:
        self._faiss_store = faiss_store
        self._settings = settings
        self._build_lock = AsyncRWLock()
        self._built_generation: int = -1
        self._bm25: BM25Okapi | None = None
        self._chunks: list[dict] = []

    async def retrieve(
        self,
        *,
        question: str,
        top_k: int,
        document_ids: list[UUID] | None = None,
    ) -> list[RetrievedChunk]:
        await self._ensure_index_current()

        if self._bm25 is None or not self._chunks:
            return []

        async with self._build_lock.read():
            tokens = _tokenize(question)
            if not tokens:
                return []

            scores = await asyncio.to_thread(self._bm25.get_scores, tokens)
            allowed = {str(d) for d in document_ids} if document_ids else None

            ranked: list[tuple[float, dict]] = []
            for score, chunk in zip(scores, self._chunks, strict=True):
                if allowed is not None and chunk["document_id"] not in allowed:
                    continue
                if score <= 0:
                    continue
                ranked.append((float(score), chunk))

            ranked.sort(key=lambda pair: pair[0], reverse=True)
            top = ranked[:top_k]

            return [
                RetrievedChunk(
                    id=chunk["chunk_id"],
                    document_id=UUID(chunk["document_id"]),
                    chunk_index=int(chunk["chunk_index"]),
                    text=chunk["text"],
                    score=score,
                )
                for score, chunk in top
            ]

    async def _ensure_index_current(self) -> None:
        generation, chunks = await self._faiss_store.snapshot_chunks()
        if generation == self._built_generation and self._bm25 is not None:
            return

        async with self._build_lock.write():
            if generation == self._built_generation and self._bm25 is not None:
                return
            if not chunks:
                self._bm25 = None
                self._chunks = []
                self._built_generation = generation
                return

            tokenised = await asyncio.to_thread(
                lambda: [_tokenize(c["text"]) for c in chunks]
            )
            self._bm25 = await asyncio.to_thread(BM25Okapi, tokenised)
            self._chunks = chunks
            self._built_generation = generation
