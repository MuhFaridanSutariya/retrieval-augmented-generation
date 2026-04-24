from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.embedders.openai_embedder import OpenAIEmbedder
from app.models.domain.chunk import RetrievedChunk
from app.models.orm.document_orm import DocumentORM
from app.storages.faiss_store import FaissStore


class VectorRetriever:
    def __init__(
        self,
        *,
        embedder: OpenAIEmbedder,
        vector_store: FaissStore,
        settings: Settings,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._settings = settings

    async def retrieve(
        self,
        *,
        question: str,
        session: AsyncSession,
        document_ids: list[UUID] | None = None,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        effective_top_k = top_k or self._settings.retrieval_top_k

        embedding = await self._embedder.embed_single(question)
        chunks = await self._vector_store.query(
            embedding=embedding,
            top_k=effective_top_k,
            document_ids=document_ids,
        )

        filtered = [c for c in chunks if c.score >= self._settings.min_relevance_score]
        if not filtered:
            return []

        await self._hydrate_filenames(filtered, session)
        return filtered

    async def _hydrate_filenames(
        self,
        chunks: list[RetrievedChunk],
        session: AsyncSession,
    ) -> None:
        document_ids = list({chunk.document_id for chunk in chunks})
        if not document_ids:
            return

        result = await session.execute(
            select(DocumentORM.id, DocumentORM.filename).where(DocumentORM.id.in_(document_ids))
        )
        filenames = dict(result.all())
        for chunk in chunks:
            chunk.filename = filenames.get(chunk.document_id)
