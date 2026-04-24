import asyncio
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.models.domain.chunk import RetrievedChunk
from app.models.orm.document_orm import DocumentORM
from app.retrievers.bm25_retriever import BM25Retriever
from app.retrievers.vector_retriever import VectorRetriever


class HybridRetriever:
    # Combines dense (vector) and sparse (BM25) retrieval via Reciprocal Rank Fusion.
    # RRF is rank-based so it does not require score normalisation between the two
    # systems — a chunk ranked r-th in either list contributes 1 / (rrf_k + r).

    def __init__(
        self,
        *,
        vector_retriever: VectorRetriever,
        bm25_retriever: BM25Retriever,
        settings: Settings,
    ) -> None:
        self._vector = vector_retriever
        self._bm25 = bm25_retriever
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
        candidate_pool = effective_top_k * self._settings.hybrid_candidate_multiplier

        vector_results, bm25_results = await asyncio.gather(
            self._vector.retrieve(
                question=question,
                session=session,
                document_ids=document_ids,
                top_k=candidate_pool,
            ),
            self._bm25.retrieve(
                question=question,
                top_k=candidate_pool,
                document_ids=document_ids,
            ),
        )

        fused = _reciprocal_rank_fusion(
            [vector_results, bm25_results],
            rrf_k=self._settings.rrf_k,
        )

        top = fused[:effective_top_k]
        await self._hydrate_filenames(top, session)
        return top

    async def _hydrate_filenames(
        self,
        chunks: list[RetrievedChunk],
        session: AsyncSession,
    ) -> None:
        document_ids = list({chunk.document_id for chunk in chunks})
        if not document_ids:
            return
        result = await session.execute(
            select(DocumentORM.id, DocumentORM.filename).where(
                DocumentORM.id.in_(document_ids)
            )
        )
        filenames = dict(result.all())
        for chunk in chunks:
            if chunk.filename is None:
                chunk.filename = filenames.get(chunk.document_id)


def _reciprocal_rank_fusion(
    result_lists: list[list[RetrievedChunk]],
    *,
    rrf_k: int,
) -> list[RetrievedChunk]:
    fused_scores: dict[str, float] = {}
    best_chunk: dict[str, RetrievedChunk] = {}

    for results in result_lists:
        for rank, chunk in enumerate(results, start=1):
            fused_scores[chunk.id] = fused_scores.get(chunk.id, 0.0) + 1.0 / (rrf_k + rank)
            # Keep whichever copy has the higher raw score so downstream consumers
            # (reranker, citation builder) see the more informative snippet text.
            existing = best_chunk.get(chunk.id)
            if existing is None or chunk.score > existing.score:
                best_chunk[chunk.id] = chunk

    ranked_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)
    fused: list[RetrievedChunk] = []
    for chunk_id in ranked_ids:
        chunk = best_chunk[chunk_id]
        fused.append(
            RetrievedChunk(
                id=chunk.id,
                document_id=chunk.document_id,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                score=fused_scores[chunk_id],
                filename=chunk.filename,
            )
        )
    return fused
