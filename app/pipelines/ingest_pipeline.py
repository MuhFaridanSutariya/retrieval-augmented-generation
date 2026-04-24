from uuid import UUID

from app.chunkers.recursive_splitter import RecursiveSplitter
from app.core.config import Settings
from app.core.logging import get_logger
from app.embedders.openai_embedder import OpenAIEmbedder
from app.enums.file_type import FileType
from app.loaders.document_loader import load_document
from app.models.domain.chunk import Chunk
from app.storages.faiss_store import FaissStore

logger = get_logger(__name__)


class IngestPipeline:
    def __init__(
        self,
        *,
        splitter: RecursiveSplitter,
        embedder: OpenAIEmbedder,
        vector_store: FaissStore,
        settings: Settings,
    ) -> None:
        self._splitter = splitter
        self._embedder = embedder
        self._vector_store = vector_store
        self._settings = settings

    async def run(
        self,
        *,
        document_id: UUID,
        file_type: FileType,
        content: bytes,
    ) -> int:
        raw_text = load_document(content, file_type)
        if not raw_text.strip():
            logger.warning("ingest_empty_text", document_id=str(document_id))
            return 0

        text_chunks = self._splitter.split(raw_text)
        if not text_chunks:
            return 0

        embeddings = await self._embedder.embed_many([c.text for c in text_chunks])

        chunks: list[Chunk] = [
            Chunk(
                id=f"{document_id}:{tc.index}",
                document_id=document_id,
                chunk_index=tc.index,
                text=tc.text,
                token_count=tc.token_count,
                embedding=embedding,
            )
            for tc, embedding in zip(text_chunks, embeddings, strict=True)
        ]

        await self._vector_store.upsert_chunks(chunks)
        logger.info(
            "ingest_complete",
            document_id=str(document_id),
            chunk_count=len(chunks),
        )
        return len(chunks)
