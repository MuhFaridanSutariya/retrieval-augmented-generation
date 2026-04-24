import asyncio
import json
import os
import tempfile
from pathlib import Path
from uuid import UUID

import faiss
import numpy as np

from app.core.config import Settings
from app.core.exceptions import VectorStoreError
from app.core.logging import get_logger
from app.models.domain.chunk import Chunk, RetrievedChunk

logger = get_logger(__name__)


class FaissStore:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._dimension = settings.openai_embedding_dimensions
        self._index_path = Path(settings.faiss_index_path)
        self._metadata_path = Path(settings.faiss_metadata_path)
        self._oversample = max(1, settings.faiss_oversample_factor)

        self._lock = asyncio.Lock()
        self._index: faiss.Index | None = None
        self._next_id: int = 0
        self._metadata: dict[int, dict] = {}
        self._chunk_to_faiss_id: dict[str, int] = {}

    async def ensure_index(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self._load_or_create)

    def _load_or_create(self) -> None:
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        self._metadata_path.parent.mkdir(parents=True, exist_ok=True)

        if self._index_path.exists() and self._metadata_path.exists():
            try:
                self._index = faiss.read_index(str(self._index_path))
                payload = json.loads(self._metadata_path.read_text(encoding="utf-8"))
                self._next_id = int(payload.get("next_id", 0))
                self._metadata = {int(k): v for k, v in payload.get("metadata", {}).items()}
                self._chunk_to_faiss_id = dict(payload.get("chunk_to_faiss_id", {}))
                logger.info(
                    "faiss_index_loaded",
                    path=str(self._index_path),
                    vector_count=self._index.ntotal,
                )
                return
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                logger.warning("faiss_index_load_failed", error=str(exc))

        base = faiss.IndexFlatIP(self._dimension)
        self._index = faiss.IndexIDMap(base)
        self._next_id = 0
        self._metadata = {}
        self._chunk_to_faiss_id = {}
        logger.info("faiss_index_created", dimension=self._dimension)

    async def upsert_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return

        async with self._lock:
            if self._index is None:
                raise VectorStoreError("FAISS index is not initialised.")
            try:
                await asyncio.to_thread(self._upsert_sync, chunks)
            except Exception as exc:
                raise VectorStoreError(f"FAISS upsert failed: {exc}") from exc

    def _upsert_sync(self, chunks: list[Chunk]) -> None:
        assert self._index is not None

        stale_ids: list[int] = []
        for chunk in chunks:
            existing = self._chunk_to_faiss_id.get(chunk.id)
            if existing is not None:
                stale_ids.append(existing)

        if stale_ids:
            self._index.remove_ids(np.asarray(stale_ids, dtype=np.int64))
            for stale in stale_ids:
                self._metadata.pop(stale, None)

        embeddings: list[list[float]] = []
        ids: list[int] = []
        for chunk in chunks:
            if chunk.embedding is None:
                continue
            faiss_id = self._next_id
            self._next_id += 1
            ids.append(faiss_id)
            embeddings.append(chunk.embedding)
            self._metadata[faiss_id] = {
                "chunk_id": chunk.id,
                "document_id": str(chunk.document_id),
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                "token_count": chunk.token_count,
            }
            self._chunk_to_faiss_id[chunk.id] = faiss_id

        if not embeddings:
            self._persist()
            return

        vectors = np.asarray(embeddings, dtype=np.float32)
        faiss.normalize_L2(vectors)
        self._index.add_with_ids(vectors, np.asarray(ids, dtype=np.int64))
        self._persist()

    async def query(
        self,
        embedding: list[float],
        top_k: int,
        document_ids: list[UUID] | None = None,
    ) -> list[RetrievedChunk]:
        async with self._lock:
            if self._index is None:
                raise VectorStoreError("FAISS index is not initialised.")
            if self._index.ntotal == 0:
                return []
            try:
                return await asyncio.to_thread(
                    self._query_sync, embedding, top_k, document_ids
                )
            except Exception as exc:
                raise VectorStoreError(f"FAISS query failed: {exc}") from exc

    def _query_sync(
        self,
        embedding: list[float],
        top_k: int,
        document_ids: list[UUID] | None,
    ) -> list[RetrievedChunk]:
        assert self._index is not None

        query_vector = np.asarray([embedding], dtype=np.float32)
        faiss.normalize_L2(query_vector)

        fetch_k = top_k * self._oversample if document_ids else top_k
        fetch_k = min(fetch_k, self._index.ntotal)

        scores, ids = self._index.search(query_vector, fetch_k)
        allowed: set[str] | None = (
            {str(d) for d in document_ids} if document_ids else None
        )

        retrieved: list[RetrievedChunk] = []
        for score, faiss_id in zip(scores[0], ids[0], strict=True):
            if faiss_id == -1:
                continue
            metadata = self._metadata.get(int(faiss_id))
            if metadata is None:
                continue
            if allowed is not None and metadata["document_id"] not in allowed:
                continue
            retrieved.append(
                RetrievedChunk(
                    id=metadata["chunk_id"],
                    document_id=UUID(metadata["document_id"]),
                    chunk_index=int(metadata["chunk_index"]),
                    text=metadata["text"],
                    score=float(score),
                )
            )
            if len(retrieved) >= top_k:
                break

        return retrieved

    async def delete_by_document(self, document_id: UUID) -> None:
        async with self._lock:
            if self._index is None:
                return
            try:
                await asyncio.to_thread(self._delete_sync, document_id)
            except Exception as exc:
                raise VectorStoreError(f"FAISS delete failed: {exc}") from exc

    def _delete_sync(self, document_id: UUID) -> None:
        assert self._index is not None
        target = str(document_id)

        doomed_ids: list[int] = [
            faiss_id
            for faiss_id, meta in self._metadata.items()
            if meta["document_id"] == target
        ]
        if not doomed_ids:
            return

        self._index.remove_ids(np.asarray(doomed_ids, dtype=np.int64))
        for faiss_id in doomed_ids:
            chunk_id = self._metadata[faiss_id]["chunk_id"]
            self._metadata.pop(faiss_id, None)
            self._chunk_to_faiss_id.pop(chunk_id, None)

        self._persist()

    def _persist(self) -> None:
        assert self._index is not None

        # Atomic writes: write to .tmp in the same directory, then rename — prevents
        # a half-written index or metadata file from being loaded after a crash.
        index_tmp = tempfile.NamedTemporaryFile(
            delete=False,
            dir=self._index_path.parent,
            suffix=".tmp",
        )
        index_tmp.close()
        try:
            faiss.write_index(self._index, index_tmp.name)
            os.replace(index_tmp.name, self._index_path)
        except Exception:
            Path(index_tmp.name).unlink(missing_ok=True)
            raise

        payload = {
            "next_id": self._next_id,
            "metadata": {str(k): v for k, v in self._metadata.items()},
            "chunk_to_faiss_id": self._chunk_to_faiss_id,
        }
        metadata_tmp = tempfile.NamedTemporaryFile(
            mode="w",
            delete=False,
            dir=self._metadata_path.parent,
            suffix=".tmp",
            encoding="utf-8",
        )
        try:
            json.dump(payload, metadata_tmp)
            metadata_tmp.close()
            os.replace(metadata_tmp.name, self._metadata_path)
        except Exception:
            Path(metadata_tmp.name).unlink(missing_ok=True)
            raise
