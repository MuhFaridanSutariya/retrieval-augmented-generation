from pathlib import Path
from uuid import uuid4

import pytest

from app.core.config import Settings
from app.models.domain.chunk import Chunk
from app.storages.faiss_store import FaissStore


@pytest.fixture
def isolated_settings(tmp_path: Path, test_settings: Settings) -> Settings:
    data = test_settings.model_dump()
    data["faiss_index_path"] = str(tmp_path / "faiss.index")
    data["faiss_metadata_path"] = str(tmp_path / "metadata.json")
    data["openai_embedding_dimensions"] = 4
    return Settings(**data)


async def _fresh_store(settings: Settings) -> FaissStore:
    store = FaissStore(settings)
    await store.ensure_index()
    return store


def _make_chunk(document_id, index: int, embedding: list[float]) -> Chunk:
    return Chunk(
        id=f"{document_id}:{index}",
        document_id=document_id,
        chunk_index=index,
        text=f"chunk {index}",
        token_count=2,
        embedding=embedding,
    )


@pytest.mark.asyncio
async def test_upsert_then_query_returns_nearest(isolated_settings: Settings) -> None:
    store = await _fresh_store(isolated_settings)
    doc_id = uuid4()
    chunks = [
        _make_chunk(doc_id, 0, [1.0, 0.0, 0.0, 0.0]),
        _make_chunk(doc_id, 1, [0.0, 1.0, 0.0, 0.0]),
        _make_chunk(doc_id, 2, [0.0, 0.0, 1.0, 0.0]),
    ]
    await store.upsert_chunks(chunks)

    results = await store.query(embedding=[1.0, 0.0, 0.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0].chunk_index == 0


@pytest.mark.asyncio
async def test_query_filters_by_document_ids(isolated_settings: Settings) -> None:
    store = await _fresh_store(isolated_settings)
    doc_a, doc_b = uuid4(), uuid4()
    await store.upsert_chunks([
        _make_chunk(doc_a, 0, [1.0, 0.0, 0.0, 0.0]),
        _make_chunk(doc_b, 0, [1.0, 0.0, 0.0, 0.0]),
    ])

    results = await store.query(
        embedding=[1.0, 0.0, 0.0, 0.0],
        top_k=5,
        document_ids=[doc_b],
    )
    assert len(results) == 1
    assert results[0].document_id == doc_b


@pytest.mark.asyncio
async def test_delete_by_document_removes_only_target(isolated_settings: Settings) -> None:
    store = await _fresh_store(isolated_settings)
    doc_a, doc_b = uuid4(), uuid4()
    await store.upsert_chunks([
        _make_chunk(doc_a, 0, [1.0, 0.0, 0.0, 0.0]),
        _make_chunk(doc_b, 0, [0.0, 1.0, 0.0, 0.0]),
    ])

    await store.delete_by_document(doc_a)
    results = await store.query(embedding=[1.0, 0.0, 0.0, 0.0], top_k=5)
    assert len(results) == 1
    assert results[0].document_id == doc_b


@pytest.mark.asyncio
async def test_upsert_same_chunk_id_replaces_vector(isolated_settings: Settings) -> None:
    store = await _fresh_store(isolated_settings)
    doc_id = uuid4()
    await store.upsert_chunks([_make_chunk(doc_id, 0, [1.0, 0.0, 0.0, 0.0])])
    await store.upsert_chunks([_make_chunk(doc_id, 0, [0.0, 1.0, 0.0, 0.0])])

    results = await store.query(embedding=[0.0, 1.0, 0.0, 0.0], top_k=5)
    assert len(results) == 1
    assert results[0].chunk_index == 0


@pytest.mark.asyncio
async def test_state_persists_across_reload(isolated_settings: Settings) -> None:
    store = await _fresh_store(isolated_settings)
    doc_id = uuid4()
    await store.upsert_chunks([_make_chunk(doc_id, 0, [1.0, 0.0, 0.0, 0.0])])

    reloaded = await _fresh_store(isolated_settings)
    results = await reloaded.query(embedding=[1.0, 0.0, 0.0, 0.0], top_k=5)
    assert len(results) == 1
    assert results[0].document_id == doc_id
