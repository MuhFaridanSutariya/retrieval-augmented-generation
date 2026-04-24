from pathlib import Path
from uuid import uuid4

import pytest

from app.core.config import Settings
from app.models.domain.chunk import Chunk
from app.retrievers.bm25_retriever import BM25Retriever, _tokenize
from app.storages.faiss_store import FaissStore

# BM25 IDF becomes degenerate on tiny corpora (a term appearing in 1 of 2 docs scores 0),
# so the seed below uses eight diverse chunks — enough for realistic IDF values.
_CORPUS = [
    "FastAPI is a modern Python web framework for building APIs with async support.",
    "Redis is an in-memory key-value data store used as a cache and message broker.",
    "PostgreSQL is a relational database supporting JSON, full-text search, and extensions.",
    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris.",
    "Mount Everest is the Earth's highest mountain, located in the Himalayas on the border of Nepal.",
    "Photosynthesis converts light energy into chemical energy stored in glucose molecules.",
    "Neural networks learn by backpropagation, adjusting weights to minimise loss.",
    "Git is a distributed version control system tracking changes to source code over time.",
]


def test_tokenize_splits_and_lowercases() -> None:
    assert _tokenize("Hello, WORLD! 123") == ["hello", "world", "123"]


def test_tokenize_handles_empty_input() -> None:
    assert _tokenize("") == []


@pytest.fixture
def isolated_settings(tmp_path: Path, test_settings: Settings) -> Settings:
    data = test_settings.model_dump()
    data["faiss_index_path"] = str(tmp_path / "faiss.index")
    data["faiss_metadata_path"] = str(tmp_path / "metadata.json")
    data["openai_embedding_dimensions"] = 4
    return Settings(**data)


async def _seed_corpus(
    settings: Settings,
    *,
    texts: list[str],
    document_id=None,
) -> tuple[FaissStore, list[str]]:
    store = FaissStore(settings)
    await store.ensure_index()

    doc_id = document_id or uuid4()
    chunk_ids: list[str] = []
    chunks: list[Chunk] = []
    for index, text in enumerate(texts):
        chunk_id = f"{doc_id}:{index}"
        chunk_ids.append(chunk_id)
        chunks.append(
            Chunk(
                id=chunk_id,
                document_id=doc_id,
                chunk_index=index,
                text=text,
                token_count=len(text.split()),
                embedding=[float(i == index % 4) for i in range(4)],
            )
        )
    await store.upsert_chunks(chunks)
    return store, chunk_ids


@pytest.mark.asyncio
async def test_bm25_retrieves_by_keyword_match(isolated_settings: Settings) -> None:
    store, _ = await _seed_corpus(isolated_settings, texts=_CORPUS)
    bm25 = BM25Retriever(faiss_store=store, settings=isolated_settings)

    results = await bm25.retrieve(question="Python web framework FastAPI", top_k=5)
    assert results
    assert "FastAPI" in results[0].text


@pytest.mark.asyncio
async def test_bm25_returns_empty_for_unrelated_query(isolated_settings: Settings) -> None:
    store, _ = await _seed_corpus(isolated_settings, texts=_CORPUS)
    bm25 = BM25Retriever(faiss_store=store, settings=isolated_settings)

    results = await bm25.retrieve(question="quantum chromodynamics thermodynamics", top_k=5)
    assert results == []


@pytest.mark.asyncio
async def test_bm25_respects_document_filter(isolated_settings: Settings) -> None:
    store = FaissStore(isolated_settings)
    await store.ensure_index()

    doc_a = uuid4()
    doc_b = uuid4()
    chunks: list[Chunk] = []
    for index, text in enumerate(_CORPUS[:4]):
        chunks.append(
            Chunk(
                id=f"{doc_a}:{index}",
                document_id=doc_a,
                chunk_index=index,
                text=text,
                token_count=len(text.split()),
                embedding=[float(i == index % 4) for i in range(4)],
            )
        )
    for index, text in enumerate(_CORPUS[4:]):
        chunks.append(
            Chunk(
                id=f"{doc_b}:{index}",
                document_id=doc_b,
                chunk_index=index,
                text=text,
                token_count=len(text.split()),
                embedding=[float(i == index % 4) for i in range(4)],
            )
        )
    await store.upsert_chunks(chunks)

    bm25 = BM25Retriever(faiss_store=store, settings=isolated_settings)
    results = await bm25.retrieve(
        question="mountain Everest Nepal",
        top_k=5,
        document_ids=[doc_b],
    )
    assert results
    assert all(chunk.document_id == doc_b for chunk in results)


@pytest.mark.asyncio
async def test_bm25_rebuilds_when_corpus_changes(isolated_settings: Settings) -> None:
    store, _ = await _seed_corpus(isolated_settings, texts=_CORPUS)
    bm25 = BM25Retriever(faiss_store=store, settings=isolated_settings)

    first = await bm25.retrieve(question="FastAPI Python", top_k=5)
    assert first

    new_doc = uuid4()
    await store.upsert_chunks([
        Chunk(
            id=f"{new_doc}:0",
            document_id=new_doc,
            chunk_index=0,
            text="Pineapple is a tropical fruit native to South America.",
            token_count=10,
            embedding=[0.5, 0.5, 0.0, 0.0],
        ),
    ])

    second = await bm25.retrieve(question="pineapple tropical", top_k=5)
    assert second
    assert "Pineapple" in second[0].text
