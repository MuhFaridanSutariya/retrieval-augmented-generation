from uuid import uuid4

from app.models.domain.chunk import RetrievedChunk
from app.retrievers.hybrid_retriever import _reciprocal_rank_fusion


def _chunk(chunk_id: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        document_id=uuid4(),
        chunk_index=0,
        text=chunk_id,
        score=score,
        filename=None,
    )


def test_rrf_prefers_chunks_appearing_in_both_lists() -> None:
    # Chunk "a" is ranked #1 in vector and #3 in BM25; "b" is #1 in BM25 only.
    vector = [_chunk("a", 0.9), _chunk("x", 0.7), _chunk("y", 0.6)]
    bm25 = [_chunk("b", 5.0), _chunk("c", 4.0), _chunk("a", 3.0)]

    fused = _reciprocal_rank_fusion([vector, bm25], rrf_k=60)
    fused_ids = [chunk.id for chunk in fused]
    assert fused_ids[0] == "a"


def test_rrf_deduplicates_by_id() -> None:
    vector = [_chunk("a", 0.9), _chunk("b", 0.7)]
    bm25 = [_chunk("a", 5.0), _chunk("b", 2.0)]

    fused = _reciprocal_rank_fusion([vector, bm25], rrf_k=60)
    assert len(fused) == 2


def test_rrf_keeps_best_snippet_for_duplicate_ids() -> None:
    chunk_id = "shared"
    low_score = RetrievedChunk(
        id=chunk_id, document_id=uuid4(), chunk_index=0, text="low", score=0.1
    )
    high_score = RetrievedChunk(
        id=chunk_id, document_id=uuid4(), chunk_index=0, text="high", score=0.9
    )
    fused = _reciprocal_rank_fusion([[low_score], [high_score]], rrf_k=60)
    assert fused[0].text == "high"


def test_rrf_stable_when_only_one_list_has_results() -> None:
    vector = [_chunk("a", 0.9), _chunk("b", 0.7)]
    fused = _reciprocal_rank_fusion([vector, []], rrf_k=60)
    assert [c.id for c in fused] == ["a", "b"]


def test_rrf_empty_inputs_produce_empty_output() -> None:
    assert _reciprocal_rank_fusion([[], []], rrf_k=60) == []
