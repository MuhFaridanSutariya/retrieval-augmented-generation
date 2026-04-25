from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

from app.enums.document_status import DocumentStatus
from app.enums.file_type import FileType
from app.models.domain.answer import Answer, StageTimings
from app.models.domain.chunk import RetrievedChunk
from app.models.domain.citation import Citation
from app.models.domain.document import Document
from app.models.mappers import (
    answer_to_response,
    document_domain_to_response,
    document_orm_to_domain,
    retrieved_chunk_to_citation,
)
from app.models.orm.document_orm import DocumentORM


def _make_orm() -> DocumentORM:
    row = DocumentORM()
    row.id = uuid4()
    row.filename = "guide.pdf"
    row.file_type = FileType.PDF.value
    row.size_bytes = 1024
    row.status = DocumentStatus.READY.value
    row.content_hash = "abc123"
    row.chunk_count = 12
    row.error_message = None
    row.created_at = datetime.now(UTC)
    row.updated_at = datetime.now(UTC)
    return row


def test_document_orm_to_domain_preserves_fields() -> None:
    row = _make_orm()
    document = document_orm_to_domain(row)
    assert document.id == row.id
    assert document.filename == row.filename
    assert document.file_type is FileType.PDF
    assert document.status is DocumentStatus.READY
    assert document.chunk_count == 12


def test_document_domain_to_response_roundtrip() -> None:
    row = _make_orm()
    domain = document_orm_to_domain(row)
    response = document_domain_to_response(domain)
    assert response.id == domain.id
    assert response.filename == domain.filename
    assert response.file_type is FileType.PDF
    assert response.status is DocumentStatus.READY


def test_retrieved_chunk_to_citation_truncates_long_snippets() -> None:
    chunk = RetrievedChunk(
        id="x:0",
        document_id=uuid4(),
        chunk_index=0,
        text="a" * 500,
        score=0.9,
        filename="doc.txt",
    )
    citation = retrieved_chunk_to_citation(chunk, snippet_chars=100)
    assert citation.snippet.endswith("...")
    assert len(citation.snippet) <= 104


def test_retrieved_chunk_to_citation_keeps_short_text() -> None:
    chunk = RetrievedChunk(
        id="x:0",
        document_id=uuid4(),
        chunk_index=0,
        text="short",
        score=0.9,
        filename="doc.txt",
    )
    citation = retrieved_chunk_to_citation(chunk)
    assert citation.snippet == "short"


def test_answer_to_response_maps_usage_and_request_id() -> None:
    citation = Citation(
        chunk_id="x:0",
        document_id=uuid4(),
        filename="doc.txt",
        chunk_index=0,
        score=0.9,
        snippet="snippet",
    )
    answer = Answer(
        text="The answer is 42 [S1].",
        citations=[citation],
        is_grounded=True,
        prompt_tokens=100,
        completion_tokens=20,
        estimated_cost_usd=Decimal("0.016"),
        model="gpt-5.4",
        prompt_version="system:v1/user:v1",
        cache_hit=False,
    )
    response = answer_to_response(answer, request_id="req-123")
    assert response.answer == "The answer is 42 [S1]."
    assert response.usage.prompt_tokens == 100
    assert response.usage.completion_tokens == 20
    assert response.usage.total_tokens == 120
    assert response.usage.estimated_cost_usd == Decimal("0.016")
    assert response.request_id == "req-123"
    assert len(response.citations) == 1


def test_answer_to_response_includes_stage_timings() -> None:
    answer = Answer(
        text="x",
        is_grounded=True,
        prompt_tokens=10,
        completion_tokens=5,
        estimated_cost_usd=Decimal("0"),
        model="gpt-5.4",
        prompt_version="system:simple-v4/user:simple-v4/rerank:1",
        cache_hit=False,
        timings=StageTimings(
            embed_ms=812.34,
            retrieve_ms=4.5,
            rerank_ms=10456.7,
            complete_ms=1234.56,
            total_ms=12508.1,
        ),
    )
    response = answer_to_response(answer, request_id="r1")
    assert response.usage.timings.embed_ms == 812.3
    assert response.usage.timings.retrieve_ms == 4.5
    assert response.usage.timings.rerank_ms == 10456.7
    assert response.usage.timings.complete_ms == 1234.6
    assert response.usage.timings.total_ms == 12508.1


def test_answer_to_response_defaults_timings_to_zero() -> None:
    answer = Answer(
        text="hi",
        is_grounded=True,
        prompt_tokens=0,
        completion_tokens=0,
        estimated_cost_usd=Decimal("0"),
        model="static",
        prompt_version="static:v1",
        cache_hit=False,
    )
    response = answer_to_response(answer, request_id="r2")
    assert response.usage.timings.total_ms == 0.0
    assert response.usage.timings.rerank_ms == 0.0
