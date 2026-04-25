from app.enums.document_status import DocumentStatus
from app.enums.file_type import FileType
from app.models.domain.answer import Answer
from app.models.domain.chunk import RetrievedChunk
from app.models.domain.citation import Citation
from app.models.domain.document import Document
from app.models.orm.document_orm import DocumentORM
from app.models.schema.ask_schema import (
    AskResponse,
    CitationResponse,
    StageTimingsResponse,
    UsageResponse,
)
from app.models.schema.document_schema import DocumentResponse


def document_orm_to_domain(row: DocumentORM) -> Document:
    return Document(
        id=row.id,
        filename=row.filename,
        file_type=FileType(row.file_type),
        size_bytes=row.size_bytes,
        status=DocumentStatus(row.status),
        content_hash=row.content_hash,
        chunk_count=row.chunk_count,
        error_message=row.error_message,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def document_domain_to_response(document: Document) -> DocumentResponse:
    return DocumentResponse(
        id=document.id,
        filename=document.filename,
        file_type=document.file_type,
        size_bytes=document.size_bytes,
        status=document.status,
        chunk_count=document.chunk_count,
        content_hash=document.content_hash,
        error_message=document.error_message,
        created_at=document.created_at,
        updated_at=document.updated_at,
    )


def retrieved_chunk_to_citation(chunk: RetrievedChunk, snippet_chars: int = 240) -> Citation:
    snippet = chunk.text if len(chunk.text) <= snippet_chars else chunk.text[:snippet_chars] + "..."
    return Citation(
        chunk_id=chunk.id,
        document_id=chunk.document_id,
        filename=chunk.filename or "",
        chunk_index=chunk.chunk_index,
        score=chunk.score,
        snippet=snippet,
    )


def citation_to_response(citation: Citation) -> CitationResponse:
    return CitationResponse(
        chunk_id=citation.chunk_id,
        document_id=citation.document_id,
        filename=citation.filename,
        chunk_index=citation.chunk_index,
        score=citation.score,
        snippet=citation.snippet,
    )


def answer_to_response(answer: Answer, request_id: str) -> AskResponse:
    return AskResponse(
        answer=answer.text,
        is_grounded=answer.is_grounded,
        refusal_reason=answer.refusal_reason,
        citations=[citation_to_response(c) for c in answer.citations],
        usage=UsageResponse(
            prompt_tokens=answer.prompt_tokens,
            completion_tokens=answer.completion_tokens,
            total_tokens=answer.prompt_tokens + answer.completion_tokens,
            estimated_cost_usd=answer.estimated_cost_usd,
            model=answer.model,
            cache_hit=answer.cache_hit,
            timings=StageTimingsResponse(
                embed_ms=round(answer.timings.embed_ms, 1),
                retrieve_ms=round(answer.timings.retrieve_ms, 1),
                rerank_ms=round(answer.timings.rerank_ms, 1),
                complete_ms=round(answer.timings.complete_ms, 1),
                total_ms=round(answer.timings.total_ms, 1),
            ),
        ),
        request_id=request_id,
    )
