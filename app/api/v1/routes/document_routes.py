from uuid import UUID

from fastapi import APIRouter, Depends, File, Query, UploadFile, status
from fastapi.responses import Response

from app.dependencies import get_document_service
from app.models.mappers import document_domain_to_response
from app.models.schema.document_schema import (
    DocumentListResponse,
    DocumentResponse,
    DocumentUpdateRequest,
)
from app.services.document_service import DocumentService

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_document(
    file: UploadFile = File(...),
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentResponse:
    content = await file.read()
    filename = file.filename or "unnamed"
    document = await document_service.create(filename=filename, content=content)
    return document_domain_to_response(document)


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentListResponse:
    documents, total = await document_service.list(limit=limit, offset=offset)
    return DocumentListResponse(
        items=[document_domain_to_response(d) for d in documents],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: UUID,
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentResponse:
    document = await document_service.get(document_id)
    return document_domain_to_response(document)


@router.patch("/{document_id}", response_model=DocumentResponse)
async def update_document(
    document_id: UUID,
    payload: DocumentUpdateRequest,
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentResponse:
    if payload.filename is None:
        document = await document_service.get(document_id)
    else:
        document = await document_service.rename(document_id, payload.filename)
    return document_domain_to_response(document)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: UUID,
    document_service: DocumentService = Depends(get_document_service),
) -> Response:
    await document_service.delete(document_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
