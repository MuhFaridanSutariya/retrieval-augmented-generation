from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from app.enums.document_status import DocumentStatus
from app.enums.file_type import FileType


class DocumentResponse(BaseModel):
    id: UUID
    filename: str
    file_type: FileType
    size_bytes: int
    status: DocumentStatus
    chunk_count: int
    content_hash: str
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime


class DocumentListResponse(BaseModel):
    items: list[DocumentResponse]
    total: int
    limit: int
    offset: int


class DocumentUpdateRequest(BaseModel):
    filename: str | None = Field(default=None, min_length=1, max_length=512)
