from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from app.enums.document_status import DocumentStatus
from app.enums.file_type import FileType


@dataclass(slots=True)
class Document:
    id: UUID
    filename: str
    file_type: FileType
    size_bytes: int
    status: DocumentStatus
    content_hash: str
    created_at: datetime
    updated_at: datetime
    chunk_count: int = 0
    error_message: str | None = None
