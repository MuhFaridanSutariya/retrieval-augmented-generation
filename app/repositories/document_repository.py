from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import DatabaseError, DocumentNotFound
from app.enums.document_status import DocumentStatus
from app.enums.file_type import FileType
from app.models.domain.document import Document
from app.models.mappers import document_orm_to_domain
from app.models.orm.document_orm import DocumentORM


class DocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        filename: str,
        file_type: FileType,
        size_bytes: int,
        content_hash: str,
        status: DocumentStatus = DocumentStatus.UPLOADED,
    ) -> Document:
        row = DocumentORM(
            filename=filename,
            file_type=file_type.value,
            size_bytes=size_bytes,
            content_hash=content_hash,
            status=status.value,
            chunk_count=0,
        )
        self._session.add(row)
        try:
            await self._session.flush()
            await self._session.refresh(row)
        except Exception as exc:
            raise DatabaseError(f"Failed to create document: {exc}") from exc
        return document_orm_to_domain(row)

    async def get(self, document_id: UUID) -> Document:
        result = await self._session.execute(
            select(DocumentORM).where(DocumentORM.id == document_id)
        )
        row = result.scalar_one_or_none()
        if row is None:
            raise DocumentNotFound(f"Document {document_id} not found.")
        return document_orm_to_domain(row)

    async def list(self, *, limit: int, offset: int) -> tuple[list[Document], int]:
        count_result = await self._session.execute(select(func.count(DocumentORM.id)))
        total = int(count_result.scalar_one())

        result = await self._session.execute(
            select(DocumentORM)
            .order_by(DocumentORM.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        rows = result.scalars().all()
        return [document_orm_to_domain(row) for row in rows], total

    async def update_status(
        self,
        document_id: UUID,
        status: DocumentStatus,
        *,
        chunk_count: int | None = None,
        error_message: str | None = None,
    ) -> Document:
        result = await self._session.execute(
            select(DocumentORM).where(DocumentORM.id == document_id)
        )
        row = result.scalar_one_or_none()
        if row is None:
            raise DocumentNotFound(f"Document {document_id} not found.")

        row.status = status.value
        if chunk_count is not None:
            row.chunk_count = chunk_count
        row.error_message = error_message

        try:
            await self._session.flush()
            await self._session.refresh(row)
        except Exception as exc:
            raise DatabaseError(f"Failed to update document: {exc}") from exc
        return document_orm_to_domain(row)

    async def update_filename(self, document_id: UUID, filename: str) -> Document:
        result = await self._session.execute(
            select(DocumentORM).where(DocumentORM.id == document_id)
        )
        row = result.scalar_one_or_none()
        if row is None:
            raise DocumentNotFound(f"Document {document_id} not found.")
        row.filename = filename
        try:
            await self._session.flush()
            await self._session.refresh(row)
        except Exception as exc:
            raise DatabaseError(f"Failed to update document: {exc}") from exc
        return document_orm_to_domain(row)

    async def delete(self, document_id: UUID) -> None:
        result = await self._session.execute(
            select(DocumentORM).where(DocumentORM.id == document_id)
        )
        row = result.scalar_one_or_none()
        if row is None:
            raise DocumentNotFound(f"Document {document_id} not found.")
        await self._session.delete(row)
