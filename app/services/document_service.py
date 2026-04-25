from uuid import UUID

from app.core.config import Settings
from app.core.logging import get_logger
from app.enums.document_status import DocumentStatus
from app.enums.file_type import FileType
from app.models.domain.document import Document
from app.pipelines.ingest_pipeline import IngestPipeline
from app.repositories.document_repository import DocumentRepository
from app.storages.database import Database
from app.storages.faiss_store import FaissStore
from app.storages.file_storage import FileStorage
from app.utils.hashing import sha256_bytes
from app.validators.upload_validator import validate_upload

logger = get_logger(__name__)


class DocumentService:
    def __init__(
        self,
        *,
        database: Database,
        vector_store: FaissStore,
        ingest_pipeline: IngestPipeline,
        file_storage: FileStorage,
        settings: Settings,
    ) -> None:
        self._database = database
        self._vector_store = vector_store
        self._ingest_pipeline = ingest_pipeline
        self._file_storage = file_storage
        self._settings = settings

    async def create(self, *, filename: str, content: bytes) -> Document:
        file_type = validate_upload(filename, len(content), self._settings)
        content_hash = sha256_bytes(content)

        async with self._database.session() as session:
            repository = DocumentRepository(session)
            document = await repository.create(
                filename=filename,
                file_type=file_type,
                size_bytes=len(content),
                content_hash=content_hash,
                status=DocumentStatus.INGESTING,
            )

        # Persist the original bytes so the FE can offer a preview link.
        # Failure here is non-fatal — ingestion can still proceed and the
        # document is queryable; the user just won't be able to download it.
        try:
            await self._file_storage.save(document.id, file_type, content)
        except OSError as exc:
            logger.warning(
                "raw_file_save_failed", document_id=str(document.id), error=str(exc)
            )

        try:
            chunk_count = await self._ingest_pipeline.run(
                document_id=document.id,
                file_type=file_type,
                content=content,
            )
        except Exception as exc:
            logger.exception("ingest_failed", document_id=str(document.id))
            async with self._database.session() as session:
                repository = DocumentRepository(session)
                return await repository.update_status(
                    document.id,
                    DocumentStatus.FAILED,
                    error_message=str(exc),
                )

        async with self._database.session() as session:
            repository = DocumentRepository(session)
            return await repository.update_status(
                document.id,
                DocumentStatus.READY,
                chunk_count=chunk_count,
                error_message=None,
            )

    async def get(self, document_id: UUID) -> Document:
        async with self._database.session() as session:
            repository = DocumentRepository(session)
            return await repository.get(document_id)

    async def list(self, *, limit: int, offset: int) -> tuple[list[Document], int]:
        async with self._database.session() as session:
            repository = DocumentRepository(session)
            return await repository.list(limit=limit, offset=offset)

    async def rename(self, document_id: UUID, filename: str) -> Document:
        async with self._database.session() as session:
            repository = DocumentRepository(session)
            return await repository.update_filename(document_id, filename)

    async def read_raw(self, document_id: UUID) -> tuple[bytes, FileType, str]:
        document = await self.get(document_id)
        content = await self._file_storage.read(document.id, document.file_type)
        return content, document.file_type, document.filename

    async def delete(self, document_id: UUID) -> None:
        # Delete vectors before the DB row — an orphan vector is recoverable via reconcile,
        # but an orphan DB row pointing to no vectors corrupts the /ask citation contract.
        document = await self.get(document_id)
        await self._vector_store.delete_by_document(document_id)
        await self._file_storage.delete(document_id, document.file_type)
        async with self._database.session() as session:
            repository = DocumentRepository(session)
            await repository.delete(document_id)
