import asyncio
from pathlib import Path
from uuid import UUID

from app.core.config import Settings
from app.core.exceptions import DocumentFileMissing
from app.core.logging import get_logger
from app.enums.file_type import FileType

logger = get_logger(__name__)


class FileStorage:
    def __init__(self, settings: Settings) -> None:
        self._base = Path(settings.upload_storage_path)
        self._base.mkdir(parents=True, exist_ok=True)

    def path_for(self, document_id: UUID, file_type: FileType) -> Path:
        return self._base / f"{document_id}.{file_type.value}"

    async def save(self, document_id: UUID, file_type: FileType, content: bytes) -> Path:
        target = self.path_for(document_id, file_type)
        await asyncio.to_thread(target.write_bytes, content)
        return target

    async def read(self, document_id: UUID, file_type: FileType) -> bytes:
        target = self.path_for(document_id, file_type)
        if not target.exists():
            raise DocumentFileMissing(
                f"Raw file for document {document_id} is not on disk.",
                details={"path": str(target)},
            )
        return await asyncio.to_thread(target.read_bytes)

    async def delete(self, document_id: UUID, file_type: FileType) -> None:
        target = self.path_for(document_id, file_type)
        try:
            await asyncio.to_thread(target.unlink, missing_ok=True)
        except OSError as exc:
            logger.warning("file_storage_delete_failed", path=str(target), error=str(exc))
