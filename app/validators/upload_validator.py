from pathlib import Path

from app.core.config import Settings
from app.core.exceptions import UnsupportedFileType, UploadTooLarge
from app.enums.file_type import FileType


def validate_upload(filename: str, size_bytes: int, settings: Settings) -> FileType:
    if size_bytes > settings.upload_max_bytes:
        raise UploadTooLarge(
            f"Upload exceeds maximum size of {settings.upload_max_bytes} bytes.",
            details={"size": size_bytes, "max": settings.upload_max_bytes},
        )

    extension = Path(filename).suffix.lower().lstrip(".")
    if extension not in settings.allowed_extensions:
        raise UnsupportedFileType(
            f"File type .{extension} is not supported.",
            details={"extension": extension, "allowed": sorted(settings.allowed_extensions)},
        )

    return FileType.from_extension(extension)
