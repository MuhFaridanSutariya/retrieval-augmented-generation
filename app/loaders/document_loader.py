from app.core.exceptions import UnsupportedFileType
from app.enums.file_type import FileType
from app.loaders.pdf_loader import load_pdf
from app.loaders.text_loader import load_text


def load_document(content: bytes, file_type: FileType) -> str:
    if file_type is FileType.PDF:
        return load_pdf(content)
    if file_type in (FileType.TEXT, FileType.MARKDOWN):
        return load_text(content)
    raise UnsupportedFileType(f"No loader registered for file type {file_type}")
