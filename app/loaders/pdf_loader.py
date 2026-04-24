import io

from pypdf import PdfReader

from app.core.exceptions import AppError


class PdfParseError(AppError):
    pass


def load_pdf(content: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(content))
    except Exception as exc:
        raise PdfParseError(f"Failed to open PDF: {exc}") from exc

    pages: list[str] = []
    for index, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            raise PdfParseError(f"Failed to extract text from page {index}: {exc}") from exc
        if text.strip():
            pages.append(text)

    return "\n\n".join(pages)
