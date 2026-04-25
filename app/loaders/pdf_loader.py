import io

import pdfplumber
from ftfy import fix_text
from pypdf import PdfReader

from app.core.exceptions import AppError
from app.core.logging import get_logger

logger = get_logger(__name__)


class PdfParseError(AppError):
    pass


def load_pdf(content: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(content))
    except Exception as exc:
        raise PdfParseError(f"Failed to open PDF: {exc}") from exc

    plumber_doc: pdfplumber.PDF | None = None
    plumber_pages: list = []
    try:
        plumber_doc = pdfplumber.open(io.BytesIO(content))
        plumber_pages = list(plumber_doc.pages)
    except Exception as exc:
        # Tables are a quality enhancement — if pdfplumber can't open the file we
        # still return prose via pypdf rather than failing the whole ingest.
        logger.warning("pdfplumber_open_failed", error=str(exc))

    pages: list[str] = []
    total_tables = 0
    try:
        for index, pypdf_page in enumerate(reader.pages):
            try:
                prose = pypdf_page.extract_text() or ""
            except Exception as exc:
                raise PdfParseError(f"Failed to extract text from page {index}: {exc}") from exc

            table_blocks: list[str] = []
            if index < len(plumber_pages):
                try:
                    raw_tables = plumber_pages[index].extract_tables() or []
                except Exception as exc:
                    logger.warning(
                        "pdfplumber_extract_failed", page=index, error=str(exc)
                    )
                    raw_tables = []
                for raw_table in raw_tables:
                    rendered = _table_to_markdown(raw_table)
                    if rendered:
                        table_blocks.append(rendered)

            total_tables += len(table_blocks)
            sections = [section for section in [prose.strip(), *table_blocks] if section]
            page_text = "\n\n".join(sections)
            if page_text:
                pages.append(fix_text(page_text))
    finally:
        if plumber_doc is not None:
            plumber_doc.close()

    if total_tables:
        logger.info("pdf_tables_extracted", count=total_tables, page_count=len(reader.pages))

    return "\n\n".join(pages)


def _table_to_markdown(table: list[list[str | None]]) -> str:
    if not table:
        return ""

    rows = [[_clean_cell(cell) for cell in row] for row in table]
    width = max((len(row) for row in rows), default=0)
    if width == 0:
        return ""

    rows = [row + [""] * (width - len(row)) for row in rows]
    if not any(any(cell for cell in row) for row in rows):
        return ""

    header = "| " + " | ".join(rows[0]) + " |"
    separator = "| " + " | ".join("---" for _ in range(width)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows[1:]]
    return "\n".join([header, separator, *body])


def _clean_cell(cell: str | None) -> str:
    if cell is None:
        return ""
    return cell.strip().replace("\n", " ").replace("|", "\\|")
