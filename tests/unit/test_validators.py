import pytest

from app.core.config import Settings
from app.core.exceptions import (
    EmptyQuery,
    QueryTooLong,
    UnsupportedFileType,
    UploadTooLarge,
)
from app.enums.file_type import FileType
from app.validators.query_validator import MAX_QUESTION_LENGTH, validate_question
from app.validators.upload_validator import validate_upload


def test_validate_question_strips_whitespace() -> None:
    assert validate_question("  what is it?  ") == "what is it?"


def test_validate_question_rejects_empty() -> None:
    with pytest.raises(EmptyQuery):
        validate_question("")


def test_validate_question_rejects_whitespace_only() -> None:
    with pytest.raises(EmptyQuery):
        validate_question("   \n\t  ")


def test_validate_question_rejects_none() -> None:
    with pytest.raises(EmptyQuery):
        validate_question(None)  # type: ignore[arg-type]


def test_validate_question_rejects_too_long() -> None:
    with pytest.raises(QueryTooLong):
        validate_question("a" * (MAX_QUESTION_LENGTH + 1))


def test_validate_upload_accepts_pdf(test_settings: Settings) -> None:
    assert validate_upload("doc.pdf", 1024, test_settings) is FileType.PDF


def test_validate_upload_accepts_markdown(test_settings: Settings) -> None:
    assert validate_upload("doc.md", 1024, test_settings) is FileType.MARKDOWN


def test_validate_upload_accepts_text(test_settings: Settings) -> None:
    assert validate_upload("doc.txt", 1024, test_settings) is FileType.TEXT


def test_validate_upload_rejects_unknown_extension(test_settings: Settings) -> None:
    with pytest.raises(UnsupportedFileType):
        validate_upload("doc.exe", 1024, test_settings)


def test_validate_upload_rejects_oversize(test_settings: Settings) -> None:
    with pytest.raises(UploadTooLarge):
        validate_upload("doc.pdf", test_settings.upload_max_bytes + 1, test_settings)
