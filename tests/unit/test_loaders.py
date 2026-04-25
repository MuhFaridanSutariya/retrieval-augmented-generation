import pytest

from app.core.exceptions import UnsupportedFileType
from app.enums.file_type import FileType
from app.loaders.document_loader import load_document
from app.loaders.text_loader import load_text


def test_load_text_utf8() -> None:
    assert load_text("hello world".encode("utf-8")) == "hello world"


def test_load_text_fixes_utf8_decoded_as_latin1_mojibake() -> None:
    mojibake_bytes = b"\xc3\xa2\xe2\x82\xac\xc5\x93hello\xc3\xa2\xe2\x82\xac\xc2\x9d"
    cleaned = load_text(mojibake_bytes)
    assert "â€" not in cleaned
    assert "hello" in cleaned


def test_load_text_handles_latin1_only_bytes() -> None:
    raw = b"caf\xe9"
    cleaned = load_text(raw)
    assert "caf" in cleaned


def test_load_text_empty_bytes() -> None:
    assert load_text(b"") == ""


def test_load_document_dispatches_text() -> None:
    assert "hi" in load_document(b"hi", FileType.TEXT)


def test_load_document_dispatches_markdown() -> None:
    assert "# title" in load_document(b"# title", FileType.MARKDOWN)


def test_load_document_unsupported_type_raises() -> None:
    class _Bogus:
        value = "bogus"

    with pytest.raises(UnsupportedFileType):
        load_document(b"x", _Bogus())  # type: ignore[arg-type]
