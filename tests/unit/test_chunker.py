import pytest

from app.chunkers.recursive_splitter import RecursiveSplitter
from app.core.config import Settings


@pytest.fixture
def splitter(test_settings: Settings) -> RecursiveSplitter:
    return RecursiveSplitter(test_settings)


def test_split_empty_text_returns_empty(splitter: RecursiveSplitter) -> None:
    assert splitter.split("") == []
    assert splitter.split("   \n\n  ") == []


def test_split_short_text_produces_single_chunk(splitter: RecursiveSplitter) -> None:
    chunks = splitter.split("One short sentence.")
    assert len(chunks) == 1
    assert chunks[0].text == "One short sentence."
    assert chunks[0].index == 0
    assert chunks[0].token_count > 0


def test_split_long_text_produces_multiple_chunks(
    splitter: RecursiveSplitter,
    test_settings: Settings,
) -> None:
    paragraph = "This is a sentence that will be repeated many times. " * 200
    chunks = splitter.split(paragraph)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.token_count <= test_settings.chunk_size_tokens


def test_split_preserves_ordering_of_indexes(splitter: RecursiveSplitter) -> None:
    text = "\n\n".join(f"Paragraph {i} with enough filler text to matter." * 20 for i in range(10))
    chunks = splitter.split(text)
    indexes = [c.index for c in chunks]
    assert indexes == list(range(len(chunks)))


def test_split_covers_full_content(splitter: RecursiveSplitter) -> None:
    original = "first paragraph. second paragraph. third paragraph."
    chunks = splitter.split(original)
    assert any("first" in c.text for c in chunks)
    assert any("third" in c.text for c in chunks)
