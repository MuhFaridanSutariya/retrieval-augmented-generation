from uuid import uuid4

from app.models.domain.chunk import RetrievedChunk
from app.prompts.answer_with_context import build_user_prompt, is_refusal
from app.prompts.system_prompt import SYSTEM_PROMPT


def test_system_prompt_contains_grounding_instructions() -> None:
    assert "only" in SYSTEM_PROMPT.lower()
    assert "CONTEXT" in SYSTEM_PROMPT
    assert "do not have enough information" in SYSTEM_PROMPT.lower()


def test_build_user_prompt_numbers_chunks_sequentially() -> None:
    chunks = [
        RetrievedChunk(
            id=f"{uuid4()}:0",
            document_id=uuid4(),
            chunk_index=0,
            text=f"Snippet number {i}.",
            score=0.9 - i * 0.1,
            filename=f"doc{i}.txt",
        )
        for i in range(3)
    ]
    prompt = build_user_prompt("What is X?", chunks)
    assert "[S1]" in prompt
    assert "[S2]" in prompt
    assert "[S3]" in prompt
    assert "What is X?" in prompt


def test_build_user_prompt_handles_empty_context() -> None:
    prompt = build_user_prompt("Tell me.", [])
    assert "no context available" in prompt.lower()
    assert "refusal" in prompt.lower()


def test_build_user_prompt_includes_source_filenames() -> None:
    chunk = RetrievedChunk(
        id="abc:0",
        document_id=uuid4(),
        chunk_index=0,
        text="Some fact.",
        score=0.9,
        filename="my_doc.pdf",
    )
    prompt = build_user_prompt("Q?", [chunk])
    assert "my_doc.pdf" in prompt


def test_is_refusal_detects_refusal_sentence() -> None:
    assert is_refusal("I do not have enough information in the provided documents to answer that.")
    assert is_refusal(
        "  I do not have enough information in the provided documents to answer that. "
    )


def test_is_refusal_returns_false_for_real_answer() -> None:
    assert not is_refusal("The answer is 42 [S1].")
    assert not is_refusal("I think...")
