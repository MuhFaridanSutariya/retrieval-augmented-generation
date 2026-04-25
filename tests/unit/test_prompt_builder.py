from uuid import uuid4

from app.models.domain.chunk import RetrievedChunk
from app.prompts.answer_with_context import (
    build_user_prompt,
    is_refusal,
    parse_response,
)
from app.prompts.system_prompt import SYSTEM_PROMPT


def test_system_prompt_contains_grounding_instructions() -> None:
    assert "only" in SYSTEM_PROMPT.lower()
    assert "CONTEXT" in SYSTEM_PROMPT
    assert "do not have enough information" in SYSTEM_PROMPT.lower()


def test_system_prompt_requests_chain_of_thought_tags() -> None:
    assert "<thinking>" in SYSTEM_PROMPT
    assert "</thinking>" in SYSTEM_PROMPT
    assert "<answer>" in SYSTEM_PROMPT
    assert "</answer>" in SYSTEM_PROMPT


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
    assert "<thinking>" in prompt or "thinking" in prompt.lower()


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


def test_parse_response_extracts_answer_and_reasoning() -> None:
    raw = (
        "<thinking>\nStep 1. The question asks about X.\n"
        "Step 2. S1 mentions X directly.\n</thinking>\n"
        "<answer>\nX is Y [S1].\n</answer>"
    )
    parsed = parse_response(raw)
    assert parsed.answer == "X is Y [S1]."
    assert parsed.reasoning is not None
    assert "Step 1" in parsed.reasoning


def test_parse_response_handles_missing_thinking_block() -> None:
    parsed = parse_response("<answer>Just the answer [S1].</answer>")
    assert parsed.answer == "Just the answer [S1]."
    assert parsed.reasoning is None


def test_parse_response_falls_back_when_no_tags() -> None:
    parsed = parse_response("X is Y.")
    assert parsed.answer == "X is Y."
    assert parsed.reasoning is None


def test_parse_response_handles_forgotten_answer_tag() -> None:
    raw = "<thinking>Reasoning here.</thinking>\nFinal sentence."
    parsed = parse_response(raw)
    assert parsed.answer == "Final sentence."
    assert parsed.reasoning == "Reasoning here."


def test_parse_response_empty_string() -> None:
    parsed = parse_response("")
    assert parsed.answer == ""
    assert parsed.reasoning is None


def test_parse_response_strips_nested_answer_tags() -> None:
    raw = (
        "<thinking>reasoning</thinking>\n"
        "<answer>\n<answer>The real answer [S1].</answer>\n</answer>"
    )
    parsed = parse_response(raw)
    assert "<answer>" not in parsed.answer
    assert "</answer>" not in parsed.answer
    assert "The real answer [S1]." in parsed.answer


def test_parse_response_strips_residue_when_only_one_set_of_tags() -> None:
    raw = "<answer>Final [S1].</answer>"
    parsed = parse_response(raw)
    assert parsed.answer == "Final [S1]."


def test_is_refusal_detects_refusal_sentence() -> None:
    assert is_refusal("I do not have enough information in the provided documents to answer that.")
    assert is_refusal(
        "  I do not have enough information in the provided documents to answer that. "
    )


def test_is_refusal_returns_false_for_real_answer() -> None:
    assert not is_refusal("The answer is 42 [S1].")
    assert not is_refusal("I think...")
