import re
from dataclasses import dataclass

from app.models.domain.chunk import RetrievedChunk

ANSWER_PROMPT_VERSION = "v2"

REFUSAL_SENTENCE = (
    "I do not have enough information in the provided documents to answer that."
)

_THINKING_PATTERN = re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL | re.IGNORECASE)
_ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


@dataclass(slots=True)
class ParsedResponse:
    answer: str
    reasoning: str | None


def build_user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return (
            f"CONTEXT:\n(no context available)\n\n"
            f"QUESTION:\n{question}\n\n"
            "Reason step by step inside <thinking>, then emit the refusal sentence "
            "inside <answer>."
        )

    context_blocks: list[str] = []
    for position, chunk in enumerate(chunks, start=1):
        source = chunk.filename or str(chunk.document_id)
        context_blocks.append(
            f"[S{position}] source={source} chunk_index={chunk.chunk_index}\n{chunk.text.strip()}"
        )

    context = "\n\n".join(context_blocks)
    return (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{question}\n\n"
        "Reason step by step inside <thinking>, then give the final answer inside <answer>. "
        "Cite snippets as [S1], [S2], etc."
    )


def parse_response(raw: str) -> ParsedResponse:
    if not raw:
        return ParsedResponse(answer="", reasoning=None)

    thinking_match = _THINKING_PATTERN.search(raw)
    answer_match = _ANSWER_PATTERN.search(raw)

    reasoning = thinking_match.group(1).strip() if thinking_match else None

    if answer_match:
        answer = answer_match.group(1).strip()
    elif thinking_match:
        # Model emitted reasoning but forgot the <answer> tag — take everything after </thinking>.
        tail = raw[thinking_match.end() :].strip()
        answer = tail if tail else raw.strip()
    else:
        # No tags at all — treat the whole response as the answer (safety fallback).
        answer = raw.strip()

    return ParsedResponse(answer=answer, reasoning=reasoning)


def is_refusal(answer_text: str) -> bool:
    return REFUSAL_SENTENCE.lower() in answer_text.strip().lower()
