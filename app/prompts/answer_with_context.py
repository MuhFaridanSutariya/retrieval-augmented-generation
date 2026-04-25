import re
from dataclasses import dataclass

from app.models.domain.chunk import RetrievedChunk

ANSWER_PROMPT_SIMPLE_VERSION = "simple-v4"
ANSWER_PROMPT_COT_VERSION = "cot-v3"

REFUSAL_SENTENCE = (
    "I do not have enough information in the provided documents to answer that."
)

_THINKING_PATTERN = re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL | re.IGNORECASE)
_ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
# Defensive cleanup: occasionally the model nests its own tags or includes a
# stray opening tag inside the captured block. Strip any residue so we never
# return literal "<answer>" text to the client.
_TAG_RESIDUE_PATTERN = re.compile(r"</?(?:answer|thinking)>", re.IGNORECASE)


@dataclass(slots=True)
class ParsedResponse:
    answer: str
    reasoning: str | None


def build_user_prompt(
    question: str,
    chunks: list[RetrievedChunk],
    *,
    use_cot: bool = False,
) -> str:
    if not chunks:
        if use_cot:
            tail = (
                "Reason step by step inside <thinking>, then emit the refusal sentence "
                "inside <answer>."
            )
        else:
            tail = (
                "Answer only from CONTEXT. If no context is available, respond with the "
                "refusal sentence."
            )
        return f"CONTEXT:\n(no context available)\n\nQUESTION:\n{question}\n\n{tail}"

    context_blocks: list[str] = []
    for position, chunk in enumerate(chunks, start=1):
        source = chunk.filename or str(chunk.document_id)
        context_blocks.append(
            f"[S{position}] source={source} chunk_index={chunk.chunk_index}\n{chunk.text.strip()}"
        )

    context = "\n\n".join(context_blocks)
    if use_cot:
        tail = (
            "Reason step by step inside <thinking>, then give the final answer inside "
            "<answer>. Cite snippets as [S1], [S2], etc."
        )
    else:
        tail = "Answer using only the CONTEXT above. Cite snippets as [S1], [S2], etc."
    return f"CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\n{tail}"


def select_user_prompt_version(*, use_cot: bool) -> str:
    return ANSWER_PROMPT_COT_VERSION if use_cot else ANSWER_PROMPT_SIMPLE_VERSION


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

    answer = _TAG_RESIDUE_PATTERN.sub("", answer).strip()
    if reasoning is not None:
        reasoning = _TAG_RESIDUE_PATTERN.sub("", reasoning).strip() or None

    return ParsedResponse(answer=answer, reasoning=reasoning)


def is_refusal(answer_text: str) -> bool:
    return REFUSAL_SENTENCE.lower() in answer_text.strip().lower()
