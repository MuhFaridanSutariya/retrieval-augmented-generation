from app.models.domain.chunk import RetrievedChunk

ANSWER_PROMPT_VERSION = "v1"

_REFUSAL_MARKER = "I do not have enough information in the provided documents to answer that."


def build_user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return (
            f"CONTEXT:\n(no context available)\n\n"
            f"QUESTION:\n{question}\n\n"
            "Answer only from CONTEXT. If no context is available, respond with the refusal sentence."
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
        "Answer using only the CONTEXT above. Cite snippets as [S1], [S2], etc."
    )


def is_refusal(answer_text: str) -> bool:
    return _REFUSAL_MARKER.lower() in answer_text.strip().lower()