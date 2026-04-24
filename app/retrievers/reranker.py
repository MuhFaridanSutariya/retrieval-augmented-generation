import json
import re

from app.core.config import Settings
from app.core.logging import get_logger
from app.llm_clients.openai_chat_client import ChatMessage, OpenAIChatClient
from app.models.domain.chunk import RetrievedChunk

logger = get_logger(__name__)

_JSON_ARRAY_PATTERN = re.compile(r"\[[\s\S]*?\]", re.MULTILINE)

_SYSTEM_PROMPT = (
    "You are a relevance-ranking assistant. Given a user's question and a numbered list "
    "of candidate text snippets, return a JSON array of the snippet numbers that are most "
    "relevant to answering the question, ordered from most to least relevant. "
    "Include only snippets that are directly relevant. Output ONLY the JSON array, no prose."
)


class LLMReranker:
    # Single-call listwise reranker. Cheaper and lower-latency than per-candidate scoring
    # because we reuse one LLM completion for the whole candidate list.

    def __init__(self, *, chat_client: OpenAIChatClient, settings: Settings) -> None:
        self._chat_client = chat_client
        self._settings = settings

    async def rerank(
        self,
        *,
        question: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        if len(chunks) <= top_k:
            return chunks

        user_prompt = self._build_prompt(question, chunks)

        try:
            completion = await self._chat_client.complete(
                [
                    ChatMessage(role="system", content=_SYSTEM_PROMPT),
                    ChatMessage(role="user", content=user_prompt),
                ],
                max_output_tokens=self._settings.rerank_max_output_tokens,
                temperature=0.0,
            )
        except Exception as exc:
            # If the reranker call fails, fall back to the fused ranking — degraded
            # quality is better than a failed /ask request.
            logger.warning("rerank_call_failed", error=str(exc))
            return chunks[:top_k]

        order = _parse_order(completion.content, max_index=len(chunks))
        if not order:
            logger.warning("rerank_parse_failed", raw_snippet=completion.content[:200])
            return chunks[:top_k]

        reranked: list[RetrievedChunk] = []
        seen: set[int] = set()
        for position in order:
            if position in seen:
                continue
            seen.add(position)
            reranked.append(chunks[position - 1])
            if len(reranked) >= top_k:
                break

        if len(reranked) < top_k:
            for index, chunk in enumerate(chunks, start=1):
                if index in seen:
                    continue
                reranked.append(chunk)
                if len(reranked) >= top_k:
                    break

        return reranked

    @staticmethod
    def _build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
        numbered = "\n\n".join(
            f"[{index}] {chunk.text.strip()}"
            for index, chunk in enumerate(chunks, start=1)
        )
        return (
            f"QUESTION:\n{question}\n\n"
            f"CANDIDATE SNIPPETS:\n{numbered}\n\n"
            "Return a JSON array of the snippet numbers in decreasing order of relevance. "
            f"Include only snippets actually relevant to the question."
        )


def _parse_order(raw: str, *, max_index: int) -> list[int]:
    if not raw:
        return []
    stripped = raw.strip()

    # Happy path: the model followed instructions and returned just a JSON array.
    # This also naturally rejects object wrappers like {"order": [1, 2]} because
    # those parse as dict, not list.
    parsed: object | None = None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        match = _JSON_ARRAY_PATTERN.search(stripped)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                parsed = None

    if not isinstance(parsed, list):
        return []

    result: list[int] = []
    for item in parsed:
        if isinstance(item, bool):
            continue
        if isinstance(item, int) and 1 <= item <= max_index:
            result.append(item)
    return result
