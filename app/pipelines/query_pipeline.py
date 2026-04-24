from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.exceptions import NoRelevantContext
from app.core.logging import get_logger
from app.llm_clients.openai_chat_client import ChatMessage, OpenAIChatClient
from app.models.domain.answer import Answer
from app.models.domain.chunk import RetrievedChunk
from app.models.mappers import retrieved_chunk_to_citation
from app.prompts.answer_with_context import (
    ANSWER_PROMPT_VERSION,
    build_user_prompt,
    is_refusal,
)
from app.prompts.system_prompt import SYSTEM_PROMPT, SYSTEM_PROMPT_VERSION
from app.retrievers.vector_retriever import VectorRetriever
from app.utils.token_counting import count_text_tokens, estimate_chat_cost_usd

logger = get_logger(__name__)


class QueryPipeline:
    def __init__(
        self,
        *,
        retriever: VectorRetriever,
        chat_client: OpenAIChatClient,
        settings: Settings,
    ) -> None:
        self._retriever = retriever
        self._chat_client = chat_client
        self._settings = settings

    async def run(
        self,
        *,
        question: str,
        session: AsyncSession,
        document_ids: list[UUID] | None = None,
        top_k: int | None = None,
    ) -> Answer:
        chunks = await self._retriever.retrieve(
            question=question,
            session=session,
            document_ids=document_ids,
            top_k=top_k,
        )

        if not chunks:
            raise NoRelevantContext(
                "No relevant context found for the question.",
                details={"document_ids": [str(d) for d in document_ids] if document_ids else None},
            )

        trimmed = self._trim_to_budget(chunks)

        messages = [
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=build_user_prompt(question, trimmed)),
        ]

        completion = await self._chat_client.complete(messages)

        refused = is_refusal(completion.content)
        citations = (
            []
            if refused
            else [retrieved_chunk_to_citation(chunk) for chunk in trimmed]
        )

        return Answer(
            text=completion.content.strip(),
            citations=citations,
            is_grounded=not refused,
            refusal_reason="no_relevant_context_in_corpus" if refused else None,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            estimated_cost_usd=estimate_chat_cost_usd(
                completion.prompt_tokens,
                completion.completion_tokens,
                self._settings,
            ),
            model=completion.model,
            prompt_version=f"system:{SYSTEM_PROMPT_VERSION}/user:{ANSWER_PROMPT_VERSION}",
            cache_hit=False,
        )

    def _trim_to_budget(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        ranked = sorted(chunks, key=lambda c: c.score, reverse=True)
        budget = self._settings.max_context_tokens - self._settings.openai_chat_max_output_tokens
        budget -= count_text_tokens(SYSTEM_PROMPT, self._settings.openai_chat_model)
        budget -= self._settings.token_budget_safety_pad

        kept: list[RetrievedChunk] = []
        used = 0
        for chunk in ranked:
            chunk_tokens = count_text_tokens(chunk.text, self._settings.openai_chat_model)
            if used + chunk_tokens > budget:
                break
            kept.append(chunk)
            used += chunk_tokens

        if not kept and ranked:
            kept.append(ranked[0])
        return kept
