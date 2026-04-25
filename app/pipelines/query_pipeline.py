import time
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.exceptions import NoRelevantContext
from app.core.logging import get_logger
from app.llm_clients.openai_chat_client import ChatMessage, OpenAIChatClient
from app.models.domain.answer import Answer, StageTimings
from app.models.domain.chunk import RetrievedChunk
from app.models.mappers import retrieved_chunk_to_citation
from app.prompts.answer_with_context import (
    build_user_prompt,
    is_refusal,
    parse_response,
    select_user_prompt_version,
)
from app.prompts.system_prompt import select_system_prompt
from app.retrievers.hybrid_retriever import HybridRetriever
from app.retrievers.reranker import LLMReranker
from app.utils.token_counting import count_text_tokens, estimate_chat_cost_usd

logger = get_logger(__name__)


class QueryPipeline:
    def __init__(
        self,
        *,
        hybrid_retriever: HybridRetriever,
        reranker: LLMReranker,
        chat_client: OpenAIChatClient,
        settings: Settings,
    ) -> None:
        self._retriever = hybrid_retriever
        self._reranker = reranker
        self._chat_client = chat_client
        self._settings = settings

    async def run(
        self,
        *,
        question: str,
        session: AsyncSession,
        document_ids: list[UUID] | None = None,
        top_k: int | None = None,
        use_cot: bool = False,
        use_rerank: bool | None = None,
    ) -> Answer:
        effective_rerank = (
            self._settings.rerank_enabled if use_rerank is None else use_rerank
        )
        effective_top_k = top_k or self._settings.retrieval_top_k
        timings = StageTimings()

        retrieve_started = time.perf_counter()
        chunks = await self._retriever.retrieve(
            question=question,
            session=session,
            document_ids=document_ids,
            top_k=effective_top_k,
        )
        timings.retrieve_ms = (time.perf_counter() - retrieve_started) * 1000

        if not chunks:
            raise NoRelevantContext(
                "No relevant context found for the question.",
                details={"document_ids": [str(d) for d in document_ids] if document_ids else None},
            )

        if effective_rerank:
            rerank_started = time.perf_counter()
            chunks = await self._reranker.rerank(
                question=question,
                chunks=chunks,
                top_k=self._settings.rerank_top_k,
            )
            timings.rerank_ms = (time.perf_counter() - rerank_started) * 1000

        if not chunks:
            raise NoRelevantContext(
                "Reranker returned no relevant snippets.",
                details={"document_ids": [str(d) for d in document_ids] if document_ids else None},
            )

        system_prompt, system_version = select_system_prompt(use_cot=use_cot)
        user_version = select_user_prompt_version(use_cot=use_cot)

        trimmed = self._trim_to_budget(chunks, system_prompt=system_prompt)

        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(
                role="user",
                content=build_user_prompt(question, trimmed, use_cot=use_cot),
            ),
        ]

        complete_started = time.perf_counter()
        completion = await self._chat_client.complete(messages)
        timings.complete_ms = (time.perf_counter() - complete_started) * 1000

        parsed = parse_response(completion.content)
        refused = is_refusal(parsed.answer)
        citations = (
            []
            if refused
            else [retrieved_chunk_to_citation(chunk) for chunk in trimmed]
        )

        if parsed.reasoning and self._settings.log_verbose:
            logger.debug("llm_reasoning", reasoning=parsed.reasoning)

        return Answer(
            text=parsed.answer,
            citations=citations,
            is_grounded=not refused,
            refusal_reason="no_relevant_context_in_corpus" if refused else None,
            reasoning=parsed.reasoning,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            estimated_cost_usd=estimate_chat_cost_usd(
                completion.prompt_tokens,
                completion.completion_tokens,
                self._settings,
            ),
            model=completion.model,
            prompt_version=(
                f"system:{system_version}/user:{user_version}"
                f"/rerank:{int(effective_rerank)}"
            ),
            cache_hit=False,
            timings=timings,
        )

    def _trim_to_budget(
        self,
        chunks: list[RetrievedChunk],
        *,
        system_prompt: str,
    ) -> list[RetrievedChunk]:
        ranked = sorted(chunks, key=lambda c: c.score, reverse=True)
        budget = self._settings.max_context_tokens - self._settings.openai_chat_max_output_tokens
        budget -= count_text_tokens(system_prompt, self._settings.openai_chat_model)
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
