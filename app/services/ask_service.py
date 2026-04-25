import time
from decimal import Decimal
from uuid import UUID, uuid4

from app.cache.response_cache import ResponseCache
from app.core.config import Settings
from app.core.exceptions import NoRelevantContext
from app.core.logging import get_logger
from app.core.metrics import RequestMetrics
from app.embedders.openai_embedder import OpenAIEmbedder
from app.enums.intent import Intent
from app.models.domain.answer import Answer
from app.pipelines.query_pipeline import QueryPipeline
from app.prompts.answer_with_context import select_user_prompt_version
from app.prompts.static_responses import (
    FAREWELL_RESPONSE,
    GREETING_RESPONSE,
    OFF_TOPIC_RESPONSE,
    STATIC_RESPONSE_VERSION,
)
from app.prompts.system_prompt import select_system_prompt
from app.storages.database import Database
from app.utils.hashing import sha256_hex
from app.validators.intent_classifier import IntentClassifier, fast_path_classify
from app.validators.query_validator import validate_question

logger = get_logger(__name__)


class AskResult:
    __slots__ = ("answer", "request_id")

    def __init__(self, answer: Answer, request_id: str) -> None:
        self.answer = answer
        self.request_id = request_id


class AskService:
    def __init__(
        self,
        *,
        database: Database,
        query_pipeline: QueryPipeline,
        response_cache: ResponseCache,
        embedder: OpenAIEmbedder,
        intent_classifier: IntentClassifier,
        settings: Settings,
    ) -> None:
        self._database = database
        self._query_pipeline = query_pipeline
        self._response_cache = response_cache
        self._embedder = embedder
        self._intent_classifier = intent_classifier
        self._settings = settings

    async def ask(
        self,
        *,
        question: str,
        document_ids: list[UUID] | None = None,
        top_k: int | None = None,
        use_cot: bool = False,
        use_rerank: bool | None = None,
        use_tools: bool = False,
    ) -> AskResult:
        effective_rerank = (
            self._settings.rerank_enabled if use_rerank is None else use_rerank
        )
        # Cache key includes the prompt-variant, rerank flag, and tools flag so all
        # eight combinations of (CoT, rerank, tools) on/off don't collide and
        # toggling any one of them invalidates the cache as expected.
        _, system_version = select_system_prompt(use_cot=use_cot)
        user_version = select_user_prompt_version(use_cot=use_cot)
        prompt_version = (
            f"system:{system_version}/user:{user_version}"
            f"/rerank:{int(effective_rerank)}/tools:{int(use_tools)}"
        )
        validated_question = validate_question(question)
        request_id = uuid4().hex
        started = time.perf_counter()
        query_hash = sha256_hex(validated_question.lower())[:16]

        # Layer 1: literal-set fast-path. Catches the most common greetings/farewells
        # (e.g. "halo", "thanks bye") with zero network calls. Anything not in the set
        # falls through to the embedding classifier so flexibility is preserved.
        fast_intent = fast_path_classify(validated_question)
        if fast_intent is not None:
            answer = self._static_answer(fast_intent)
            answer.timings.total_ms = (time.perf_counter() - started) * 1000
            self._emit_metrics(
                request_id=request_id,
                query_hash=query_hash,
                answer=answer,
                started=started,
                intent=fast_intent,
            )
            return AskResult(answer, request_id)

        # Layer 2: embedding classifier. Embed once, reuse for retrieval via the cache.
        embed_started = time.perf_counter()
        query_embedding = await self._embedder.embed_single(validated_question)
        embed_ms = (time.perf_counter() - embed_started) * 1000
        intent = await self._intent_classifier.classify(query_embedding)

        if intent is not Intent.RAG_QUERY:
            answer = self._static_answer(intent)
            answer.timings.embed_ms = embed_ms
            answer.timings.total_ms = (time.perf_counter() - started) * 1000
            self._emit_metrics(
                request_id=request_id,
                query_hash=query_hash,
                answer=answer,
                started=started,
                intent=intent,
            )
            return AskResult(answer, request_id)

        cached = await self._response_cache.get(
            question=validated_question,
            document_ids=document_ids,
            model=self._settings.openai_chat_model,
            prompt_version=prompt_version,
        )
        if cached is not None:
            cached.timings.embed_ms = embed_ms
            cached.timings.total_ms = (time.perf_counter() - started) * 1000
            self._emit_metrics(
                request_id=request_id,
                query_hash=query_hash,
                answer=cached,
                started=started,
                intent=intent,
            )
            return AskResult(cached, request_id)

        try:
            async with self._database.session() as session:
                answer = await self._query_pipeline.run(
                    question=validated_question,
                    session=session,
                    document_ids=document_ids,
                    top_k=top_k,
                    use_cot=use_cot,
                    use_rerank=use_rerank,
                    use_tools=use_tools,
                )
        except NoRelevantContext:
            answer = self._off_topic_answer()
            answer.timings.embed_ms = embed_ms
            answer.timings.total_ms = (time.perf_counter() - started) * 1000
            self._emit_metrics(
                request_id=request_id,
                query_hash=query_hash,
                answer=answer,
                started=started,
                intent=intent,
            )
            return AskResult(answer, request_id)

        answer.timings.embed_ms = embed_ms
        answer.timings.total_ms = (time.perf_counter() - started) * 1000

        if answer.is_grounded:
            await self._response_cache.set(
                question=validated_question,
                document_ids=document_ids,
                model=self._settings.openai_chat_model,
                prompt_version=prompt_version,
                answer=answer,
            )

        self._emit_metrics(
            request_id=request_id,
            query_hash=query_hash,
            answer=answer,
            started=started,
            intent=intent,
        )
        return AskResult(answer, request_id)

    def _static_answer(self, intent: Intent) -> Answer:
        text = GREETING_RESPONSE if intent is Intent.GREETING else FAREWELL_RESPONSE
        return Answer(
            text=text,
            is_grounded=True,
            refusal_reason=None,
            estimated_cost_usd=Decimal("0"),
            model="static",
            prompt_version=f"static:{STATIC_RESPONSE_VERSION}",
        )

    def _off_topic_answer(self) -> Answer:
        return Answer(
            text=OFF_TOPIC_RESPONSE,
            is_grounded=False,
            refusal_reason="off_topic",
            estimated_cost_usd=Decimal("0"),
            model="static",
            prompt_version=f"static:{STATIC_RESPONSE_VERSION}",
        )

    def _emit_metrics(
        self,
        *,
        request_id: str,
        query_hash: str,
        answer: Answer,
        started: float,
        intent: Intent,
    ) -> None:
        metrics = RequestMetrics(
            request_id=request_id,
            query_hash=query_hash,
            retrieved_chunk_ids=[c.chunk_id for c in answer.citations],
            prompt_tokens=answer.prompt_tokens,
            completion_tokens=answer.completion_tokens,
            total_tokens=answer.prompt_tokens + answer.completion_tokens,
            estimated_cost_usd=answer.estimated_cost_usd,
            latency_ms=(time.perf_counter() - started) * 1000,
            cache_hit=answer.cache_hit,
            prompt_version=answer.prompt_version,
            model=answer.model or self._settings.openai_chat_model,
        )
        metrics.emit()
        logger.info("ask_intent", intent=str(intent), request_id=request_id)
