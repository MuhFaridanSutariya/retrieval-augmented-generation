import time
from uuid import UUID, uuid4

from app.cache.response_cache import ResponseCache
from app.core.config import Settings
from app.core.logging import get_logger
from app.core.metrics import RequestMetrics
from app.models.domain.answer import Answer
from app.pipelines.query_pipeline import QueryPipeline
from app.prompts.answer_with_context import ANSWER_PROMPT_VERSION
from app.prompts.system_prompt import SYSTEM_PROMPT_VERSION
from app.storages.database import Database
from app.utils.hashing import sha256_hex
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
        settings: Settings,
    ) -> None:
        self._database = database
        self._query_pipeline = query_pipeline
        self._response_cache = response_cache
        self._settings = settings
        self._prompt_version = f"system:{SYSTEM_PROMPT_VERSION}/user:{ANSWER_PROMPT_VERSION}"

    async def ask(
        self,
        *,
        question: str,
        document_ids: list[UUID] | None = None,
        top_k: int | None = None,
    ) -> AskResult:
        validated_question = validate_question(question)
        request_id = uuid4().hex
        started = time.perf_counter()
        query_hash = sha256_hex(validated_question.lower())[:16]

        cached = await self._response_cache.get(
            question=validated_question,
            document_ids=document_ids,
            model=self._settings.openai_chat_model,
            prompt_version=self._prompt_version,
        )
        if cached is not None:
            self._emit_metrics(
                request_id=request_id,
                query_hash=query_hash,
                answer=cached,
                started=started,
            )
            return AskResult(cached, request_id)

        async with self._database.session() as session:
            answer = await self._query_pipeline.run(
                question=validated_question,
                session=session,
                document_ids=document_ids,
                top_k=top_k,
            )

        if answer.is_grounded:
            await self._response_cache.set(
                question=validated_question,
                document_ids=document_ids,
                model=self._settings.openai_chat_model,
                prompt_version=self._prompt_version,
                answer=answer,
            )

        self._emit_metrics(
            request_id=request_id,
            query_hash=query_hash,
            answer=answer,
            started=started,
        )
        return AskResult(answer, request_id)

    def _emit_metrics(
        self,
        *,
        request_id: str,
        query_hash: str,
        answer: Answer,
        started: float,
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
