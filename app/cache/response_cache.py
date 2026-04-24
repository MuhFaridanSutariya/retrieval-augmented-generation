import json
from decimal import Decimal
from uuid import UUID

from app.core.config import Settings
from app.core.exceptions import CacheError
from app.core.logging import get_logger
from app.models.domain.answer import Answer
from app.models.domain.citation import Citation
from app.storages.redis_store import RedisStore
from app.utils.hashing import build_response_cache_key

logger = get_logger(__name__)


class ResponseCache:
    def __init__(self, redis_store: RedisStore, settings: Settings) -> None:
        self._redis = redis_store
        self._ttl = settings.response_cache_ttl_seconds

    async def get(
        self,
        *,
        question: str,
        document_ids: list[UUID] | None,
        model: str,
        prompt_version: str,
    ) -> Answer | None:
        key = build_response_cache_key(
            question=question,
            document_ids=document_ids,
            model=model,
            prompt_version=prompt_version,
        )
        try:
            raw = await self._redis.get(key)
        except CacheError:
            return None
        if not raw:
            return None
        try:
            return _deserialize_answer(raw)
        except (ValueError, KeyError) as exc:
            logger.warning("response_cache_corrupt", key=key, error=str(exc))
            return None

    async def set(
        self,
        *,
        question: str,
        document_ids: list[UUID] | None,
        model: str,
        prompt_version: str,
        answer: Answer,
    ) -> None:
        key = build_response_cache_key(
            question=question,
            document_ids=document_ids,
            model=model,
            prompt_version=prompt_version,
        )
        try:
            await self._redis.set(key, _serialize_answer(answer), ttl_seconds=self._ttl)
        except CacheError:
            pass


def _serialize_answer(answer: Answer) -> str:
    payload = {
        "text": answer.text,
        "is_grounded": answer.is_grounded,
        "refusal_reason": answer.refusal_reason,
        "reasoning": answer.reasoning,
        "prompt_tokens": answer.prompt_tokens,
        "completion_tokens": answer.completion_tokens,
        "estimated_cost_usd": str(answer.estimated_cost_usd),
        "model": answer.model,
        "prompt_version": answer.prompt_version,
        "citations": [
            {
                "chunk_id": c.chunk_id,
                "document_id": str(c.document_id),
                "filename": c.filename,
                "chunk_index": c.chunk_index,
                "score": c.score,
                "snippet": c.snippet,
            }
            for c in answer.citations
        ],
    }
    return json.dumps(payload)


def _deserialize_answer(raw: str) -> Answer:
    payload = json.loads(raw)
    citations = [
        Citation(
            chunk_id=c["chunk_id"],
            document_id=UUID(c["document_id"]),
            filename=c["filename"],
            chunk_index=c["chunk_index"],
            score=c["score"],
            snippet=c["snippet"],
        )
        for c in payload.get("citations", [])
    ]
    return Answer(
        text=payload["text"],
        citations=citations,
        is_grounded=payload["is_grounded"],
        refusal_reason=payload.get("refusal_reason"),
        reasoning=payload.get("reasoning"),
        prompt_tokens=payload["prompt_tokens"],
        completion_tokens=payload["completion_tokens"],
        estimated_cost_usd=Decimal(payload["estimated_cost_usd"]),
        model=payload["model"],
        prompt_version=payload["prompt_version"],
        cache_hit=True,
    )
