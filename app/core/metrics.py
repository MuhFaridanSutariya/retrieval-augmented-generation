from dataclasses import dataclass, field
from decimal import Decimal

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class RequestMetrics:
    request_id: str
    query_hash: str
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: Decimal = Decimal("0")
    latency_ms: float = 0.0
    cache_hit: bool = False
    prompt_version: str | None = None
    model: str | None = None

    def emit(self) -> None:
        logger.info(
            "request_metrics",
            request_id=self.request_id,
            query_hash=self.query_hash,
            retrieved_chunk_ids=self.retrieved_chunk_ids,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            total_tokens=self.total_tokens,
            estimated_cost_usd=str(self.estimated_cost_usd),
            latency_ms=round(self.latency_ms, 2),
            cache_hit=self.cache_hit,
            prompt_version=self.prompt_version,
            model=self.model,
        )
