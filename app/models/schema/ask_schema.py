from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    document_ids: list[UUID] | None = Field(default=None, max_length=50)
    top_k: int | None = Field(default=None, ge=1, le=20)
    enable_cot: bool = Field(
        default=False,
        description="Opt in to Chain-of-Thought prompting (slower, ~5x output tokens, "
        "better for complex multi-hop questions). Default off.",
    )
    enable_rerank: bool | None = Field(
        default=None,
        description="Override the server-side rerank default (RERANK_ENABLED). "
        "When true, runs an extra LLM call to re-pick the most relevant snippets "
        "(more accurate, slower). When false, skips it (faster, slightly less "
        "accurate). When null, uses the configured server default.",
    )


class CitationResponse(BaseModel):
    chunk_id: str
    document_id: UUID
    filename: str
    chunk_index: int
    score: float
    snippet: str


class StageTimingsResponse(BaseModel):
    embed_ms: float = 0.0
    retrieve_ms: float = 0.0
    rerank_ms: float = 0.0
    complete_ms: float = 0.0
    total_ms: float = 0.0


class UsageResponse(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: Decimal
    model: str
    cache_hit: bool
    timings: StageTimingsResponse = Field(default_factory=StageTimingsResponse)


class AskResponse(BaseModel):
    answer: str
    is_grounded: bool
    refusal_reason: str | None = None
    citations: list[CitationResponse] = Field(default_factory=list)
    usage: UsageResponse
    request_id: str
