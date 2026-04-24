from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    document_ids: list[UUID] | None = Field(default=None, max_length=50)
    top_k: int | None = Field(default=None, ge=1, le=20)


class CitationResponse(BaseModel):
    chunk_id: str
    document_id: UUID
    filename: str
    chunk_index: int
    score: float
    snippet: str


class UsageResponse(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: Decimal
    model: str
    cache_hit: bool


class AskResponse(BaseModel):
    answer: str
    is_grounded: bool
    refusal_reason: str | None = None
    citations: list[CitationResponse] = Field(default_factory=list)
    usage: UsageResponse
    request_id: str
