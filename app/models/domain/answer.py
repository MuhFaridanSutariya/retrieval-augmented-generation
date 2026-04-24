from dataclasses import dataclass, field
from decimal import Decimal

from app.models.domain.citation import Citation


@dataclass(slots=True)
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    is_grounded: bool = True
    refusal_reason: str | None = None
    reasoning: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: Decimal = Decimal("0")
    model: str = ""
    prompt_version: str = ""
    cache_hit: bool = False
