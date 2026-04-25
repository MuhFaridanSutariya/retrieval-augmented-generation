from dataclasses import dataclass, field
from decimal import Decimal

from app.models.domain.citation import Citation


@dataclass(slots=True)
class StageTimings:
    embed_ms: float = 0.0
    retrieve_ms: float = 0.0
    rerank_ms: float = 0.0
    complete_ms: float = 0.0
    tool_ms: float = 0.0
    total_ms: float = 0.0


@dataclass(slots=True)
class ToolInvocationRecord:
    name: str
    arguments: dict
    output: str
    ok: bool
    error: str | None = None
    elapsed_ms: float = 0.0


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
    timings: StageTimings = field(default_factory=StageTimings)
    tool_invocations: list[ToolInvocationRecord] = field(default_factory=list)
