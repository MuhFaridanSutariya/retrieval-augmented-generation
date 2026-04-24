from dataclasses import dataclass
from uuid import UUID


@dataclass(slots=True)
class Chunk:
    id: str
    document_id: UUID
    chunk_index: int
    text: str
    token_count: int
    embedding: list[float] | None = None


@dataclass(slots=True)
class RetrievedChunk:
    id: str
    document_id: UUID
    chunk_index: int
    text: str
    score: float
    filename: str | None = None
