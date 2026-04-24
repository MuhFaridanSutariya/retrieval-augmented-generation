from dataclasses import dataclass
from uuid import UUID


@dataclass(slots=True)
class Citation:
    chunk_id: str
    document_id: UUID
    filename: str
    chunk_index: int
    score: float
    snippet: str
