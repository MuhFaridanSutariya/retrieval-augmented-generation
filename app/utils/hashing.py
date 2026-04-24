import hashlib
from uuid import UUID


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_response_cache_key(
    *,
    question: str,
    document_ids: list[UUID] | None,
    model: str,
    prompt_version: str,
) -> str:
    normalized = question.strip().lower()
    scope = ",".join(sorted(str(d) for d in document_ids)) if document_ids else "*"
    payload = f"{normalized}|{scope}|{model}|{prompt_version}"
    return f"response:{sha256_hex(payload)}"


def build_embedding_cache_key(text: str, model: str) -> str:
    return f"embedding:{model}:{sha256_hex(text)}"
