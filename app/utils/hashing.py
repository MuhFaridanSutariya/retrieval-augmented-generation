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
    history_fingerprint: str = "",
) -> str:
    normalized = question.strip().lower()
    scope = ",".join(sorted(str(d) for d in document_ids)) if document_ids else "*"
    payload = f"{normalized}|{scope}|{model}|{prompt_version}|h:{history_fingerprint}"
    return f"response:{sha256_hex(payload)}"


def fingerprint_history(history: list[tuple[str, str]]) -> str:
    if not history:
        return ""
    joined = "\n".join(f"{role}:{content.strip()}" for role, content in history)
    return sha256_hex(joined)


def build_embedding_cache_key(text: str, model: str) -> str:
    return f"embedding:{model}:{sha256_hex(text)}"
