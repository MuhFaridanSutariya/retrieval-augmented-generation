from uuid import UUID

from app.utils.hashing import (
    build_embedding_cache_key,
    build_response_cache_key,
    fingerprint_history,
    sha256_bytes,
    sha256_hex,
)


def test_sha256_hex_deterministic() -> None:
    assert sha256_hex("hello") == sha256_hex("hello")
    assert sha256_hex("hello") != sha256_hex("world")


def test_sha256_bytes_matches_known_value() -> None:
    assert sha256_bytes(b"abc").startswith("ba7816bf")


def test_response_cache_key_is_case_insensitive_and_whitespace_tolerant() -> None:
    key_a = build_response_cache_key(
        question="What is RAG?",
        document_ids=None,
        model="gpt-5.4",
        prompt_version="v1",
    )
    key_b = build_response_cache_key(
        question="  what is rag?  ",
        document_ids=None,
        model="gpt-5.4",
        prompt_version="v1",
    )
    assert key_a == key_b


def test_response_cache_key_varies_by_document_scope() -> None:
    doc_id = UUID("11111111-1111-1111-1111-111111111111")
    key_global = build_response_cache_key(
        question="q", document_ids=None, model="m", prompt_version="v1"
    )
    key_scoped = build_response_cache_key(
        question="q", document_ids=[doc_id], model="m", prompt_version="v1"
    )
    assert key_global != key_scoped


def test_response_cache_key_independent_of_document_order() -> None:
    a = UUID("11111111-1111-1111-1111-111111111111")
    b = UUID("22222222-2222-2222-2222-222222222222")
    key_1 = build_response_cache_key(
        question="q", document_ids=[a, b], model="m", prompt_version="v1"
    )
    key_2 = build_response_cache_key(
        question="q", document_ids=[b, a], model="m", prompt_version="v1"
    )
    assert key_1 == key_2


def test_response_cache_key_varies_by_prompt_version() -> None:
    key_v1 = build_response_cache_key(
        question="q", document_ids=None, model="m", prompt_version="v1"
    )
    key_v2 = build_response_cache_key(
        question="q", document_ids=None, model="m", prompt_version="v2"
    )
    assert key_v1 != key_v2


def test_embedding_cache_key_varies_by_model() -> None:
    key_small = build_embedding_cache_key("hello", "text-embedding-3-small")
    key_large = build_embedding_cache_key("hello", "text-embedding-3-large")
    assert key_small != key_large


def test_fingerprint_history_empty_is_blank() -> None:
    assert fingerprint_history([]) == ""


def test_fingerprint_history_changes_with_history() -> None:
    fp_a = fingerprint_history([("user", "hi"), ("assistant", "hello")])
    fp_b = fingerprint_history([("user", "different"), ("assistant", "hello")])
    assert fp_a != fp_b


def test_response_cache_key_varies_by_history_fingerprint() -> None:
    key_no_history = build_response_cache_key(
        question="and Q4?",
        document_ids=None,
        model="m",
        prompt_version="v1",
        history_fingerprint="",
    )
    key_with_history = build_response_cache_key(
        question="and Q4?",
        document_ids=None,
        model="m",
        prompt_version="v1",
        history_fingerprint="abc123",
    )
    assert key_no_history != key_with_history
