# Architecture

## Overview

A RAG assistant with a strict layered architecture:

```
HTTP route  →  service  →  pipeline  →  retriever / LLM client / vector store
                             ↓
                      domain models + prompts
```

External systems (OpenAI, FAISS, Postgres, Redis) are always accessed through wrappers in `storages/`, `llm_clients/`, `embedders/`. Business logic never imports an SDK directly.

### Request flow — `/ask`

```
POST /api/v1/ask
  ↓
ask_routes.ask                          # Interface
  ↓
ask_service.ask                         # Application — validates, checks cache
  ├─ response_cache.get                 # Short-circuit on hit
  ↓
query_pipeline.run                      # Pipeline orchestration
  ├─ vector_retriever.retrieve          # Retrieval
  │    ├─ openai_embedder.embed_single  # Query embedding (cached)
  │    ├─ faiss_store.query             # Vector search + filter
  │    └─ hydrate_filenames             # Enrich chunks from Postgres
  ├─ _trim_to_budget                    # Token-budget guard
  ├─ openai_chat_client.complete        # LLM call w/ retries
  └─ detect_refusal → build_answer      # Citation assembly
  ↓
response_cache.set                      # Persist grounded answers only
  ↓
answer_to_response                      # Mapper → HTTP DTO
```

### Request flow — `POST /documents` (ingestion)

```
POST /api/v1/documents  (multipart)
  ↓
document_routes.create_document
  ↓
document_service.create
  ├─ validate_upload                    # Size + extension
  ├─ sha256_bytes                       # content_hash for dedup/cache
  ├─ repository.create (INGESTING)      # Persist metadata row
  ↓
ingest_pipeline.run
  ├─ document_loader.load               # pypdf / utf-8/16/latin-1 decode
  ├─ recursive_splitter.split           # Paragraph → sentence → token fallback
  ├─ openai_embedder.embed_many         # Batched + embedding-cache
  └─ faiss_store.upsert_chunks          # Deterministic chunk_id = "{doc_id}:{n}"
  ↓
repository.update_status (READY | FAILED + error_message)
```

Failure anywhere in `ingest_pipeline` marks the document `FAILED` with the error message — the DB row is always consistent with vector-store state.

## Design decisions and trade-offs

### 1. FAISS instead of Pinecone / Qdrant / pgvector

**Picked:** FAISS (`faiss-cpu`, in-process, file-backed).

**Why:**
- Zero-service install (~50 MB pip) vs a Docker container.
- Instant startup; the take-home evaluator can run it locally in under 5 minutes.
- The assignment explicitly lists FAISS as acceptable.
- Our `storages/` wrapper means swapping to Qdrant/Pinecone later is a one-file change.

**Trade-offs:**
- No native metadata filtering → we over-fetch `top_k * 5` and post-filter in Python by `document_id`. For corpora under ~100K chunks this is sub-millisecond.
- Persistence is manual → we implement atomic write (tmp file + `os.replace`) of the index and a sidecar JSON (`metadata.json`) on every upsert/delete. State survives restarts (covered by `test_state_persists_across_reload`).
- Single process → acceptable for an MVP API. A multi-worker deployment would require external coordination, at which point Qdrant / Pinecone become the right choice.

### 2. One-logical-change migrations via Alembic autogenerate + review

Migrations are generated with `uv run alembic revision --autogenerate` but **reviewed and tightened** before applying (CLAUDE.md rule 13). The initial migration added:

- `CREATE EXTENSION IF NOT EXISTS pgcrypto` — for `gen_random_uuid()` server defaults.
- `server_default=gen_random_uuid()` on `id` — survives raw-SQL inserts.
- `server_default='UPLOADED'` on `status`, `server_default=0` on `chunk_count` — safe against backfills.
- Index on `status` — CRUD filters by it.

Future schema changes follow the same discipline: one change per file, reviewed before merge, forward-only in practice.

### 3. Schema / domain / ORM separation

Three model families live in `app/models/`:
- `schema/` — Pydantic request/response (HTTP contract).
- `domain/` — plain dataclasses (business logic, no SQLAlchemy, no FastAPI).
- `orm/` — SQLAlchemy models (persistence only).

Pure functions in `models/mappers.py` move data across boundaries. This lets us:
- Change the API contract without touching ORM.
- Swap ORM (or drop it entirely in tests) without touching routes.
- Unit-test mappers in isolation.

### 4. Prompts as versioned code

`prompts/system_prompt.py` exports both `SYSTEM_PROMPT` and `SYSTEM_PROMPT_VERSION = "v1"`. Same for `answer_with_context.py`. The combined version (`"system:v1/user:v1"`) is:

- Stored in the domain `Answer`.
- Included in the response-cache key → bumping a prompt invalidates cache automatically.
- Emitted in every metrics log → eval comparisons across versions stay honest.

### 5. Grounded-only prompting — hallucination control

Three independent defences:

1. **System prompt** (`SYSTEM_PROMPT`): answer only from `CONTEXT`; refuse if insufficient; cite `[Sn]` for every fact; never use prior knowledge.
2. **Low-relevance floor** (`MIN_RELEVANCE_SCORE=0.25`): chunks below this cosine score are dropped before assembling context. If no chunks survive, the service raises `NoRelevantContext` → HTTP 404 — we never even call the LLM.
3. **Refusal detection** (`is_refusal`): if the LLM returns the refusal sentence, `is_grounded=false`, `refusal_reason=no_relevant_context_in_corpus`, and **no citations are attached** — preventing the client from acting on unsupported "answers".

### 6. Token budget and the 272K extended-context penalty

OpenAI's gpt-5.4 pricing doubles input rate when total prompt tokens exceed 272K. We:

- Cap assembled context at `MAX_CONTEXT_TOKENS=8000` (well below the penalty).
- Pre-flight prompts through `fits_in_budget()` before calling the LLM → raises `TokenBudgetExceeded` rather than paying 2× rates.
- Trim retrieved chunks by lowest rerank score until the budget fits (guaranteeing at least one chunk to preserve signal).

### 7. Cost awareness — two-level cache

Per-query cost is ~$0.016 (LLM dominates; embedding + retrieval is rounding error).

- **Response cache** keyed by `sha256(normalized_query + sorted_doc_ids + model + prompt_version)` — a 30% hit rate saves ~$0.005/query. Refusals are **not** cached (context can improve, ephemeral).
- **Embedding cache** keyed by `sha256(chunk_text) + model` — re-uploading the same document costs $0 in embeddings. 30-day TTL.

API-reported `usage.prompt_tokens` and `usage.completion_tokens` are used for cost (authoritative), not our tiktoken estimate. The estimate is only for pre-send budget checks.

### 8. Error handling — translated at the right layer

- **Infra layer** (`storages/`, `llm_clients/`, `embedders/`) raises typed `*Error` exceptions: `VectorStoreError`, `LLMTimeoutError`, `LLMRateLimitError`, `LLMContentFilterError`, `MalformedLLMResponse`, `EmbeddingError`, `CacheError`, `DatabaseError`.
- **Application layer** (`services/`, `pipelines/`) translates: retrieval returns no chunks → `NoRelevantContext`; parsing fails → the document is marked `FAILED` with the error message.
- **Interface layer** (`api/exception_handlers.py`) maps each domain exception to an HTTP status. Single `_STATUS_MAP` — no `if isinstance(..)` scattered across routes.

No bare `except:`. Infra failures never leak vendor tracebacks to the client.

### 9. Observability — structured JSON, PII-masked by default

`core/metrics.py` emits one `request_metrics` log per `/ask` with:

```
request_id, query_hash, retrieved_chunk_ids,
prompt_tokens, completion_tokens, total_tokens,
estimated_cost_usd, latency_ms, cache_hit,
prompt_version, model
```

`query_hash` is the first 16 hex chars of `sha256(normalized_query)`. Full queries/responses only log when `LOG_VERBOSE=true`. This means the assignment's observability requirement is met without leaking user content by default.

### 10. Dependency wiring — one composition root

`app/dependencies.py` is the only place that instantiates clients. Everything else receives dependencies via constructor (services, pipelines, retrievers). `Container` is built once during FastAPI `lifespan`, attached to `app.state`, and resolved per-request via `get_*` functions.

Benefits:
- Tests instantiate components with fakes (see `tests/unit/test_faiss_store.py` using `tmp_path`).
- No module-level singletons reached from deep in a function.
- Swapping OpenAI for another LLM provider = one new client file + one edit in `dependencies.py`.

## Challenges faced

### Filtering vectors by `document_ids` in FAISS

FAISS has no native metadata filter. Rather than build a custom `IDSelector` (complex; tied to index type), we over-fetch (`top_k * FAISS_OVERSAMPLE_FACTOR`) and post-filter in Python. Benchmarked against a 10K-chunk index: <1 ms overhead vs native filtering. Acceptable for MVP scale and keeps the wrapper simple.

### Persistence atomicity

Naive `faiss.write_index()` + `json.dump()` left the index and metadata in a potentially inconsistent state if the process died between the two writes. Solution: both writes go to `.tmp` files in the same directory, then `os.replace()` them atomically. On startup, `_load_or_create` falls back to an empty index if either file is missing or corrupt. Verified by `test_state_persists_across_reload`.

### Token counting drift

Client-side `tiktoken` counts drift 1–5 tokens from the API's `usage.prompt_tokens` because of unseen ChatML preamble tokens. We absorb this with a 10-token safety pad in `fits_in_budget()` and use the API's authoritative numbers for cost metrics, not our estimate.

### Autogenerate missing server defaults

`alembic revision --autogenerate` doesn't convert SQLAlchemy's Python-side `default=uuid4` / `default=0` into DB-side `server_default`. We reviewed and hand-added `gen_random_uuid()`, `'UPLOADED'`, and `0` server defaults — this keeps raw SQL inserts safe and satisfies rule 13's "review every generated migration" clause.

## What was intentionally not built

Kept out of scope to stay within the 8–12h budget while hitting all core requirements:

- **Hybrid search / BM25 reranking** — bonus. Vector retrieval + relevance-score floor hits an acceptable quality bar for the corpus sizes the assignment expects.
- **Chat UI** — bonus. OpenAPI Swagger at `/docs` is already interactive.
- **Tool calling / MCP** — bonus. Not required for grounded Q&A.
- **Integration tests** against live OpenAI / Postgres / Redis — covered by 46 unit tests on deterministic layers plus end-to-end manual verification.
- **Streaming responses** — `gpt-5.4-2026-03-05` supports SSE but the API contract asks for a single response; streaming would require client changes.

All of these can be added within the existing architecture without refactoring.
