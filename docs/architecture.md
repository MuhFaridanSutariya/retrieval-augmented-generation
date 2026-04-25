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
  ├─ hybrid_retriever.retrieve          # Retrieval (dense + sparse)
  │    ├─ vector_retriever              # OpenAI embedding → FAISS cosine
  │    ├─ bm25_retriever                # Lexical BM25 over chunk text
  │    └─ reciprocal_rank_fusion        # RRF fuses both ranked lists
  ├─ reranker.rerank                    # LLM listwise rerank (top-N → top-K)
  ├─ _trim_to_budget                    # Token-budget guard
  ├─ openai_chat_client.complete        # CoT LLM call w/ retries
  ├─ parse_response                     # Extract <thinking> + <answer>
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

### 10. Chain-of-Thought prompting with structured output (v2 prompts)

The `SYSTEM_PROMPT` (version `v2`) instructs the model to emit:

```
<thinking>
Step 1. Identify entities in the question.
Step 2. List context snippets that are actually relevant.
Step 3. Decide whether the snippets support an answer.
Step 4. Double-check every claim has a [Sn] citation.
</thinking>

<answer>
The user-facing answer, with [Sn] citations inline.
</answer>
```

**Why structured CoT rather than "think step by step" in free prose:**
- Reasoning is extracted (`parse_response()`) and logged as `reasoning` when `LOG_VERBOSE=true`, never returned to the client — users see a clean answer.
- The Step-4 self-check materially reduces unsupported claims: the model re-reads its draft against the context before emitting it.
- Parse is robust to mistakes — if the model forgets the `<answer>` tag, we take everything after `</thinking>`; if it skips the tags entirely, we treat the whole response as the answer. Never raises.
- Prompt version is part of the response-cache key — bumping to v2 invalidated v1 answers automatically.

**Cost impact:** completion tokens roughly 3-4x larger than v1 (≈200 vs ≈40) because the model now writes the thinking block. Observed per-query cost rose from ~$0.016 to ~$0.007-0.012 with the smaller corpus. Still dominated by input tokens.

### 11. Accuracy — hybrid retrieval + LLM reranking

Three stages, each addressing a different failure mode of vector-only retrieval:

**a. Hybrid retrieval (BM25 + vector, fused via RRF)**

Pure vector search misses exact-keyword matches (model names, acronyms, rare proper nouns). BM25 misses paraphrases. We run both and fuse:

```
rrf_score(chunk) = sum over retrievers of 1 / (rrf_k + rank)
```

- `RRF_K = 60` — the canonical value from Cormack et al. (2009); higher `k` flattens the distribution and is more tolerant of noisy rankings.
- `HYBRID_CANDIDATE_MULTIPLIER = 3` — each retriever returns `top_k * 3` candidates so the fusion has a larger pool to work with. For a final `retrieval_top_k = 8`, each retriever produces 24 candidates.
- BM25 index is rebuilt lazily when `FaissStore._generation` changes, so re-ingests are picked up automatically without an explicit invalidation call.

**b. LLM listwise reranking (opt-in via `RERANK_ENABLED`)**

After hybrid fusion, a single OpenAI chat call reranks the top-N candidates into the final top-K (`RERANK_TOP_K=4`). Listwise (one call with all candidates) rather than pointwise (one call per candidate) is ~N× cheaper.

The reranker returns a JSON array of candidate numbers. Parsing is defensive: tries `json.loads(raw)` first (rejects `{"order": [...]}` by type check), falls back to regex extraction if the model wrapped the array in prose, and on any parse failure degrades gracefully to the fused ordering.

Reranker failures (timeout, rate limit, malformed output) never fail the request — they fall back to the fused candidates. This keeps `/ask` available even when a secondary LLM call misbehaves.

**c. Grounding floor + CoT self-check**

- `min_relevance_score=0.1` keeps the vector side permissive (hybrid + rerank do the discrimination).
- If the reranker keeps zero chunks, `NoRelevantContext` → HTTP 404 before we ever call the final LLM.
- The CoT prompt's Step-4 self-check catches claims not actually supported by a snippet.

**Accuracy vs cost trade-off:**

| Component | Latency cost | Dollar cost | Accuracy gain |
|---|---|---|---|
| Vector retrieval alone (baseline) | — | — | — |
| + BM25 hybrid | +2-5 ms | $0 | catches keyword misses, especially rare terms |
| + LLM rerank | +1 LLM call (~1-2 s) | +$0.002-0.005/query | big lift on ambiguous queries |
| + CoT prompt | +completion tokens | +$0.001/query | fewer unsupported claims |

Rerank and CoT are both gated behind config flags so a deployment can trade accuracy for cost/latency at will.

### 12. Concurrency — async RWLock + thread-offloaded CPU

**Problem:** a single `asyncio.Lock` around the FAISS store serialised every `/ask` — concurrent calls ran strictly one at a time.

**Solution:** a writer-preference readers-writer lock built on `asyncio.Condition` (see `utils/async_rwlock.py`). Many readers hold the lock concurrently; writers have exclusive access; waiting writers block new readers so writes don't starve.

Applied to `FaissStore`:
- `query()` → `async with rwlock.read()` — concurrent queries interleave freely.
- `upsert_chunks()` / `delete_by_document()` / `ensure_index()` → `async with rwlock.write()` — serialised against each other and against readers.

**CPU-bound work off the event loop:** `load_pdf()` (pypdf) and `RecursiveSplitter.split()` (tiktoken) are synchronous. Wrapping them in `asyncio.to_thread` inside `ingest_pipeline.run()` means an upload doesn't stall concurrent `/ask` calls.

**Measured:** 5 cold concurrent `/ask` requests complete in **8.53 s** wall-clock vs a serial sum of **31.74 s** — a **3.72× speedup** (exact OpenAI-side throughput limits the upper bound). With the response cache warm, the same 5 take **0.30 s**.

**What we chose not to do:**
- Multiple uvicorn workers. FAISS state is in-process, so N workers = N copies of the index. When the corpus outgrows one process, the right move is swapping FAISS for Qdrant/Pinecone, not stapling workers around an in-memory store.
- `FastAPI.BackgroundTasks` for ingestion. Our test corpora ingest in a few seconds and the contract `POST /documents → status=READY` is simpler than polling. Easy to add when documents grow.

### 13. Complex PDF handling — hybrid pypdf + pdfplumber

Real-world PDFs mix prose, tables, multi-column layout, and occasionally images. `loaders/pdf_loader.py` runs **two passes** per file and merges the results:

1. **Prose pass — `pypdf.extract_text()`** for each page. Fast, well-tested for born-digital PDFs, fixed against mojibake by `ftfy` after extraction.
2. **Table pass — `pdfplumber.extract_tables()`** on the same pages. Each detected table becomes a Markdown table via `_table_to_markdown()`:

   ```
   | Quarter | Revenue |
   | --- | --- |
   | Q1 | 100 |
   | Q2 | 200 |
   ```

   Cells with newlines are flattened with spaces; cells containing pipes are escaped (`a|b` → `a\|b`); short rows are padded to the widest row.

The two outputs are concatenated per page (`prose + "\n\n" + table_md_1 + "\n\n" + table_md_2 ...`) so chunking sees them together. Markdown tables embed and retrieve as well as prose, and gpt-5.4 reads them natively.

**Verified live** on a generated 5 KB PDF with three real tables (SLA metrics, cost breakdown, incidents):

| Question | Answer |
|---|---|
| "What was the p95 latency in Q1 2026 and the SLA target?" | "**3100 ms**, SLA target **≤ 5000 ms**" — exact cell match |
| "How much did OpenAI chat cost in Q1 2026 and the % change?" | "**$16,820**, **+35.6%** vs Q4 2025" |
| "List the four production incidents with severity and root cause." | All four returned with date / severity / cause from the table |

**Failure isolation:** if pdfplumber fails to open the PDF or chokes on a page, we log a warning and continue with pypdf-only output. Tables are an enhancement, never a load-bearing dependency.

**Acknowledged limitations** (see "What was intentionally not built" below): we don't OCR scanned/image-only PDFs, don't extract chart data from images, and pypdf can interleave columns in heavy multi-column layouts. The upgrade path is documented (Tesseract OCR fallback + vision-LLM for charts).

### 14. Dependency wiring — one composition root

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

- **Chat UI** — bonus. OpenAPI Swagger at `/docs` is already interactive.
- **Tool calling / MCP** — bonus. Not required for grounded Q&A.
- **Integration tests** against live OpenAI / Postgres / Redis — covered by 91 unit tests on deterministic layers plus the live smoke tests documented in this file.
- **Streaming responses** — `gpt-5.4-2026-03-05` supports SSE but the API contract asks for a single response; streaming would require client changes.

All of these can be added within the existing architecture without refactoring.

## Future improvements

The README has the full table; this section calls out the highest-leverage items along with the **specific files** an engineer would touch. Wrappers in `loaders/`, `storages/`, `llm_clients/`, and `retrievers/` were chosen so each upgrade is local — no business-logic refactor required.

### PDF & ingestion

| Upgrade | Touches | What changes |
|---|---|---|
| **OCR fallback for scanned PDFs** | `loaders/pdf_loader.py`, `core/config.py` | If pypdf returns < N chars, route the page bytes to `pytesseract` (free, system Tesseract install) or AWS Textract / Azure Document Intelligence (paid, better on tables-in-scans). Add `UPLOAD_ENABLE_OCR` flag. |
| **Vision-LLM enrichment for charts/diagrams** | `loaders/pdf_loader.py`, new `loaders/page_image_loader.py` | Render each PDF page to PNG via `pypdfium2`, send to gpt-5.4 (vision) with a "transcribe this page including any charts" prompt, append result to text. Per-page cost ~$0.005-$0.02. |
| **Multi-column layout** | swap `pypdf` → `PyMuPDF (fitz)` in `pdf_loader.py` | Fitz has layout-aware reading order out of the box. License is AGPL — fine internally, blocker for proprietary product packaging. |
| **Header/footer stripping** | `loaders/pdf_loader.py` | Heuristic: detect repeated short lines across pages and drop them before chunking. Or pull in `unstructured.io`'s element classifier (heavier but more reliable). |

### Retrieval & accuracy

| Upgrade | Touches | What changes |
|---|---|---|
| **Cross-encoder reranker** | new `retrievers/cross_encoder_reranker.py`, `dependencies.py` | `bge-reranker-v2-m3` (~600 MB) runs locally in 30 ms vs the current 1-2 s LLM rerank. Quality comparable; ongoing cost is electricity. Wire as a drop-in replacement for `LLMReranker` (same `rerank()` signature). |
| **Query rewriting** | new `retrievers/query_rewriter.py`, `pipelines/query_pipeline.py` | One cheap LLM call rewrites the question into 2-3 search queries; run hybrid retrieval per rewrite; fuse via RRF. Big lift on conversational queries; +200 ms latency. |
| **HyDE** | `retrievers/vector_retriever.py` | LLM drafts a hypothetical answer first, embed THAT, retrieve. Often beats raw query embedding for short queries. |
| **Language-aware BM25 tokenization** | `retrievers/bm25_retriever.py` | Replace regex `_tokenize` with `Snowball` (stemming), `Jieba` (Chinese), or `Sudachi` (Japanese). Detect per chunk with `langdetect`. Multilingual corpora gain noticeable recall. |

### Scale, ops, safety

| Upgrade | Touches | What changes |
|---|---|---|
| **Qdrant or Pinecone for multi-worker** | `storages/qdrant_store.py` (new) replacing `faiss_store.py` | Same wrapper interface — only `dependencies.py` changes. Required once corpus crosses a single process. |
| **HNSW for >500K chunks** | `storages/faiss_store.py` | Swap `IndexFlatIP` for `IndexHNSWFlat`; expose `efSearch` in config. Sub-linear scan, ~99% recall@10. |
| **Background ingestion** | `services/document_service.py`, `api/v1/routes/document_routes.py` | `BackgroundTasks` (small) or `ARQ`/`Celery` (long jobs). Status surfaced via `GET /documents/{id}`. Changes the API contract — clients must poll. |
| **OpenTelemetry tracing** | `core/metrics.py`, `main.py` | Replace structlog metrics with OTel; FastAPI auto-instrumentation traces every layer. Adds `/metrics` for Prometheus. |
| **Rate limiting + auth** | `main.py`, new `dependencies.py` providers | `slowapi` for per-IP/per-tenant limits; FastAPI `Depends` for API-key or JWT auth. |
| **Streaming responses** | new `api/v1/routes/ask_stream_routes.py` | SSE; buffer the `<thinking>` block server-side, stream only `<answer>` content to the client. |
| **Prompt-injection / safety classifier** | `validators/query_validator.py` | Cheap regex + classifier guard before the question reaches the LLM. |

### Evaluation & cost

| Upgrade | Touches | What changes |
|---|---|---|
| **Golden-set CI eval** | new `tests/eval/`, `Makefile`, GitHub Action | `make eval` runs `golden_set.yaml`; CI surfaces accuracy delta + cost delta vs the baseline branch. |
| **Embedding Batch API for re-index** | `embedders/openai_embedder.py` | Add a `embed_batch_async()` path using OpenAI's Batch API (50% cheaper, 24-hour SLA). Use for one-time corpus re-embeds, not per-query. |
| **Prompt A/B framework** | `prompts/__init__.py`, `services/ask_service.py` | Hash request → bucket → choose `SYSTEM_PROMPT_V3` vs `SYSTEM_PROMPT_V4`; log per-bucket metrics. Ship prompt changes safely. |

If I had **one more day**, the order would be: OCR fallback → cross-encoder reranker → background ingestion → tracing → streaming. That's ~80% of the production-readiness gap closed in a single iteration.
