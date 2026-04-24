# AI Knowledge Assistant

A production-like RAG-based AI assistant that answers questions grounded **only** in the documents you upload.

- **LLM:** OpenAI `gpt-5.4-2026-03-05`
- **Embeddings:** OpenAI `text-embedding-3-large` (3072 dim)
- **Vector store:** FAISS (local, file-backed, zero-service)
- **Metadata store:** PostgreSQL (via SQLAlchemy async + Alembic)
- **Cache:** Redis (response + embedding caches)
- **API:** FastAPI

## Prerequisites

- Python 3.11+
- Docker Desktop (for Postgres + Redis)
- An OpenAI API key

## Install `uv`

[`uv`](https://github.com/astral-sh/uv) is the Python package manager used by this project. It manages the virtualenv and dependencies — no `pip install` needed.

### Windows (PowerShell)

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### macOS / Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Verify

```bash
uv --version
# uv 0.10.x  (or newer)
```

If you already have Python installed, you can also install via `pip`:

```bash
pip install uv
```

## Quick start

```bash
# 1. Install project dependencies into a managed virtualenv (.venv)
uv sync

# 2. Configure environment
cp .env.example .env
#   then set OPENAI_API_KEY in .env

# 3. Start Postgres + Redis
docker compose up -d

# 4. Apply database migrations
uv run alembic upgrade head

# 5. Run the API
uv run uvicorn app.main:app --reload
```

API is live at http://localhost:8000. OpenAPI docs at http://localhost:8000/docs.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness |
| `GET` | `/health/ready` | Readiness (DB + Redis check) |
| `POST` | `/api/v1/documents` | Upload a PDF / TXT / MD document |
| `GET` | `/api/v1/documents` | List documents (paginated) |
| `GET` | `/api/v1/documents/{id}` | Get one document |
| `PATCH` | `/api/v1/documents/{id}` | Rename a document |
| `DELETE` | `/api/v1/documents/{id}` | Delete a document + its vectors |
| `POST` | `/api/v1/ask` | Ask a grounded question |

## Sample usage

### Upload

```bash
curl -X POST http://localhost:8000/api/v1/documents \
  -F "file=@ASSIGNMENT.md"
```

Response:
```json
{
  "id": "c1a7b2e0-...",
  "filename": "ASSIGNMENT.md",
  "file_type": "md",
  "status": "READY",
  "chunk_count": 12,
  "size_bytes": 3675,
  "content_hash": "...",
  "created_at": "...",
  "updated_at": "..."
}
```

### Ask

```bash
curl -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the objective of the assignment?"}'
```

Response:
```json
{
  "answer": "The objective is to build a production-like AI assistant... [S1]",
  "is_grounded": true,
  "refusal_reason": null,
  "citations": [
    {
      "chunk_id": "...:0",
      "document_id": "...",
      "filename": "ASSIGNMENT.md",
      "chunk_index": 0,
      "score": 0.78,
      "snippet": "..."
    }
  ],
  "usage": {
    "prompt_tokens": 1024,
    "completion_tokens": 87,
    "total_tokens": 1111,
    "estimated_cost_usd": "0.003866",
    "model": "gpt-5.4-2026-03-05",
    "cache_hit": false
  },
  "request_id": "..."
}
```

Scope the question to specific documents:
```bash
curl -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What file types are accepted?",
    "document_ids": ["c1a7b2e0-..."]
  }'
```

## Behaviour when context is missing

If no chunk in the corpus is relevant enough (below `MIN_RELEVANCE_SCORE`) or the LLM refuses, the response is:
```json
{
  "answer": "I do not have enough information in the provided documents to answer that.",
  "is_grounded": false,
  "refusal_reason": "no_relevant_context_in_corpus",
  "citations": [],
  ...
}
```

The route returns **HTTP 404** for `NoRelevantContext` and **HTTP 400** for empty queries.

## Project layout

```
app/
├── api/
│   ├── application/        # Health endpoints
│   ├── v1/routes/          # Versioned HTTP routes
│   └── exception_handlers.py
├── core/                   # config, exceptions, logging, metrics
├── enums/
├── models/
│   ├── domain/             # Pure domain entities
│   ├── orm/                # SQLAlchemy models
│   ├── schema/             # Pydantic request/response
│   └── mappers.py          # One-way conversions
├── storages/
│   ├── database.py         # Postgres async engine + session
│   ├── redis_store.py
│   └── faiss_store.py      # FAISS index wrapper (upsert/query/delete)
├── repositories/           # DB access only, raises typed exceptions
├── services/               # Application layer orchestration
├── pipelines/              # ingest / query pipelines
├── retrievers/             # Vector retrieval
├── chunkers/               # Recursive token-aware splitter
├── embedders/              # OpenAI embedding wrapper + cache
├── llm_clients/            # OpenAI chat wrapper (timeout, retry)
├── loaders/                # PDF / text / markdown
├── prompts/                # Versioned prompt templates
├── cache/                  # Response + embedding caches
├── validators/
├── utils/                  # hashing, token counting + cost
├── dependencies.py         # Single DI composition root
└── main.py                 # FastAPI app factory + lifespan

migrations/versions/        # Alembic (autogenerated, reviewed)
tests/unit/                 # 46 unit tests for deterministic layers
data/
├── corpus/                 # Source documents (gitignored)
└── index/                  # FAISS index file + metadata
```

See `docs/architecture.md` for design decisions and trade-offs.

## Development

```bash
make dev         # docker compose up + uvicorn --reload
make migrate     # alembic upgrade head
make test        # pytest tests/
make lint        # ruff check
make format      # ruff format + ruff check --fix
```

### Running tests

```bash
uv run pytest tests/unit -v
```

46 unit tests cover: chunker, token counting, cost estimation, hashing / cache keys, validators, prompt builder, mappers, and FAISS store (upsert, filter, delete, persistence).

### Adding a migration

```bash
uv run alembic revision --autogenerate -m "describe_the_change"
# Review migrations/versions/<new>.py — tighten server defaults, add extensions.
uv run alembic upgrade head
```

## Configuration

All tunables live in `.env` (loaded by `app/core/config.py`). No hardcoded values in business logic.

Key settings:
- `OPENAI_CHAT_MODEL`, `OPENAI_EMBEDDING_MODEL`
- `CHUNK_SIZE_TOKENS` (default 500), `CHUNK_OVERLAP_TOKENS` (50)
- `RETRIEVAL_TOP_K` (8), `MIN_RELEVANCE_SCORE` (0.25)
- `MAX_CONTEXT_TOKENS` (8000) — caps prompt size well below the 272K extended-context boundary
- `RESPONSE_CACHE_TTL_SECONDS` (3600), `EMBEDDING_CACHE_TTL_SECONDS` (30 days)
- `UPLOAD_MAX_BYTES` (25 MB), `UPLOAD_ALLOWED_EXTENSIONS` (pdf,txt,md)

## Security

- `.env` is gitignored. Never commit it.
- Queries are hashed before logging; raw content is only logged when `LOG_VERBOSE=true`.
- No PII is emitted to metrics by default.
