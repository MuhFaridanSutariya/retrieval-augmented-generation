.PHONY: install sync run dev stop migrate migrate-create lint format test test-unit test-integration eval clean

install:
	uv sync

sync:
	uv sync

run:
	uv run uvicorn app.main:app --host $${API_HOST:-0.0.0.0} --port $${API_PORT:-8000}

dev:
	docker compose up -d
	uv run uvicorn app.main:app --reload --host $${API_HOST:-0.0.0.0} --port $${API_PORT:-8000}

stop:
	docker compose down

migrate:
	uv run alembic upgrade head

migrate-create:
	uv run alembic revision --autogenerate -m "$(name)"

lint:
	uv run ruff check app tests

format:
	uv run ruff format app tests
	uv run ruff check --fix app tests

test:
	uv run pytest

test-unit:
	uv run pytest tests/unit

test-integration:
	uv run pytest tests/integration

eval:
	uv run python scripts/run_evaluation.py

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} +
