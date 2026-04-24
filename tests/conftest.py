import os

import pytest

from app.core.config import Settings


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/knowledge_assistant")
    os.environ.setdefault("OPENAI_API_KEY", "sk-test")
    return Settings()
