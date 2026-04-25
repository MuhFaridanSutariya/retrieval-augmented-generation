from decimal import Decimal
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    environment: str = "local"
    log_level: str = "INFO"
    log_verbose: bool = False

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_v1_prefix: str = "/api/v1"

    database_url: str
    database_pool_size: int = 10
    database_max_overflow: int = 5

    redis_url: str = "redis://localhost:6379/0"
    response_cache_ttl_seconds: int = 3600
    embedding_cache_ttl_seconds: int = 2_592_000

    openai_api_key: str
    openai_chat_model: str = "gpt-5.4-2026-03-05"
    openai_chat_temperature: float = 0.0
    openai_chat_max_output_tokens: int = 800
    openai_chat_timeout_seconds: float = 30.0
    openai_chat_max_retries: int = 3
    openai_chat_input_usd_per_1m: Decimal = Field(default=Decimal("2.50"))
    openai_chat_output_usd_per_1m: Decimal = Field(default=Decimal("15.00"))
    openai_chat_extended_context_threshold_tokens: int = 272_000
    openai_chat_input_extended_usd_per_1m: Decimal = Field(default=Decimal("5.00"))

    openai_embedding_model: str = "text-embedding-3-large"
    openai_embedding_dimensions: int = 3072
    openai_embedding_batch_size: int = 96
    openai_embedding_usd_per_1m: Decimal = Field(default=Decimal("0.13"))

    faiss_index_path: str = "data/index/faiss.index"
    faiss_metadata_path: str = "data/index/metadata.json"
    faiss_oversample_factor: int = 5

    intent_classification_threshold: float = 0.55

    chunk_size_tokens: int = 500
    chunk_overlap_tokens: int = 50
    retrieval_top_k: int = 8
    rerank_top_k: int = 4
    rerank_enabled: bool = True
    rerank_max_output_tokens: int = 100
    max_tool_iterations: int = 4
    max_tool_output_chars: int = 4000
    hybrid_candidate_multiplier: int = 3
    rrf_k: int = 60
    min_relevance_score: float = 0.1
    max_context_tokens: int = 8000
    token_budget_safety_pad: int = 10

    upload_max_bytes: int = 26_214_400
    upload_allowed_extensions: str = "pdf,txt,md"
    upload_storage_path: str = "data/uploads"

    @property
    def allowed_extensions(self) -> set[str]:
        return {ext.strip().lower() for ext in self.upload_allowed_extensions.split(",")}

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
