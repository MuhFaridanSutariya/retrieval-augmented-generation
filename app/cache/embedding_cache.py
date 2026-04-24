import json

from app.core.config import Settings
from app.core.exceptions import CacheError
from app.storages.redis_store import RedisStore
from app.utils.hashing import build_embedding_cache_key


class EmbeddingCache:
    def __init__(self, redis_store: RedisStore, settings: Settings) -> None:
        self._redis = redis_store
        self._ttl = settings.embedding_cache_ttl_seconds
        self._model = settings.openai_embedding_model

    async def get(self, text: str) -> list[float] | None:
        key = build_embedding_cache_key(text, self._model)
        try:
            raw = await self._redis.get(key)
        except CacheError:
            return None
        if not raw:
            return None
        try:
            return json.loads(raw)
        except ValueError:
            return None

    async def set(self, text: str, embedding: list[float]) -> None:
        key = build_embedding_cache_key(text, self._model)
        try:
            await self._redis.set(key, json.dumps(embedding), ttl_seconds=self._ttl)
        except CacheError:
            pass
