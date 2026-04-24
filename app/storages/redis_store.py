import redis.asyncio as redis_async

from app.core.config import Settings
from app.core.exceptions import CacheError
from app.core.logging import get_logger

logger = get_logger(__name__)


class RedisStore:
    def __init__(self, settings: Settings) -> None:
        self._client: redis_async.Redis = redis_async.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_timeout=5.0,
            socket_connect_timeout=5.0,
        )

    async def get(self, key: str) -> str | None:
        try:
            return await self._client.get(key)
        except redis_async.RedisError as exc:
            logger.warning("redis_get_failed", key=key, error=str(exc))
            raise CacheError(f"Redis GET failed for key {key}") from exc

    async def set(self, key: str, value: str, ttl_seconds: int | None = None) -> None:
        try:
            await self._client.set(key, value, ex=ttl_seconds)
        except redis_async.RedisError as exc:
            logger.warning("redis_set_failed", key=key, error=str(exc))
            raise CacheError(f"Redis SET failed for key {key}") from exc

    async def delete(self, key: str) -> None:
        try:
            await self._client.delete(key)
        except redis_async.RedisError as exc:
            logger.warning("redis_delete_failed", key=key, error=str(exc))
            raise CacheError(f"Redis DEL failed for key {key}") from exc

    async def ping(self) -> bool:
        try:
            return bool(await self._client.ping())
        except redis_async.RedisError:
            return False

    async def close(self) -> None:
        await self._client.aclose()
