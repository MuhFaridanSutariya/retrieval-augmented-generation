from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, RateLimitError
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.cache.embedding_cache import EmbeddingCache
from app.core.config import Settings
from app.core.exceptions import EmbeddingError
from app.core.logging import get_logger

logger = get_logger(__name__)


class OpenAIEmbedder:
    def __init__(self, settings: Settings, embedding_cache: EmbeddingCache) -> None:
        self._settings = settings
        self._cache = embedding_cache
        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.openai_chat_timeout_seconds,
            max_retries=0,
        )

    async def embed_single(self, text: str) -> list[float]:
        cached = await self._cache.get(text)
        if cached is not None:
            return cached

        embedding = (await self._embed_batch([text]))[0]
        await self._cache.set(text, embedding)
        return embedding

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        results: list[list[float] | None] = [None] * len(texts)
        missing_indexes: list[int] = []
        missing_texts: list[str] = []

        for index, text in enumerate(texts):
            cached = await self._cache.get(text)
            if cached is not None:
                results[index] = cached
            else:
                missing_indexes.append(index)
                missing_texts.append(text)

        batch_size = self._settings.openai_embedding_batch_size
        for start in range(0, len(missing_texts), batch_size):
            batch = missing_texts[start : start + batch_size]
            batch_indexes = missing_indexes[start : start + batch_size]
            embeddings = await self._embed_batch(batch)
            for text, embedding, target_index in zip(batch, embeddings, batch_indexes, strict=True):
                results[target_index] = embedding
                await self._cache.set(text, embedding)

        return [r for r in results if r is not None]

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self._settings.openai_chat_max_retries),
                wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
                retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
                reraise=True,
            ):
                with attempt:
                    response = await self._client.embeddings.create(
                        model=self._settings.openai_embedding_model,
                        input=batch,
                    )
        except APITimeoutError as exc:
            raise EmbeddingError("OpenAI embedding request timed out.") from exc
        except (APIConnectionError, RateLimitError, Exception) as exc:
            raise EmbeddingError(f"OpenAI embedding request failed: {exc}") from exc

        return [item.embedding for item in response.data]
