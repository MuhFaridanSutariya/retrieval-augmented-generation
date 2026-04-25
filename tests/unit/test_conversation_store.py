import pytest

from app.core.config import Settings
from app.core.exceptions import CacheError
from app.storages.conversation_store import (
    CONVERSATION_KEY_PREFIX,
    ConversationStore,
    ConversationTurn,
)


class FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.last_ttl: int | None = None
        self.fail_get = False
        self.fail_set = False

    async def get(self, key: str) -> str | None:
        if self.fail_get:
            raise CacheError("boom")
        return self.store.get(key)

    async def set(self, key: str, value: str, ttl_seconds: int | None = None) -> None:
        if self.fail_set:
            raise CacheError("boom")
        self.store[key] = value
        self.last_ttl = ttl_seconds

    async def delete(self, key: str) -> None:
        self.store.pop(key, None)


def _store(redis: FakeRedis, settings: Settings) -> ConversationStore:
    return ConversationStore(redis, settings)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_get_returns_empty_when_session_unknown(test_settings: Settings) -> None:
    store = _store(FakeRedis(), test_settings)
    assert await store.get("nope") == []


@pytest.mark.asyncio
async def test_append_then_get_roundtrip(test_settings: Settings) -> None:
    redis = FakeRedis()
    store = _store(redis, test_settings)

    await store.append_turn("s1", user_message="hi", assistant_message="hello")
    turns = await store.get("s1")

    assert turns == [
        ConversationTurn(role="user", content="hi"),
        ConversationTurn(role="assistant", content="hello"),
    ]
    assert redis.last_ttl == test_settings.conversation_ttl_seconds
    assert f"{CONVERSATION_KEY_PREFIX}s1" in redis.store


@pytest.mark.asyncio
async def test_append_trims_to_max_turns(test_settings: Settings) -> None:
    redis = FakeRedis()
    store = _store(redis, test_settings)

    # max_turns=3 means 6 messages max (user+assistant per turn). Push 5 turns.
    for i in range(5):
        await store.append_turn("s1", user_message=f"q{i}", assistant_message=f"a{i}")

    turns = await store.get("s1")
    assert len(turns) == test_settings.conversation_max_turns * 2
    # The oldest two turns should have been dropped; we should still have q2..q4 / a2..a4.
    assert [t.content for t in turns] == ["q2", "a2", "q3", "a3", "q4", "a4"]


@pytest.mark.asyncio
async def test_get_swallows_redis_failure(test_settings: Settings) -> None:
    redis = FakeRedis()
    redis.fail_get = True
    store = _store(redis, test_settings)
    assert await store.get("s1") == []


@pytest.mark.asyncio
async def test_append_swallows_redis_failure(test_settings: Settings) -> None:
    redis = FakeRedis()
    redis.fail_set = True
    store = _store(redis, test_settings)
    # Should not raise — conversation memory is best-effort.
    await store.append_turn("s1", user_message="hi", assistant_message="hello")


@pytest.mark.asyncio
async def test_get_handles_corrupt_payload(test_settings: Settings) -> None:
    redis = FakeRedis()
    redis.store[f"{CONVERSATION_KEY_PREFIX}s1"] = "{not json"
    store = _store(redis, test_settings)
    assert await store.get("s1") == []


@pytest.mark.asyncio
async def test_get_filters_invalid_roles(test_settings: Settings) -> None:
    redis = FakeRedis()
    redis.store[f"{CONVERSATION_KEY_PREFIX}s1"] = (
        '[{"role":"user","content":"ok"},'
        '{"role":"system","content":"nope"},'
        '{"role":"assistant","content":42},'
        '{"role":"assistant","content":"good"}]'
    )
    store = _store(redis, test_settings)
    turns = await store.get("s1")
    assert [t.role for t in turns] == ["user", "assistant"]
    assert [t.content for t in turns] == ["ok", "good"]


@pytest.mark.asyncio
async def test_clear_removes_session(test_settings: Settings) -> None:
    redis = FakeRedis()
    store = _store(redis, test_settings)
    await store.append_turn("s1", user_message="hi", assistant_message="hello")
    await store.clear("s1")
    assert await store.get("s1") == []
