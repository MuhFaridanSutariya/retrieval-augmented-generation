import json
from dataclasses import dataclass

from app.core.config import Settings
from app.core.exceptions import CacheError
from app.core.logging import get_logger
from app.storages.redis_store import RedisStore

logger = get_logger(__name__)

CONVERSATION_KEY_PREFIX = "conversation:"


@dataclass(slots=True)
class ConversationTurn:
    role: str
    content: str

    def as_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


class ConversationStore:
    def __init__(self, redis_store: RedisStore, settings: Settings) -> None:
        self._redis = redis_store
        self._ttl_seconds = settings.conversation_ttl_seconds
        self._max_messages = settings.conversation_max_turns * 2

    def _key(self, session_id: str) -> str:
        return f"{CONVERSATION_KEY_PREFIX}{session_id}"

    async def get(self, session_id: str) -> list[ConversationTurn]:
        try:
            raw = await self._redis.get(self._key(session_id))
        except CacheError:
            return []
        if not raw:
            return []
        try:
            payload = json.loads(raw)
        except (ValueError, TypeError):
            logger.warning("conversation_store_corrupt", session_id=session_id)
            return []
        if not isinstance(payload, list):
            return []
        turns: list[ConversationTurn] = []
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            role = entry.get("role")
            content = entry.get("content")
            if role in ("user", "assistant") and isinstance(content, str):
                turns.append(ConversationTurn(role=role, content=content))
        return turns

    async def append_turn(
        self,
        session_id: str,
        *,
        user_message: str,
        assistant_message: str,
    ) -> None:
        existing = await self.get(session_id)
        existing.append(ConversationTurn(role="user", content=user_message))
        existing.append(ConversationTurn(role="assistant", content=assistant_message))
        # Keep only the latest N messages so we don't grow the prompt indefinitely.
        trimmed = existing[-self._max_messages :] if self._max_messages > 0 else []
        payload = json.dumps([turn.as_dict() for turn in trimmed])
        try:
            await self._redis.set(self._key(session_id), payload, ttl_seconds=self._ttl_seconds)
        except CacheError:
            # Conversation memory is best-effort; never fail the user request because
            # of a Redis hiccup. Next turn will just see no history.
            pass

    async def clear(self, session_id: str) -> None:
        try:
            await self._redis.delete(self._key(session_id))
        except CacheError:
            pass
