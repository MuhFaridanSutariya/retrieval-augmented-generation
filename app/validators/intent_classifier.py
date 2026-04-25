import asyncio

import numpy as np

from app.core.config import Settings
from app.core.logging import get_logger
from app.embedders.openai_embedder import OpenAIEmbedder
from app.enums.intent import Intent

logger = get_logger(__name__)

GREETING_ANCHORS: tuple[str, ...] = (
    "hello",
    "hi there",
    "hey, how are you?",
    "good morning",
    "good afternoon",
    "halo",
    "ciao",
    "hola",
    "bonjour",
    "namaste",
)

FAREWELL_ANCHORS: tuple[str, ...] = (
    "goodbye",
    "bye for now",
    "see you later",
    "thanks for your help",
    "thank you so much",
    "talk to you later",
    "adios",
    "grazie",
    "danke",
    "arigato",
)

# Bare-literal phrases that bypass the embedding call entirely. Anything not in
# these sets falls through to the embedding classifier — flexibility preserved.
_FAST_GREETINGS: frozenset[str] = frozenset(
    {
        "hi", "hii", "hello", "hellow", "hey", "heyy", "halo", "yo", "sup",
        "hola", "ciao", "salve", "bonjour", "namaste", "konnichiwa",
        "good morning", "good afternoon", "good evening", "good day",
        "morning", "afternoon", "evening",
        "hi there", "hello there", "hey there",
    }
)

_FAST_FAREWELLS: frozenset[str] = frozenset(
    {
        "bye", "byee", "goodbye", "good bye", "farewell",
        "thanks", "thank you", "thanks bye", "thx", "ty", "tysm",
        "see you", "see ya", "cya", "ttyl",
        "adios", "grazie", "danke", "merci", "arigato", "cheers",
    }
)

_TRAILING_PUNCTUATION = "!.?,;:"


def fast_path_classify(question: str) -> Intent | None:
    if not question:
        return None
    normalized = question.strip().rstrip(_TRAILING_PUNCTUATION).strip().lower()
    if not normalized:
        return None
    normalized = _collapse_long_runs(normalized)
    if normalized in _FAST_GREETINGS:
        return Intent.GREETING
    if normalized in _FAST_FAREWELLS:
        return Intent.FAREWELL
    return None


def _collapse_long_runs(text: str) -> str:
    # Collapse any run of 3+ identical characters down to a single character.
    # "holaaaaaa" -> "hola"; "hellooo" -> "hello"; "hello" stays "hello"
    # because its run of 'l' is only 2. No regex — just a single linear scan.
    if not text:
        return text
    out: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        run_end = index
        while run_end < len(text) and text[run_end] == char:
            run_end += 1
        run_length = run_end - index
        out.append(char if run_length >= 3 else char * run_length)
        index = run_end
    return "".join(out)


class IntentClassifier:
    def __init__(self, *, embedder: OpenAIEmbedder, settings: Settings) -> None:
        self._embedder = embedder
        self._threshold = settings.intent_classification_threshold
        self._greeting_anchors: np.ndarray | None = None
        self._farewell_anchors: np.ndarray | None = None
        self._lock = asyncio.Lock()

    async def warm(self) -> None:
        await self._ensure_anchors_loaded()

    async def classify(self, query_embedding: list[float]) -> Intent:
        await self._ensure_anchors_loaded()
        assert self._greeting_anchors is not None
        assert self._farewell_anchors is not None

        query_vector = np.asarray(query_embedding, dtype=np.float32)
        query_norm = float(np.linalg.norm(query_vector)) or 1.0
        unit = query_vector / query_norm

        greeting_score = float((self._greeting_anchors @ unit).max())
        farewell_score = float((self._farewell_anchors @ unit).max())

        if max(greeting_score, farewell_score) < self._threshold:
            return Intent.RAG_QUERY
        return Intent.GREETING if greeting_score >= farewell_score else Intent.FAREWELL

    async def _ensure_anchors_loaded(self) -> None:
        if self._greeting_anchors is not None and self._farewell_anchors is not None:
            return
        async with self._lock:
            if self._greeting_anchors is not None and self._farewell_anchors is not None:
                return
            greetings = await self._embedder.embed_many(list(GREETING_ANCHORS))
            farewells = await self._embedder.embed_many(list(FAREWELL_ANCHORS))
            self._greeting_anchors = _normalize_rows(np.asarray(greetings, dtype=np.float32))
            self._farewell_anchors = _normalize_rows(np.asarray(farewells, dtype=np.float32))
            logger.info(
                "intent_classifier_anchors_loaded",
                greetings=len(GREETING_ANCHORS),
                farewells=len(FAREWELL_ANCHORS),
                threshold=self._threshold,
            )


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms
