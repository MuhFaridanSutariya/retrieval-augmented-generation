from unittest.mock import AsyncMock

import numpy as np
import pytest

from app.core.config import Settings
from app.enums.intent import Intent
from app.validators.intent_classifier import (
    FAREWELL_ANCHORS,
    GREETING_ANCHORS,
    IntentClassifier,
    fast_path_classify,
)


@pytest.fixture
def classifier(test_settings: Settings) -> IntentClassifier:
    embedder = AsyncMock()

    async def fake_embed_many(texts: list[str]) -> list[list[float]]:
        return [_synthetic_embedding(text) for text in texts]

    embedder.embed_many.side_effect = fake_embed_many
    return IntentClassifier(embedder=embedder, settings=test_settings)


def _synthetic_embedding(text: str) -> list[float]:
    rng = np.random.default_rng(seed=hash(text) & 0xFFFF_FFFF)
    vector = rng.standard_normal(8)
    if any(text.lower() in anchor.lower() or anchor.lower() in text.lower() for anchor in GREETING_ANCHORS):
        vector[0] += 5
    elif any(text.lower() in anchor.lower() or anchor.lower() in text.lower() for anchor in FAREWELL_ANCHORS):
        vector[1] += 5
    norm = np.linalg.norm(vector)
    return (vector / norm).tolist()


@pytest.mark.asyncio
async def test_classifies_greeting(classifier: IntentClassifier) -> None:
    embedding = _synthetic_embedding("hello")
    assert await classifier.classify(embedding) is Intent.GREETING


@pytest.mark.asyncio
async def test_classifies_farewell(classifier: IntentClassifier) -> None:
    embedding = _synthetic_embedding("goodbye")
    assert await classifier.classify(embedding) is Intent.FAREWELL


@pytest.mark.asyncio
async def test_classifies_question_as_rag_query(classifier: IntentClassifier) -> None:
    orthogonal_question_vector = [0.0, 0.0, 0.7, 0.7, 0.0, 0.0, 0.0, 0.0]
    norm = float(np.linalg.norm(orthogonal_question_vector))
    unit = [v / norm for v in orthogonal_question_vector]
    assert await classifier.classify(unit) is Intent.RAG_QUERY


@pytest.mark.asyncio
async def test_anchors_are_loaded_only_once(classifier: IntentClassifier) -> None:
    await classifier.classify(_synthetic_embedding("hello"))
    await classifier.classify(_synthetic_embedding("hello"))
    await classifier.classify(_synthetic_embedding("question"))
    embedder = classifier._embedder
    assert embedder.embed_many.await_count == 2


@pytest.mark.asyncio
async def test_threshold_falls_back_to_rag_when_score_low(test_settings: Settings) -> None:
    embedder = AsyncMock()
    embedder.embed_many.side_effect = lambda texts: [[1.0, 0.0, 0.0, 0.0] for _ in texts]
    data = test_settings.model_dump()
    data["intent_classification_threshold"] = 0.99
    high_threshold_settings = Settings(**data)
    classifier = IntentClassifier(embedder=embedder, settings=high_threshold_settings)

    near_anchor = [0.9, 0.0, 0.0, 0.4358898943540674]
    assert await classifier.classify(near_anchor) is Intent.RAG_QUERY


def test_fast_path_matches_common_greetings() -> None:
    assert fast_path_classify("hi") is Intent.GREETING
    assert fast_path_classify("hello") is Intent.GREETING
    assert fast_path_classify("halo") is Intent.GREETING
    assert fast_path_classify("good morning") is Intent.GREETING
    assert fast_path_classify("hola") is Intent.GREETING


def test_fast_path_matches_common_farewells() -> None:
    assert fast_path_classify("bye") is Intent.FAREWELL
    assert fast_path_classify("thanks") is Intent.FAREWELL
    assert fast_path_classify("thank you") is Intent.FAREWELL
    assert fast_path_classify("thanks bye") is Intent.FAREWELL
    assert fast_path_classify("grazie") is Intent.FAREWELL


def test_fast_path_is_case_insensitive_and_whitespace_tolerant() -> None:
    assert fast_path_classify("  HELLO  ") is Intent.GREETING
    assert fast_path_classify("Halo") is Intent.GREETING
    assert fast_path_classify("\tBYE\n") is Intent.FAREWELL


def test_fast_path_strips_trailing_punctuation() -> None:
    assert fast_path_classify("hi!") is Intent.GREETING
    assert fast_path_classify("hello.") is Intent.GREETING
    assert fast_path_classify("bye?!") is Intent.FAREWELL
    assert fast_path_classify("thanks!!!") is Intent.FAREWELL


def test_fast_path_misses_anything_with_extra_words() -> None:
    assert fast_path_classify("halo, can you tell me about the SLA?") is None
    assert fast_path_classify("hi there friend, what about latency?") is None
    assert fast_path_classify("thanks for explaining the cost breakdown") is None


def test_fast_path_misses_unknown_variants() -> None:
    # Truly unknown variants (not literal matches and not just elongated forms of known phrases)
    # still fall through to the embedding classifier.
    assert fast_path_classify("greetings") is None
    assert fast_path_classify("howdy partner") is None
    assert fast_path_classify("salutations") is None


def test_fast_path_misses_real_questions() -> None:
    assert fast_path_classify("What was the p95 latency in Q1 2026?") is None
    assert fast_path_classify("List the four production incidents") is None
    assert fast_path_classify("How much did OpenAI chat cost?") is None


def test_fast_path_handles_empty_input() -> None:
    assert fast_path_classify("") is None
    assert fast_path_classify("   ") is None
    assert fast_path_classify("!!!") is None


def test_fast_path_collapses_elongated_greetings() -> None:
    assert fast_path_classify("holaaaaaaaaaaa") is Intent.GREETING
    assert fast_path_classify("hellooooo") is Intent.GREETING
    assert fast_path_classify("hiiiiii") is Intent.GREETING
    assert fast_path_classify("heyyyyyy") is Intent.GREETING


def test_fast_path_collapses_elongated_farewells() -> None:
    assert fast_path_classify("byeeeeee") is Intent.FAREWELL
    assert fast_path_classify("thxxxxx") is Intent.FAREWELL


def test_fast_path_preserves_real_double_letters() -> None:
    assert fast_path_classify("hello") is Intent.GREETING
    assert fast_path_classify("goodbye") is Intent.FAREWELL
    assert fast_path_classify("cheers") is Intent.FAREWELL


def test_fast_path_collapse_with_punctuation_and_case() -> None:
    assert fast_path_classify("HOLAAAAAAA!!") is Intent.GREETING
    assert fast_path_classify("  Byeeeee.  ") is Intent.FAREWELL
