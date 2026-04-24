from decimal import Decimal

from app.core.config import Settings
from app.utils.token_counting import (
    count_message_tokens,
    count_text_tokens,
    estimate_chat_cost_usd,
    estimate_embedding_cost_usd,
    fits_in_budget,
)

MODEL = "gpt-5.4-2026-03-05"


def test_count_text_tokens_empty_returns_zero() -> None:
    assert count_text_tokens("", MODEL) == 0


def test_count_text_tokens_nonempty_is_positive() -> None:
    assert count_text_tokens("The quick brown fox jumps over the lazy dog.", MODEL) > 0


def test_count_message_tokens_includes_chatml_overhead() -> None:
    messages = [{"role": "user", "content": "hi"}]
    content_only = count_text_tokens("hi", MODEL)
    total = count_message_tokens(messages, MODEL)
    # ChatML adds 3 tokens/message + 3 reply-priming tokens — total must exceed raw content.
    assert total > content_only
    assert total - content_only >= 6


def test_estimate_chat_cost_standard_tier(test_settings: Settings) -> None:
    prompt_tokens = 100_000
    assert prompt_tokens < test_settings.openai_chat_extended_context_threshold_tokens
    cost = estimate_chat_cost_usd(prompt_tokens, 0, test_settings)
    expected = (
        Decimal(prompt_tokens)
        * test_settings.openai_chat_input_usd_per_1m
        / Decimal(1_000_000)
    ).quantize(Decimal("0.000001"))
    assert cost == expected


def test_estimate_chat_cost_crosses_extended_threshold(test_settings: Settings) -> None:
    prompt = test_settings.openai_chat_extended_context_threshold_tokens + 1
    cost = estimate_chat_cost_usd(prompt, 0, test_settings)
    expected = (
        Decimal(prompt)
        * test_settings.openai_chat_input_extended_usd_per_1m
        / Decimal(1_000_000)
    ).quantize(Decimal("0.000001"))
    assert cost == expected


def test_estimate_embedding_cost_linear(test_settings: Settings) -> None:
    single = estimate_embedding_cost_usd(500_000, test_settings)
    double = estimate_embedding_cost_usd(1_000_000, test_settings)
    assert double == single * 2


def test_fits_in_budget_rejects_oversize_prompt() -> None:
    huge_message = [{"role": "user", "content": "x " * 5000}]
    assert not fits_in_budget(
        huge_message,
        model=MODEL,
        max_output_tokens=500,
        max_context_tokens=200,
        safety_pad=10,
    )


def test_fits_in_budget_accepts_small_prompt() -> None:
    messages = [{"role": "user", "content": "hello"}]
    assert fits_in_budget(
        messages,
        model=MODEL,
        max_output_tokens=500,
        max_context_tokens=8000,
        safety_pad=10,
    )
