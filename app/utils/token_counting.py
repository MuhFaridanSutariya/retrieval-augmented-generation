from decimal import Decimal

import tiktoken

from app.core.config import Settings

# tiktoken does not yet ship encodings keyed on future model IDs; fall back to cl100k_base
# (the encoding used by gpt-4 family) for any model it does not recognize.
_FALLBACK_ENCODING = "cl100k_base"

# ChatML overhead per message (tokens for role, separators, content wrapper).
# Per OpenAI cookbook "How to count tokens with tiktoken": ~3 tokens/message + 3 tokens to prime reply.
_TOKENS_PER_MESSAGE = 3
_TOKENS_REPLY_PRIMING = 3


def _get_encoding(model: str) -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding(_FALLBACK_ENCODING)


def count_text_tokens(text: str, model: str) -> int:
    encoding = _get_encoding(model)
    return len(encoding.encode(text))


def count_message_tokens(messages: list[dict], model: str) -> int:
    encoding = _get_encoding(model)
    total = 0
    for message in messages:
        total += _TOKENS_PER_MESSAGE
        for value in message.values():
            if isinstance(value, str):
                total += len(encoding.encode(value))
    total += _TOKENS_REPLY_PRIMING
    return total


def estimate_chat_cost_usd(
    prompt_tokens: int,
    completion_tokens: int,
    settings: Settings,
) -> Decimal:
    if prompt_tokens > settings.openai_chat_extended_context_threshold_tokens:
        input_rate = settings.openai_chat_input_extended_usd_per_1m
    else:
        input_rate = settings.openai_chat_input_usd_per_1m

    input_cost = Decimal(prompt_tokens) * input_rate / Decimal(1_000_000)
    output_cost = Decimal(completion_tokens) * settings.openai_chat_output_usd_per_1m / Decimal(1_000_000)
    return (input_cost + output_cost).quantize(Decimal("0.000001"))


def estimate_embedding_cost_usd(total_tokens: int, settings: Settings) -> Decimal:
    return (Decimal(total_tokens) * settings.openai_embedding_usd_per_1m / Decimal(1_000_000)).quantize(
        Decimal("0.000001")
    )


def fits_in_budget(
    messages: list[dict],
    *,
    model: str,
    max_output_tokens: int,
    max_context_tokens: int,
    safety_pad: int,
) -> bool:
    prompt_tokens = count_message_tokens(messages, model)
    return prompt_tokens + max_output_tokens + safety_pad <= max_context_tokens
