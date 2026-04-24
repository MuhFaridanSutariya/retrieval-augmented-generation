from dataclasses import dataclass

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    BadRequestError,
    RateLimitError,
)
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import Settings
from app.core.exceptions import (
    LLMContentFilterError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
    MalformedLLMResponse,
    TokenBudgetExceeded,
)
from app.core.logging import get_logger
from app.utils.token_counting import count_message_tokens, fits_in_budget

logger = get_logger(__name__)


@dataclass(slots=True)
class ChatMessage:
    role: str
    content: str

    def as_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass(slots=True)
class ChatCompletion:
    content: str
    prompt_tokens: int
    completion_tokens: int
    model: str
    finish_reason: str


class OpenAIChatClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.openai_chat_timeout_seconds,
            max_retries=0,
        )

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> ChatCompletion:
        temp = temperature if temperature is not None else self._settings.openai_chat_temperature
        max_out = max_output_tokens or self._settings.openai_chat_max_output_tokens
        payload = [m.as_dict() for m in messages]

        if not fits_in_budget(
            payload,
            model=self._settings.openai_chat_model,
            max_output_tokens=max_out,
            max_context_tokens=self._settings.max_context_tokens,
            safety_pad=self._settings.token_budget_safety_pad,
        ):
            estimated = count_message_tokens(payload, self._settings.openai_chat_model)
            raise TokenBudgetExceeded(
                "Prompt exceeds configured token budget.",
                details={
                    "estimated_prompt_tokens": estimated,
                    "max_output_tokens": max_out,
                    "max_context_tokens": self._settings.max_context_tokens,
                },
            )

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self._settings.openai_chat_max_retries),
                wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
                retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
                reraise=True,
            ):
                with attempt:
                    response = await self._client.chat.completions.create(
                        model=self._settings.openai_chat_model,
                        messages=payload,
                        temperature=temp,
                        max_completion_tokens=max_out,
                    )
        except APITimeoutError as exc:
            raise LLMTimeoutError("OpenAI chat completion timed out.") from exc
        except RateLimitError as exc:
            raise LLMRateLimitError("OpenAI rate limit exceeded.") from exc
        except BadRequestError as exc:
            message = str(exc).lower()
            if "content" in message and "filter" in message:
                raise LLMContentFilterError("Request blocked by content filter.") from exc
            raise LLMError(f"OpenAI rejected the request: {exc}") from exc
        except (APIConnectionError, Exception) as exc:
            raise LLMError(f"OpenAI chat completion failed: {exc}") from exc

        choice = response.choices[0]
        if choice.message.content is None:
            raise MalformedLLMResponse("OpenAI returned empty content.")

        usage = response.usage
        return ChatCompletion(
            content=choice.message.content,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            model=response.model,
            finish_reason=choice.finish_reason or "",
        )
