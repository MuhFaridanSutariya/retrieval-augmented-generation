import json
import time
from dataclasses import dataclass, field
from typing import Any

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
    ToolLoopExceeded,
)
from app.core.logging import get_logger
from app.tools.base import ToolRegistry
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


@dataclass(slots=True)
class ToolInvocation:
    name: str
    arguments: dict[str, Any]
    output: str
    ok: bool
    error: str | None
    elapsed_ms: float


@dataclass(slots=True)
class ChatCompletionWithTools:
    content: str
    prompt_tokens: int
    completion_tokens: int
    model: str
    finish_reason: str
    tool_invocations: list[ToolInvocation] = field(default_factory=list)
    iterations: int = 1
    elapsed_ms: float = 0.0


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

    async def complete_with_tools(
        self,
        messages: list[ChatMessage],
        *,
        registry: ToolRegistry,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        max_iterations: int | None = None,
    ) -> ChatCompletionWithTools:
        temp = temperature if temperature is not None else self._settings.openai_chat_temperature
        max_out = max_output_tokens or self._settings.openai_chat_max_output_tokens
        max_iter = max_iterations or self._settings.max_tool_iterations
        tools_payload = registry.to_openai_payload()

        payload: list[dict[str, Any]] = [m.as_dict() for m in messages]
        invocations: list[ToolInvocation] = []
        accumulated_prompt_tokens = 0
        accumulated_completion_tokens = 0
        last_model = ""
        last_finish = ""
        loop_started = time.perf_counter()

        for iteration in range(1, max_iter + 1):
            response = await self._invoke_chat(
                payload=payload,
                temp=temp,
                max_out=max_out,
                tools=tools_payload,
            )
            usage = response.usage
            accumulated_prompt_tokens += usage.prompt_tokens if usage else 0
            accumulated_completion_tokens += usage.completion_tokens if usage else 0
            last_model = response.model
            choice = response.choices[0]
            last_finish = choice.finish_reason or ""
            assistant_message = choice.message

            if not assistant_message.tool_calls:
                if assistant_message.content is None:
                    raise MalformedLLMResponse("OpenAI returned empty content.")
                return ChatCompletionWithTools(
                    content=assistant_message.content,
                    prompt_tokens=accumulated_prompt_tokens,
                    completion_tokens=accumulated_completion_tokens,
                    model=last_model,
                    finish_reason=last_finish,
                    tool_invocations=invocations,
                    iterations=iteration,
                    elapsed_ms=(time.perf_counter() - loop_started) * 1000,
                )

            payload.append(_assistant_message_dict(assistant_message))
            for tool_call in assistant_message.tool_calls:
                arguments = _safe_json(tool_call.function.arguments)
                started = time.perf_counter()
                result = await registry.invoke(tool_call.function.name, arguments)
                elapsed_ms = (time.perf_counter() - started) * 1000
                invocations.append(
                    ToolInvocation(
                        name=tool_call.function.name,
                        arguments=arguments,
                        output=result.output,
                        ok=result.ok,
                        error=result.error,
                        elapsed_ms=elapsed_ms,
                    )
                )
                logger.info(
                    "tool_call",
                    name=tool_call.function.name,
                    ok=result.ok,
                    error=result.error,
                    elapsed_ms=round(elapsed_ms, 1),
                )
                payload.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result.output,
                    }
                )

        raise ToolLoopExceeded(
            f"Tool-call loop exceeded {max_iter} iterations.",
            details={"iterations": max_iter, "invocations": [i.name for i in invocations]},
        )

    async def _invoke_chat(
        self,
        *,
        payload: list[dict[str, Any]],
        temp: float,
        max_out: int,
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": self._settings.openai_chat_model,
            "messages": payload,
            "temperature": temp,
            "max_completion_tokens": max_out,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self._settings.openai_chat_max_retries),
                wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
                retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
                reraise=True,
            ):
                with attempt:
                    return await self._client.chat.completions.create(**kwargs)
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


def _assistant_message_dict(assistant_message: Any) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": assistant_message.content,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in assistant_message.tool_calls
        ],
    }


def _safe_json(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}
