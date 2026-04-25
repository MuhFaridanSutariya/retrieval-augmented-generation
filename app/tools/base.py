from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

from app.core.exceptions import ToolNotFound, ToolValidationError


@dataclass(slots=True)
class ToolCallResult:
    name: str
    ok: bool
    output: str
    error: str | None = None


@dataclass(slots=True)
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: Callable[..., Awaitable[str]]

    def to_openai_format(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_model.model_json_schema(),
            },
        }


class ToolRegistry:
    def __init__(self, *, max_output_chars: int = 4000) -> None:
        self._tools: dict[str, Tool] = {}
        self._max_output_chars = max_output_chars

    def register(self, tool: Tool) -> None:
        # Sanity-check the args model once at registration so we fail fast
        # if someone wires up a tool with a broken schema.
        try:
            tool.args_model.model_json_schema()
        except Exception as exc:
            raise ToolValidationError(
                f"Tool {tool.name!r} has an invalid args model: {exc}"
            ) from exc
        if tool.name in self._tools:
            raise ToolValidationError(f"Tool {tool.name!r} is already registered.")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise ToolNotFound(
                f"Unknown tool {name!r}.",
                details={"available": sorted(self._tools)},
            )
        return self._tools[name]

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def to_openai_payload(self) -> list[dict[str, Any]]:
        return [t.to_openai_format() for t in self._tools.values()]

    async def invoke(self, name: str, raw_arguments: dict[str, Any]) -> ToolCallResult:
        try:
            tool = self.get(name)
        except ToolNotFound as exc:
            return ToolCallResult(name=name, ok=False, output=str(exc), error="ToolNotFound")

        try:
            validated = tool.args_model.model_validate(raw_arguments)
        except ValidationError as exc:
            return ToolCallResult(
                name=name,
                ok=False,
                output=f"invalid arguments: {exc.errors()}",
                error="ValidationError",
            )

        try:
            output = await tool.handler(**validated.model_dump())
        except Exception as exc:
            return ToolCallResult(
                name=name,
                ok=False,
                output=str(exc),
                error=type(exc).__name__,
            )

        truncated = output if len(output) <= self._max_output_chars else output[: self._max_output_chars] + "…[truncated]"
        return ToolCallResult(name=name, ok=True, output=truncated, error=None)
