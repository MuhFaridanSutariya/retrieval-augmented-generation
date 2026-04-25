import pytest
from pydantic import BaseModel, Field

from app.core.exceptions import ToolNotFound, ToolValidationError
from app.tools.base import Tool, ToolCallResult, ToolRegistry


class _EchoArgs(BaseModel):
    value: str = Field(description="any string")


async def _echo_handler(value: str) -> str:
    return value.upper()


class _AddArgs(BaseModel):
    a: int
    b: int


async def _add_handler(a: int, b: int) -> str:
    return str(a + b)


def _make_tool(name: str = "echo") -> Tool:
    return Tool(
        name=name,
        description=f"echo {name}",
        args_model=_EchoArgs,
        handler=_echo_handler,
    )


def test_register_and_get_returns_the_same_tool() -> None:
    registry = ToolRegistry()
    tool = _make_tool()
    registry.register(tool)
    assert registry.get("echo") is tool
    assert "echo" in {t.name for t in registry.all()}


def test_double_registration_raises() -> None:
    registry = ToolRegistry()
    registry.register(_make_tool())
    with pytest.raises(ToolValidationError):
        registry.register(_make_tool())


def test_get_unknown_tool_raises() -> None:
    registry = ToolRegistry()
    with pytest.raises(ToolNotFound):
        registry.get("missing")


@pytest.mark.asyncio
async def test_invoke_returns_ok_result_on_success() -> None:
    registry = ToolRegistry()
    registry.register(_make_tool())
    result = await registry.invoke("echo", {"value": "hi"})
    assert isinstance(result, ToolCallResult)
    assert result.ok is True
    assert result.output == "HI"
    assert result.error is None


@pytest.mark.asyncio
async def test_invoke_unknown_tool_returns_error_result_not_raise() -> None:
    registry = ToolRegistry()
    result = await registry.invoke("nope", {})
    assert result.ok is False
    assert result.error == "ToolNotFound"


@pytest.mark.asyncio
async def test_invoke_with_invalid_arguments_returns_validation_error() -> None:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="add",
            description="add two ints",
            args_model=_AddArgs,
            handler=_add_handler,
        )
    )
    result = await registry.invoke("add", {"a": "not-an-int"})
    assert result.ok is False
    assert result.error == "ValidationError"


@pytest.mark.asyncio
async def test_invoke_handler_exception_is_captured() -> None:
    async def _boom(value: str) -> str:
        raise RuntimeError(f"boom: {value}")

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="boom",
            description="explodes",
            args_model=_EchoArgs,
            handler=_boom,
        )
    )
    result = await registry.invoke("boom", {"value": "kaboom"})
    assert result.ok is False
    assert result.error == "RuntimeError"
    assert "kaboom" in result.output


@pytest.mark.asyncio
async def test_invoke_truncates_oversized_output() -> None:
    async def _big(value: str) -> str:
        return "x" * 5000

    registry = ToolRegistry(max_output_chars=100)
    registry.register(
        Tool(
            name="big",
            description="returns a wall of text",
            args_model=_EchoArgs,
            handler=_big,
        )
    )
    result = await registry.invoke("big", {"value": "anything"})
    assert result.ok is True
    assert len(result.output) <= 100 + len("…[truncated]")
    assert result.output.endswith("…[truncated]")


def test_to_openai_format_emits_function_schema() -> None:
    tool = _make_tool()
    openai_format = tool.to_openai_format()
    assert openai_format["type"] == "function"
    assert openai_format["function"]["name"] == "echo"
    assert openai_format["function"]["parameters"]["type"] == "object"
    assert "value" in openai_format["function"]["parameters"]["properties"]


def test_registry_to_openai_payload_lists_all_tools() -> None:
    registry = ToolRegistry()
    registry.register(_make_tool("a"))
    registry.register(_make_tool("b"))
    payload = registry.to_openai_payload()
    assert len(payload) == 2
    names = {entry["function"]["name"] for entry in payload}
    assert names == {"a", "b"}
