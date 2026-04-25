import ast
import operator
from typing import Any

from pydantic import BaseModel, Field

from app.tools.base import Tool

_BIN_OPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: dict[type, Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class CalculatorArgs(BaseModel):
    expression: str = Field(
        description=(
            "An arithmetic expression using only numbers and the operators +, -, *, /, //, %, **, "
            "and parentheses. Examples: '12400 * 1.356', '(17940 - 13335) / 13335 * 100'. "
            "Variables, function calls, and any non-numeric values are rejected."
        ),
    )


def _evaluate(expression: str) -> float:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"could not parse {expression!r}: {exc.msg}") from exc
    return _eval_node(tree.body)


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(f"unsupported constant: {node.value!r}")
        return float(node.value)
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported binary operator: {type(node.op).__name__}")
        return op(_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported unary operator: {type(node.op).__name__}")
        return op(_eval_node(node.operand))
    raise ValueError(f"unsupported expression node: {type(node).__name__}")


async def _calculate(expression: str) -> str:
    result = _evaluate(expression)
    if isinstance(result, float) and result.is_integer():
        return f"{int(result)}"
    return f"{result:.6g}"


CALCULATOR_TOOL = Tool(
    name="calculate",
    description=(
        "Evaluate an arithmetic expression and return the numeric result as a string. "
        "Use this for any computation involving numbers from the documents — percentage "
        "changes, totals, ratios, averages — instead of computing them in your head. "
        "The result is exact to 6 significant figures."
    ),
    args_model=CalculatorArgs,
    handler=_calculate,
)
