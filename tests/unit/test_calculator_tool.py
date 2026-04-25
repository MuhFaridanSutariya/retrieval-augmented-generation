import pytest

from app.tools.calculator_tool import _evaluate, _calculate


def test_evaluates_basic_arithmetic() -> None:
    assert _evaluate("1 + 2") == 3
    assert _evaluate("10 * 5") == 50
    assert _evaluate("100 / 4") == 25
    assert _evaluate("2 ** 10") == 1024


def test_evaluates_parenthesised_and_nested_expressions() -> None:
    assert _evaluate("(17940 - 13335) / 13335 * 100") == pytest.approx(34.531, rel=1e-3)
    assert _evaluate("(1 + 2) * (3 + 4)") == 21


def test_evaluates_negative_and_unary() -> None:
    assert _evaluate("-5 + 3") == -2
    assert _evaluate("+10") == 10
    assert _evaluate("--5") == 5


def test_evaluates_floor_div_and_modulo() -> None:
    assert _evaluate("17 // 5") == 3
    assert _evaluate("17 % 5") == 2


def test_rejects_function_calls() -> None:
    with pytest.raises(ValueError):
        _evaluate("__import__('os').system('rm -rf /')")
    with pytest.raises(ValueError):
        _evaluate("open('/etc/passwd').read()")


def test_rejects_attribute_access() -> None:
    with pytest.raises(ValueError):
        _evaluate("os.system")


def test_rejects_variable_references() -> None:
    with pytest.raises(ValueError):
        _evaluate("x + 1")


def test_rejects_string_constants() -> None:
    with pytest.raises(ValueError):
        _evaluate("'hello'")


def test_rejects_booleans() -> None:
    with pytest.raises(ValueError):
        _evaluate("True + 1")


def test_rejects_lambda() -> None:
    with pytest.raises(ValueError):
        _evaluate("(lambda: 1)()")


def test_rejects_invalid_syntax() -> None:
    with pytest.raises(ValueError):
        _evaluate("1 +")


@pytest.mark.asyncio
async def test_handler_returns_integer_form_when_result_is_whole() -> None:
    assert await _calculate("12 + 8") == "20"
    assert await _calculate("100 / 4") == "25"


@pytest.mark.asyncio
async def test_handler_returns_six_significant_figures_for_floats() -> None:
    output = await _calculate("(17940 - 13335) / 13335 * 100")
    assert output.startswith("34.5")
