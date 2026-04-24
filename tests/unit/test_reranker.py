from app.retrievers.reranker import _parse_order


def test_parse_order_extracts_clean_json_array() -> None:
    assert _parse_order("[3, 1, 2]", max_index=5) == [3, 1, 2]


def test_parse_order_handles_leading_and_trailing_text() -> None:
    raw = "Here are the rankings: [2, 4, 1] — hope that helps!"
    assert _parse_order(raw, max_index=5) == [2, 4, 1]


def test_parse_order_drops_out_of_range_indices() -> None:
    assert _parse_order("[1, 99, 3]", max_index=5) == [1, 3]


def test_parse_order_drops_non_integer_items() -> None:
    assert _parse_order("[1, \"two\", 3, null, 2.5]", max_index=5) == [1, 3]


def test_parse_order_returns_empty_on_invalid_json() -> None:
    assert _parse_order("not even close to json", max_index=5) == []


def test_parse_order_returns_empty_on_empty_input() -> None:
    assert _parse_order("", max_index=5) == []


def test_parse_order_rejects_non_array_json() -> None:
    assert _parse_order('{"order": [1, 2]}', max_index=5) == []


def test_parse_order_tolerates_zero_and_negative_indices() -> None:
    assert _parse_order("[0, -1, 2]", max_index=5) == [2]
