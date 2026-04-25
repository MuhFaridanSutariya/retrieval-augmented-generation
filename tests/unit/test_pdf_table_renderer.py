from app.loaders.pdf_loader import _clean_cell, _table_to_markdown


def test_renders_simple_table_with_header_and_separator() -> None:
    table = [
        ["Quarter", "Revenue"],
        ["Q1", "100"],
        ["Q2", "200"],
    ]
    rendered = _table_to_markdown(table)
    assert rendered == (
        "| Quarter | Revenue |\n"
        "| --- | --- |\n"
        "| Q1 | 100 |\n"
        "| Q2 | 200 |"
    )


def test_pads_short_rows_to_table_width() -> None:
    table = [
        ["A", "B", "C"],
        ["1", "2"],
        ["x"],
    ]
    rendered = _table_to_markdown(table)
    assert "| 1 | 2 |  |" in rendered
    assert "| x |  |  |" in rendered


def test_replaces_none_cells_with_empty_string() -> None:
    table = [["A", "B"], [None, "v"]]
    rendered = _table_to_markdown(table)
    assert "|  | v |" in rendered


def test_collapses_newlines_inside_cells() -> None:
    table = [["Header"], ["line1\nline2"]]
    rendered = _table_to_markdown(table)
    assert "line1 line2" in rendered
    assert "\nline1" not in rendered


def test_escapes_pipe_characters_inside_cells() -> None:
    assert _clean_cell("a|b") == "a\\|b"


def test_returns_empty_for_completely_empty_table() -> None:
    assert _table_to_markdown([]) == ""
    assert _table_to_markdown([[None, None], [None, None]]) == ""


def test_single_row_table_renders_header_and_separator_only() -> None:
    rendered = _table_to_markdown([["Only", "Header"]])
    assert rendered == "| Only | Header |\n| --- | --- |"
