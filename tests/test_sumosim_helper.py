import pytest

import training.helper as hlp

row_col_testdata = [
    (10, (4, 3)),
    (13, (4, 4)),
]


@pytest.mark.parametrize("n, expected", row_col_testdata)
def test_row_col(n: int, expected: (int, int)):
    result = hlp.row_col(n)
    assert result == expected


parse_integers_data = [
    ("", []),
    ("1", [1]),
    ("1,200,4", [1, 200, 4]),
    ("1, 3", [1, 3]),
    ("1, \n3\n", [1, 3]),
    ("\t1,\t3\n", [1, 3]),
]


@pytest.mark.parametrize("integers, expected", parse_integers_data)
def test_parse_integers(integers: str, expected: list[int]) -> None:
    result = hlp.parse_integers(integers)
    assert result == expected
