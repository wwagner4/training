import pytest

from training.helper import row_col

row_col_testdata = [
    (10, (4, 3)),
    (13, (4, 4)),
]


@pytest.mark.parametrize("n, expected", row_col_testdata)
def test_row_col(n: int, expected: (int, int)):
    result = row_col(n)
    assert result == expected
