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


compress_means_data = [
    (1, 1),
    (1, 2),
    (1, 3),
    (1, 5),
    (1, 10),
    (1, 100),
    (10, 1),
    (10, 2),
    (10, 3),
    (10, 5),
    (10, 10),
    (10, 100),
    (100, 1),
    (100, 2),
    (100, 3),
    (100, 5),
    (100, 10),
    (100, 100),
    (0, 1),
    (0, 10),
    (0, 0),
]


@pytest.mark.parametrize("len_data, n", compress_means_data)
def test_compress_means(len_data: int, n: int):
    data = [float(n) for n in range(len_data - 1)]
    xs, ys = hlp.compress_means(data, n)
    assert len(xs) == len(ys)


# split_data_for_boxplot
split_data_for_boxplot_data = [
    (1, 1),
    (1, 2),
    (1, 3),
    (1, 5),
    (1, 10),
    (1, 100),
    (10, 1),
    (10, 2),
    (10, 3),
    (10, 5),
    (10, 10),
    (10, 100),
    (100, 1),
    (100, 2),
    (100, 3),
    (100, 5),
    (100, 10),
    (100, 100),
    (0, 1),
    (0, 10),
    (0, 0),
]


@pytest.mark.parametrize("len_data, n", compress_means_data)
def test_split_data_for_boxplot(len_data: int, n: int):
    data = [float(n) for n in range(len_data - 1)]
    xs, ys = hlp.split_data_for_boxplot(data, n)
    assert len(xs) == len(ys)
