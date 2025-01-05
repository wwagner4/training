import math
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def row_col(n: int) -> tuple[int, int]:
    if n < 4:
        return 1, n
    rows = int(math.ceil(math.sqrt(float(n))))
    cols = int(math.ceil(float(n) / rows))
    return rows, cols


def write_dict_data(data: pd.DataFrame, out_dir: Path, name: str) -> Path:
    filename = f"{name}.json"
    file = out_dir / filename
    data.to_json(file, indent=2)
    return file


def create_lines(desc: dict, line_index_list: list[list[int]]) -> str:
    """
    Creates a multiline string from key value pairs
    :param desc: The key value pairs stored in a dict
    :param line_index_list: List of index for lines
    :return:
    """
    k = dict([x for x in enumerate(desc)])

    def elem(i: int) -> str:
        key = k[i]
        value = desc[key]
        return f"{key}:{value}"

    def line(index: list[int]) -> str:
        return " ".join([elem(i) for i in index])

    return "\n".join([line(a) for a in line_index_list])


def unique() -> str:
    return str(uuid.uuid4())[0:6]


def cont_to_discrete(
    value: float, min_value: float, max_value: float, step_count: int
) -> int:
    d = (max_value - min_value) / step_count
    i = int(math.floor((value - min_value) / d))
    return min(max(0, i), (step_count - 1))


def cont_values(min_value: float, max_value: float, n: int) -> list[float]:
    diff = (max_value - min_value) / (n - 1)
    return [min_value + i * diff for i in range(n)]


def compress_means(data: list[float], n: int) -> list[list[float], list[float]]:
    data_len = len(data)
    if data_len <= n:
        return range(data_len), data
    d = np.array(data)
    cropped = (data_len // n) * n
    split = np.split(d[0:cropped], n)
    diff = cropped // n
    xs = range(0, cropped, diff)
    return xs, np.mean(split, axis=1)


def progress_str(nr: int, count: int, start_time: datetime) -> str:
    def f(t: datetime) -> str:
        return t.strftime("%H:%M:%S")

    def f1(seconds: int) -> str:
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

    start = f(start_time)
    now = datetime.now()
    if nr == 0:
        return f"{nr + 1}/{count} since::{start}"
    diff = now - start_time
    step_time = diff / nr
    until_time = start_time + (count * step_time)
    until = f(until_time)
    _for = f1(int((until_time - now).total_seconds()))
    _for_all = f1(int((until_time - start_time).total_seconds()))
    return f"{nr + 1}/{count} {start} -> {until} {_for}/{_for_all}"


def create_values(n: int, min: float, max: float) -> list[float]:
    diff = (max - min) / (n - 1)
    return [min + x * diff for x in range(n)]


def parse_integers(integers: str) -> list[int]:
    if not integers:
        return []
    split = integers.split(",")
    return [int(i.strip()) for i in split]
