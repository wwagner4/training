import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class Interval:
    start: int
    end: int


@dataclass
class CollisionIntervals:
    inner: list[Interval]
    end: Interval | None


def intervals_from_threshold(
    values1: list[float], threshold: float, end_len: int
) -> CollisionIntervals:
    """

    :rtype: object
    """
    result = []
    in_interval = False
    start = 0
    for i in range(len(values1) - 1):
        a = values1[i]
        b = values1[i + 1]
        if not in_interval and a > threshold and b <= threshold:
            in_interval = True
            start = i
        elif in_interval and a <= threshold and b > threshold:
            in_interval = False
            result.append(Interval(start, i))
    if in_interval:
        return CollisionIntervals(result, Interval(start, start + end_len))
    return CollisionIntervals(result, None)


def overlapping(a: Interval, b: Interval) -> bool:
    # assume that a and b are valid intervals
    if a.end < b.start:
        return False
    return not a.start > b.end


def validate_interval(i: Interval):
    if i.start > i.end:
        raise ValueError(f"End value of interval must be grater than start value {i}")


def cart2pol(x: float, y: float) -> Tuple[float, float]:
    rho = np.sqrt(x**2 + y**2)
    phi = norm(np.arctan2(y, x))
    return rho, phi


def pol2cart(r: float, phi: float) -> Tuple[float, float]:
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return x, y


def np_cart2pol(cart: np.array) -> np.array:
    r = np.sqrt(cart[0] ** 2 + cart[1] ** 2)
    phi = norm(np.arctan2(cart[1], cart[0]))
    return np.array([r, phi])


def np_pol2cart(pol: np.array) -> np.array:
    x = pol[0] * np.cos(pol[1])
    y = pol[0] * np.sin(pol[1])
    return np.array([x, y])


def f_pol(pol: np.array) -> str:
    return f"[{pol[0]:.3f}, {pol[1]:.3f}r|{math.degrees(pol[1]):.3f}d]"


def norm(phi: float) -> float:
    if phi < 0.0:
        return phi + (math.pi * 2.0)
    return phi


def intervals_boolean(booleans: list[bool]) -> list[Interval]:
    in_interval = False
    start = 0
    result = []
    for i in range(len(booleans) - 1):
        a = booleans[i]
        b = booleans[i + 1]
        if not in_interval and not a and b:
            start = i
            in_interval = True
        elif in_interval and a and not b:
            in_interval = False
            result.append(Interval(start, i))
    if in_interval:
        result.append(Interval(start, len(booleans) - 1))
    return result


@dataclass
class CollisionsCount:
    push_count: int
    is_pushed_count: int


def collisions_count(
    collisions: list[Interval], sees: list[Interval]
) -> CollisionsCount:
    def any_match(collision: Interval) -> int:
        for see in sees:
            if overlapping(collision, see):
                return 1
        return 0

    push_count = sum([any_match(coll) for coll in collisions])
    return CollisionsCount(
        push_count=push_count,
        is_pushed_count=len(collisions) - push_count,
    )
