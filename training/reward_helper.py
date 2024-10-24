from dataclasses import dataclass


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
