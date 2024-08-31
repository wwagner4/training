import math
import random as ran
from dataclasses import dataclass
from pprint import pprint
from typing import Tuple


@dataclass
class D:
    range: list[float]
    x: float
    y: float
    dx: float
    dy: float


@dataclass
class T:
    desc: str
    robot: Tuple[float, float, float]
    data: D
    see: dict[int, float]


datas1 = [
    D(
        range=[3.0 + i * 0.1 for i in range(20)],
        x=-350,
        y=400,
        dx=0,
        dy=-100,
    ),
    D(
        range=[0.0, 1.0, 2.0, 3.0, 4.0, 4.2, 4.5, 5.0, 6.0, 7.0, 8.0],
        x=-400,
        y=-200,
        dx=100,
        dy=0,
    ),
    D(
        range=[0.0, 1.0, 1.4, 1.6, 2.0, 3.0, 3.6, 3.7, 4.0, 5.0, 6.0],
        x=-200,
        y=400,
        dx=100,
        dy=-100,
    ),
    D(
        range=[0.0, 1.0, 2.0, 3.0, 3.4, 3.5, 4.0, 5.0],
        x=-200,
        y=400,
        dx=50,
        dy=-150,
    ),
    D(
        range=[0.0, 1.0, 2.0, 3.0, 4.0, 4.5, 4.6, 5.0, 6.0],
        x=100,
        y=400,
        dx=50,
        dy=-100,
    ),
]

datas2 = [
    D(
        range=range(7),
        x=250,
        y=75,
        dx=0,
        dy=-25,
    ),
    D(
        range=range(9),
        x=-125,
        y=375,
        dx=25,
        dy=0,
    ),
    D(
        range=range(7),
        x=-225,
        y=75,
        dx=0,
        dy=-25,
    ),
    D(
        range=range(5),
        x=-50,
        y=-150,
        dx=25,
        dy=0,
    ),
]

test_data1 = [
    # A
    T(desc="A1", robot=(0, 0, 0), data=datas1[0], see={}),
    T(desc="A2", robot=(0, 0, 0), data=datas1[1], see={}),
    T(desc="A3", robot=(0, 0, 0), data=datas1[2], see={7: 280.0, 8: 300.0}),
    T(desc="A4", robot=(0, 0, 0), data=datas1[3], see={}),
    T(desc="A5", robot=(0, 0, 0), data=datas1[4], see={4: 300.0, 5: 340.0}),
    # B
    T(desc="B1", robot=(0, 0, math.pi / 2.0), data=datas1[0], see={}),
    T(desc="B2", robot=(0, 0, math.pi / 2.0), data=datas1[1], see={}),
    T(desc="B3", robot=(0, 0, math.pi / 2.0), data=datas1[2], see={3: 280.0, 4: 200.0}),
    T(desc="B4", robot=(0, 0, math.pi / 2.0), data=datas1[3], see={}),
    T(desc="B5", robot=(0, 0, math.pi / 2.0), data=datas1[4], see={}),
    # C
    T(desc="C1", robot=(0, 0, math.pi), data=datas1[0], see={4: 350.0, 5: 355.0}),
    T(desc="C2", robot=(0, 0, math.pi), data=datas1[1], see={}),
    T(desc="C3", robot=(0, 0, math.pi), data=datas1[2], see={}),
    T(desc="C4", robot=(0, 0, math.pi), data=datas1[3], see={}),
    T(desc="C5", robot=(0, 0, math.pi), data=datas1[4], see={}),
    # D
    T(desc="D1", robot=(0, 0, -math.pi / 2.0), data=datas1[0], see={}),
    T(
        desc="D2",
        robot=(0, 0, -math.pi / 2.0),
        data=datas1[1],
        see={4: 200.0, 5: 201.0},
    ),
    T(desc="D3", robot=(0, 0, -math.pi / 2.0), data=datas1[2], see={}),
    T(
        desc="D4",
        robot=(0, 0, -math.pi / 2.0),
        data=datas1[3],
        see={5: 120.0, 6: 200.0, 7: 350.0},
    ),
    T(desc="D5", robot=(0, 0, -math.pi / 2.0), data=datas1[4], see={}),
]

test_data2 = [
    T(desc="T2_C1", robot=(0, 0, math.pi), data=datas1[0], see={4: 350.0}),
]

test_data3 = [
    T(desc="T3_A1", robot=(0, 0, 0), data=datas2[0], see={}),
    T(desc="T3_A2", robot=(0, 0, 0), data=datas2[1], see={}),
    T(desc="T3_A3", robot=(0, 0, 0), data=datas2[2], see={}),
    T(desc="T3_A4", robot=(0, 0, 0), data=datas2[3], see={}),
    T(desc="T3_B1", robot=(0, 0, math.pi / 2), data=datas2[0], see={}),
    T(desc="T3_B2", robot=(0, 0, math.pi / 2), data=datas2[1], see={}),
    T(desc="T3_B3", robot=(0, 0, math.pi / 2), data=datas2[2], see={}),
    T(desc="T3_B4", robot=(0, 0, math.pi / 2), data=datas2[3], see={}),
    T(desc="T3_C1", robot=(0, 0, math.pi), data=datas2[0], see={}),
    T(desc="T3_C2", robot=(0, 0, math.pi), data=datas2[1], see={}),
    T(desc="T3_C3", robot=(0, 0, math.pi), data=datas2[2], see={}),
    T(desc="T3_C4", robot=(0, 0, math.pi), data=datas2[3], see={}),
    T(desc="T3_D1", robot=(0, 0, -math.pi / 2), data=datas2[0], see={}),
    T(desc="T3_D2", robot=(0, 0, -math.pi / 2), data=datas2[1], see={}),
    T(desc="T3_D3", robot=(0, 0, -math.pi / 2), data=datas2[2], see={}),
    T(desc="T3_D4", robot=(0, 0, -math.pi / 2), data=datas2[3], see={}),
]


def pos(r: D, i: float) -> Tuple[float, float]:
    x = r.x + i * r.dx
    y = r.y + i * r.dy
    return x, y


def ran_direction() -> float:
    return ran.random() * 2.0 * math.pi


def t1(ox: float, oy: float, i: int, t: T) -> str:
    rx = t.robot[0]
    ry = t.robot[1]
    rdir = t.robot[2]
    odir = ran_direction()
    dist = t.see.get(i)
    sdist = f"{dist:.2f}" if dist else "None"
    desc = f"can_see_{t.desc}_{i}"
    return (
        f"    ('{desc}', ({rx:.2f}, {ry:.2f}, {rdir:.2f}), "
        f"({ox:.2f}, {oy:.2f}, {odir:.2f}), {sdist}),"
    )


def flatten(xss):
    return [x for xs in xss for x in xs]


def mktest(t: T) -> list[str]:
    poss = [pos(t.data, i) for i in t.data.range]
    return [t1(o[0], o[1], i, t) for (i, o) in enumerate(poss)]


def print_tests():
    td = test_data3
    ts = flatten([mktest(t) for t in td])
    for line in ts:
        print(line)


def print_poss():
    d = datas1[2]
    poss = [pos(d, i) for i in d.range]
    pprint(poss)
