import math
from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import numpy as np
from dataclasses_json import dataclass_json

import training.consts as consts
import training.reward_helper as rh


@dataclass
class RobotState:
    xpos: float
    ypos: float
    direction: float


@dataclass_json
@dataclass
class RobotsState:
    robot1: RobotState
    robot2: RobotState


@dataclass
class Sim:
    name: str
    winner: int  # 0: none, 1: robot1, 2: robot2
    states: list[RobotsState]


class RobotEventsResult(Enum):
    WINNER = "winner"
    LOOSER = "looser"
    DRAW = "draw"


class RobotEventsEnd(Enum):
    PUSH = "push"
    IS_PUSHED = "is-pushed"
    NONE = "none"


@dataclass
class RobotEvents:
    result: RobotEventsResult
    push_collision_count: int
    is_pushed_collision_count: int
    end: RobotEventsEnd


@dataclass
class SimEvents:
    robot1: RobotEvents
    robot2: RobotEvents


@dataclass
class RobotsSeeIntervals:
    robot1: list[rh.Interval]
    robot2: list[rh.Interval]


def can_see(robot1: RobotState, robot2: RobotState) -> float | None:
    def are_clockwise(v1, v2):
        return (-(v1[0] * v2[1]) + (v1[1] * v2[0])) < 0.0

    def is_point_in_sector(v_rel_point, v_start, v_end):
        c1 = are_clockwise(v_rel_point, v_end)
        c2 = are_clockwise(v_start, v_rel_point)
        return c1 and c2

    def inner(alpha: float, r: float) -> bool | None:
        start = robot1.direction - (alpha / 2.0)
        end = robot1.direction + (alpha / 2.0)
        pol_start = np.array([1.0, start])
        pol_end = np.array([1.0, end])

        v_start = rh.np_pol2cart(pol_start) * 400
        v_end = rh.np_pol2cart(pol_end) * 400
        v_origin = np.array([robot1.xpos, robot1.ypos])
        v_point = np.array([robot2.xpos, robot2.ypos])
        v_rel_point = v_point - v_origin

        if is_point_in_sector(v_rel_point, v_start, v_end):
            d = np.linalg.norm(v_point - v_origin).item()
            return d if d <= r else None
        return None

    i1 = inner(math.radians(20.0), 400)
    return i1 if i1 else inner(math.radians(80.0), 150)


def see_intervals(states: list[RobotsState]) -> RobotsSeeIntervals:
    def to_bool(see: float | None) -> bool:
        return see is not None

    r1_see = [to_bool(can_see(s.robot1, s.robot2)) for s in states]
    r2_see = [to_bool(can_see(s.robot2, s.robot1)) for s in states]

    r1_see_intervals = rh.intervals_boolean(r1_see)
    r2_see_intervals = rh.intervals_boolean(r2_see)

    return RobotsSeeIntervals(
        robot1=r1_see_intervals,
        robot2=r2_see_intervals,
    )


def ends(
    collision_end_interval: rh.Interval | None,
    robot_see_intervals: RobotsSeeIntervals,
    winner: int,
) -> Tuple[RobotEventsEnd, RobotEventsEnd]:
    def is_winner_push(
        end_interval: rh.Interval, see_intervals: list[rh.Interval]
    ) -> bool:
        return any(rh.overlapping(si, end_interval) for si in see_intervals)

    if collision_end_interval:
        if winner == 1:
            robot1_push_out = any(
                rh.overlapping(si, collision_end_interval)
                for si in robot_see_intervals.robot1
            )
            if robot1_push_out:
                return RobotEventsEnd.PUSH, RobotEventsEnd.IS_PUSHED
            return RobotEventsEnd.NONE, RobotEventsEnd.NONE
        elif winner == 2:
            robot2_push_out = any(
                rh.overlapping(si1, collision_end_interval)
                for si1 in robot_see_intervals.robot2
            )
            if robot2_push_out:
                return RobotEventsEnd.IS_PUSHED, RobotEventsEnd.PUSH
            return RobotEventsEnd.NONE, RobotEventsEnd.NONE
        return RobotEventsEnd.NONE, RobotEventsEnd.NONE
    return RobotEventsEnd.NONE, RobotEventsEnd.NONE


def sim_events(sim: Sim) -> SimEvents:
    r1_result = RobotEventsResult.DRAW
    r2_result = RobotEventsResult.DRAW
    if sim.winner == 1:
        r1_result = RobotEventsResult.WINNER
        r2_result = RobotEventsResult.LOOSER
    elif sim.winner == 2:
        r1_result = RobotEventsResult.LOOSER
        r2_result = RobotEventsResult.WINNER

    distances = list([dist(s) for s in sim.states])
    collision_intervals: rh.CollisionIntervals = rh.intervals_from_threshold(
        distances, consts.ROBOT_DIAMETER + 5, int(consts.FIELD_DIAMETER)
    )
    see_interval = see_intervals(sim.states)

    coll_count1 = rh.collisions_count(collision_intervals.inner, see_interval.robot1)
    coll_count2 = rh.collisions_count(collision_intervals.inner, see_interval.robot2)

    r1_end, r2_end = ends(collision_intervals.end, see_interval, sim.winner)

    return SimEvents(
        RobotEvents(
            result=r1_result,
            push_collision_count=coll_count1.push_count,
            is_pushed_collision_count=coll_count1.is_pushed_count,
            end=r1_end,
        ),
        RobotEvents(
            result=r2_result,
            push_collision_count=coll_count2.push_count,
            is_pushed_collision_count=coll_count2.is_pushed_count,
            end=r2_end,
        ),
    )


def dist(state: RobotsState) -> float:
    r1 = state.robot1
    r2 = state.robot2
    dx = r1.xpos - r2.xpos
    dy = r1.ypos - r2.ypos
    return math.sqrt(dx * dx + dy * dy)
