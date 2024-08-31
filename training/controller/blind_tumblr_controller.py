"""
A robot that tumbles around, tries to stay inside the field and
kicks out the opponent if it sees it.
"""

import random
from enum import Enum

import training.simrunner as sr


class State(Enum):
    FORWARD_RIGHT = "forward_right"
    FORWARD_LEFT = "forward_left"


def _description() -> dict:
    return {
        "description": "Tumbles around",
    }


def _ran_forward_dir() -> State:
    return random.choice([State.FORWARD_RIGHT, State.FORWARD_LEFT])


def _ran_forward_count_max() -> int:
    return random.randint(100, 200)


def _ran_fast_wheel() -> float:
    return 4.8 + random.random() * 0.4


def _ran_slow_wheel() -> float:
    return 2.2 + random.random() * 0.4


class BlindTumblrController(sr.Controller):
    state = _ran_forward_dir()
    forward_count = 0
    forward_count_max = _ran_forward_count_max()
    fast_wheel = _ran_fast_wheel()
    slow_wheel = _ran_slow_wheel()

    def take_step(self, sensor: sr.CombiSensor) -> sr.DiffDriveValues:
        # print("####>> ", sensor.front_distance, self.turn_count)
        if (
            self.state == State.FORWARD_LEFT
            and self.forward_count >= self.forward_count_max
        ):
            self.state = State.FORWARD_RIGHT
            self.forward_count_max = _ran_forward_count_max()
            self.forward_count = 0
            self.fast_wheel = _ran_fast_wheel()
            self.slow_wheel = _ran_slow_wheel()
        elif (
            self.state == State.FORWARD_RIGHT
            and self.forward_count >= self.forward_count_max
        ):
            self.state = State.FORWARD_LEFT
            self.forward_count_max = _ran_forward_count_max()
            self.forward_count = 0
            self.fast_wheel = _ran_fast_wheel()
            self.slow_wheel = _ran_slow_wheel()

        # print("## BLIND TUMBLR", self.state, sensor.front_distance)
        match self.state:
            case State.FORWARD_RIGHT:
                self.forward_count += 1
                return sr.DiffDriveValues(self.slow_wheel, self.fast_wheel)
            case State.FORWARD_LEFT:
                self.forward_count += 1
                return sr.DiffDriveValues(self.fast_wheel, self.slow_wheel)

    def name(self) -> str:
        return "blind-tumblr"

    def description(self) -> dict:
        return _description()
