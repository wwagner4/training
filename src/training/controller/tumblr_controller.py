"""
A robot that tumbles around, tries to stay inside the field and
kicks out the opponent if it sees it.
"""

import random
from enum import Enum

import training.simrunner as sr


class State(Enum):
    ATTACK_LEFT = "attack_left"
    ATTACK_CENTER = "attack_center"
    ATTACK_RIGHT = "attack_right"
    FORWARD_RIGHT = "forward_right"
    FORWARD_LEFT = "forward_left"
    TURN = "turn"


def _description() -> dict:
    return {
        "description": "Tumbles around and tries to kick out the "
        "opponent if it sees it",
    }


def _ran_forward_dir() -> State:
    return random.choice([State.FORWARD_RIGHT, State.FORWARD_LEFT])


def _ran_forward_count_max() -> int:
    return random.randint(40, 80)


def _is_forward(state: State) -> bool:
    return state == State.FORWARD_RIGHT or state == State.FORWARD_LEFT


def _is_lr_attack(state: State) -> bool:
    return state == State.ATTACK_RIGHT or state == State.ATTACK_LEFT


class TumblrController(sr.Controller):
    state = _ran_forward_dir()
    turn_count = 0
    turn_right = True
    turn_count_max = 80
    forward_count = 0
    forward_count_max = _ran_forward_count_max()

    def take_step(self, sensor: sr.CombiSensor) -> sr.DiffDriveValues:
        # print("####>> ", sensor.front_distance, self.turn_count)
        if self.state == State.TURN and self.turn_count >= self.turn_count_max:
            self.state = _ran_forward_dir()
            self.forward_count = 0
        else:
            if sensor.opponent_in_sector == sr.SectorName.RIGHT:
                self.state = State.ATTACK_RIGHT
            elif sensor.opponent_in_sector == sr.SectorName.CENTER:
                self.state = State.ATTACK_CENTER
            elif sensor.opponent_in_sector == sr.SectorName.LEFT:
                self.state = State.ATTACK_LEFT
            elif (
                _is_forward(self.state) or _is_lr_attack(self.state)
            ) and sensor.front_distance < 50:
                self.turn_right = random.choice([True, False])
                self.turn_count = 0
                self.turn_count_max = random.randint(70, 90)
                self.state = State.TURN
            elif (
                self.state == State.FORWARD_LEFT
                and self.forward_count >= self.forward_count_max
            ):
                self.state = State.FORWARD_RIGHT
                self.forward_count_max = _ran_forward_count_max()
                self.forward_count = 0
            elif (
                self.state == State.FORWARD_RIGHT
                and self.forward_count >= self.forward_count_max
            ):
                self.state = State.FORWARD_LEFT
                self.forward_count_max = _ran_forward_count_max()
                self.forward_count = 0

        # print("## TUMBLR", self.state, sensor.front_distance, self.turn_count)
        match self.state:
            case State.ATTACK_RIGHT:
                return sr.DiffDriveValues(4.5, 5)
            case State.ATTACK_CENTER:
                return sr.DiffDriveValues(6, 6)
            case State.ATTACK_LEFT:
                return sr.DiffDriveValues(5, 4.5)
            case State.FORWARD_RIGHT:
                self.forward_count += 1
                return sr.DiffDriveValues(2, 5)
            case State.FORWARD_LEFT:
                self.forward_count += 1
                return sr.DiffDriveValues(5, 2)
            case State.TURN:
                self.turn_count += 1
                if self.turn_right:
                    return sr.DiffDriveValues(-1, 1)
                else:
                    return sr.DiffDriveValues(1, -1)

    def name(self) -> str:
        return "tumblr"

    def description(self) -> dict:
        return _description()
