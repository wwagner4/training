"""
A simple controller that ignores the other robot
but tries to stay inside the field
"""

import random
from enum import Enum

import training.simrunner as sr


class State(Enum):
    FORWARD = "forward"
    TURN = "turn"


def _description() -> dict:
    return {
        "description": "Stays in the field but ignores the other robot",
    }


class StayInFieldController(sr.Controller):
    state = State.FORWARD
    turn_count = 0
    turn_right = True
    max_turn_count = 80

    def take_step(self, sensor: sr.CombiSensor) -> sr.DiffDriveValues:
        # print("####>> ", sensor.front_distance, self.turn_count)
        if self.state == State.TURN:
            if self.turn_count >= self.max_turn_count:
                self.state = State.FORWARD
        elif self.state == State.FORWARD and sensor.front_distance < 50:
            self.turn_right = random.choice([True, False])
            self.turn_count = 0
            self.max_turn_count = random.randint(70, 90)
            self.state = State.TURN

        # print("## STAY IN FIELD", self.state, sensor.front_distance, self.turn_count)
        match self.state:
            case State.FORWARD:
                return sr.DiffDriveValues(5, 5)
            case State.TURN:
                self.turn_count += 1
                if self.turn_right:
                    return sr.DiffDriveValues(-1, 1)
                else:
                    return sr.DiffDriveValues(1, -1)

    def name(self) -> str:
        return "stay-in-field"

    def description(self) -> dict:
        return _description()
