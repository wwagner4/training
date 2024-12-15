"""
Dummy controller that runs in a circle.
First dummy controller to test the sumosim simrunner udp connection
To be deleted soon
"""

import training.simrunner as sr


def _description(left: float, right: float) -> dict:
    return {
        "description": "Drives with a constant speed for each wheel",
        "left wheel": left,
        "right wheel": right,
    }


class FastCircleController(sr.Controller):
    left_wheel = 0.9
    right_wheel = 0.8

    def take_step(self, sensor: sr.CombiSensor) -> sr.DiffDriveValues:
        return sr.DiffDriveValues(self.left_wheel, self.right_wheel)

    def name(self) -> str:
        return "fast-circle"

    def description(self) -> dict:
        return _description(self.left_wheel, self.right_wheel)


class SlowCircleController(sr.Controller):
    left_wheel = 0.2
    right_wheel = 0.4

    def take_step(self, sensor: sr.CombiSensor) -> sr.DiffDriveValues:
        return sr.DiffDriveValues(self.left_wheel, self.right_wheel)

    def name(self) -> str:
        return "slow-circle"

    def description(self) -> dict:
        return _description(self.left_wheel, self.right_wheel)
