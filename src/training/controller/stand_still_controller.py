"""
Dummy controller that does not move at all
"""

import training.simrunner as sr


class StandStillController(sr.Controller):
    left_wheel = 0.0
    right_wheel = 0.0

    def take_step(self, sensor: sr.CombiSensor) -> sr.DiffDriveValues:
        return sr.DiffDriveValues(self.left_wheel, self.right_wheel)

    def name(self) -> str:
        return "stand-still"

    def description(self) -> dict:
        return {
            "description": "Robot which does not move at all"
        }

