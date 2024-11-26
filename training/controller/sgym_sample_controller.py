"""
A controller that uses the gymnasium sample function
to move around.
Needed to check out if the reward function creates some values different from 0
in order to start the training process.
"""

import numpy as np

import training.sgym as sgym
import training.simrunner as sr


def _description() -> dict:
    return {
        "description": "Use gymnasium actions and sample function",
    }


def sgym_config():
    return sgym.SEnvConfig(
        max_wheel_speed=7,
        max_view_distance=700,
        dtype=np.float32,
        max_simulation_steps=1000,
    )


class SGymSampleController(sr.Controller):
    def __init__(
        self,
    ):
        self.action_space = sgym.cont_act_space(sgym_config())

    def take_step(self, sensor: sr.CombiSensor) -> sr.DiffDriveValues:
        action = self.action_space.sample()
        ddv = sgym.map_cont_act_to_diff_drive(action)
        return ddv

    def name(self) -> str:
        return "sgym-sample"

    def description(self) -> dict:
        return _description()
