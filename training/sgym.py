from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict, Discrete

import training.simrunner as sr


@dataclass(frozen=True)
class SEnvConfig:
    max_wheel_speed: float
    max_view_distance: float
    dtype: np.generic = np.float32


default_senv_config = SEnvConfig(
    max_wheel_speed=7,
    max_view_distance=700,
    dtype=np.float32,
)


class SEnv(gym.Env):
    def __init__(
        self,
        senv_config: SEnvConfig,
        port: int,
        sim_name: str,
        opponent: sr.Controller,
        reward_handler: sr.RewardHandler,
        record: bool,
    ):
        self.senv_config = senv_config
        self.port = port
        self.sim_name = sim_name
        self.opponent_controller = opponent
        self.reward_handler = reward_handler
        self.record = record

        self.sim_action_response: sr.SensorResponse | None = None

        self.action_space = crete_action_space(senv_config)
        self.observation_space = create_observation_space(senv_config)

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        response = sr.reset(
            self.port,
            self.sim_name,
            "GYM",
            {},
            self.opponent_controller.name(),
            self.opponent_controller.description(),
            self.reward_handler,
            self.record,
        )
        match response:
            case sr.SensorResponse(sensor1=sensor1):
                self.sim_action_response = response
                return mapping_sensor_to_observation_space(sensor1, self.senv_config)
            case sr.ErrorResponse(msg):
                raise RuntimeError(f"Error on reset: '{msg}'")
            case sr.FinishedResponse(msg):
                raise RuntimeError(f"Error on reset: Finished immediately '{msg}'")

    def step(self, action):
        sensor2 = self.sim_action_response.sensor2
        cnt = self.sim_action_response.cnt
        request = sr.ActionRequest(
            diffDrive1=mapping_action_space_to_diff_drive(action),
            diffDrive2=self.opponent_controller.take_step(sensor2),
            simulation_states=self.sim_action_response.simulation_states,
            obj_id=self.sim_action_response.obj_id,
            cnt=cnt + 1,
        )
        response = sr.step(request, self.reward_handler, self.port)
        match response:
            case sr.SensorResponse(sensor1=sensor1, reward=reward):
                self.sim_action_response = response
                observation = mapping_sensor_to_observation_space(
                    sensor1, self.senv_config
                )
                reward = reward
                terminated = False
                truncated = False
                info = {}
                return observation, reward[0], terminated, truncated, info
            case sr.FinishedResponse(reward, message):
                observation = {}
                reward = reward
                terminated = True
                truncated = False
                info = {"status": "OK", "message": message}
                return observation, reward[0], terminated, truncated, info
            case sr.ErrorResponse(message):
                observation = {}
                terminated = True
                truncated = True
                info = {"status": "ERROR", "message": message}
                return observation, 0.0, terminated, truncated, info


# Define action and observation space
def crete_action_space(config: SEnvConfig) -> gym.Space:
    return Box(
        low=-config.max_wheel_speed,
        high=config.max_wheel_speed,
        shape=(1, 2),
        dtype=config.dtype,
    )


def create_observation_space(config: SEnvConfig) -> gym.Space:
    observation_view_space = Discrete(n=4)
    observation_border_space = Box(
        low=0.0, high=config.max_view_distance, shape=(1, 3), dtype=config.dtype
    )
    return Dict(
        {
            "view": observation_view_space,
            "border": observation_border_space,
        }
    )


def mapping_sensor_to_observation_space(
    sensor: sr.CombiSensor, config: SEnvConfig
) -> dict[str, any]:
    def view_mapping() -> int:
        match sensor.opponent_in_sector:
            case sr.SectorName.UNDEF:
                return 0
            case sr.SectorName.LEFT:
                return 1
            case sr.SectorName.CENTER:
                return 2
            case sr.SectorName.RIGHT:
                return 3
            case _:
                raise ValueError(f"Wrong sector name {sensor.opponent_in_sector}")

    return {
        "view": view_mapping(),
        "border": _create_numpy_array(
            [
                [
                    sensor.left_distance,
                    sensor.front_distance,
                    sensor.right_distance,
                ]
            ],
            config,
        ),
    }


def mapping_action_space_to_diff_drive(action_space: list[list]) -> sr.DiffDriveValues:
    return sr.DiffDriveValues(
        left_velo=action_space[0][1],
        right_velo=action_space[0][0],
    )


def _create_numpy_array(value: Any, config: SEnvConfig) -> np.array:
    return np.array(value, dtype=config.dtype)
