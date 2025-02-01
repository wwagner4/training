from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

import training.simrunner as sr


@dataclass(frozen=True)
class SEnvConfig:
    max_wheel_speed: float
    wheel_speed_steps: int
    max_view_distance: float
    view_distance_steps: int
    max_simulation_steps: int
    dtype: np.generic = np.float32


@dataclass(frozen=True)
class SEnvMapping:
    act_space: Callable[[SEnvConfig], gym.Space]
    obs_space: Callable[[SEnvConfig], gym.Space]
    map_act: Callable[[any, SEnvConfig], sr.DiffDriveValues]
    map_sensor: Callable[[sr.CombiSensor, SEnvConfig], any]


default_senv_config = SEnvConfig(
    max_wheel_speed=7,
    wheel_speed_steps=10,
    max_view_distance=700,
    view_distance_steps=10,
    max_simulation_steps=1000,
    dtype=np.float32,
)


class SEnv(gym.Env):
    def __init__(
        self,
        senv_config: SEnvConfig,
        senv_mapping: SEnvMapping,
        sim_host: str,
        sim_port: int,
        db_host: str,
        db_port: int,
        opponent: sr.Controller,
        reward_handler: sr.RewardHandler,
    ):
        self.senv_config = senv_config
        self.senv_mapping = senv_mapping
        self.sim_host = sim_host
        self.sim_port = sim_port
        self.db_host = db_host
        self.db_port = db_port
        self.sim_name = None
        self.opponent_controller = opponent
        self.reward_handler = reward_handler
        self.sim_info = None

        self.sim_action_response: sr.SensorResponse | None = None

        self.action_space = senv_mapping.act_space(senv_config)
        self.observation_space = senv_mapping.obs_space(senv_config)

    def reset(
        self,
        sim_info: sr.SimInfo | None,
        sim_name: str,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        self.sim_info = sim_info
        self.sim_name = sim_name
        response = sr.reset(
            self.sim_host,
            self.sim_port,
            self.db_host,
            self.db_port,
            self.senv_config.max_simulation_steps,
            self.reward_handler,
        )
        match response:
            case sr.SensorResponse(sensor1=sensor1):
                self.sim_action_response = response
                obs = self.senv_mapping.map_sensor(sensor1, self.senv_config)
                return obs, {}
            case sr.ErrorResponse(msg):
                raise RuntimeError(f"Error on reset: '{msg}'")
            case sr.FinishedResponse(msg):
                raise RuntimeError(f"Error on reset: Finished immediately '{msg}'")

    def step(self, action):
        sensor2 = self.sim_action_response.sensor2
        cnt = self.sim_action_response.cnt
        request = sr.ActionRequest(
            diffDrive1=self.senv_mapping.map_act(action, self.senv_config),
            diffDrive2=self.opponent_controller.take_step(sensor2),
            simulation_states=self.sim_action_response.simulation_states,
            cnt=cnt + 1,
        )
        should_stop = cnt > self.senv_config.max_simulation_steps
        response = sr.step(
            request,
            self.reward_handler,
            self.sim_host,
            self.sim_port,
            self.db_host,
            self.db_port,
            should_stop,
            self.senv_config.max_simulation_steps,
            self.sim_info,
        )
        match response:
            case sr.SensorResponse(sensor1=sensor1, reward1=reward):
                self.sim_action_response = response
                observation = self.senv_mapping.map_sensor(sensor1, self.senv_config)
                terminated = False
                truncated = False
                info = {}
                return observation, reward, terminated, truncated, info
            case sr.FinishedResponse(reward, message):
                observation = self.observation_space.sample()
                terminated = True
                truncated = False
                info = {"status": "OK", "message": message}
                return observation, reward, terminated, truncated, info
            case sr.ErrorResponse(message):
                observation = self.observation_space.sample()
                terminated = True
                truncated = True
                info = {"status": "ERROR", "message": message}
                return observation, 0.0, terminated, truncated, info
