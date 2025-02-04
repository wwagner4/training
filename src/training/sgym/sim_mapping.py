from collections.abc import Callable

import gymnasium as gym
import gymnasium.spaces as gyms

import training.helper as hlp
import training.sgym.core as sgym
import training.simrunner as sr


def get_q_act_space(config: sgym.SEnvConfig) -> gym.Space:
    n = (config.wheel_speed_steps + 1) * (config.wheel_speed_steps + 1)
    return gyms.Discrete(n)


def get_q_obs_space(config: sgym.SEnvConfig) -> gym.Space:
    return gyms.Tuple(
        (
            gyms.Discrete(n=4),
            gyms.Discrete(n=config.view_distance_steps),
            gyms.Discrete(n=config.view_distance_steps),
            gyms.Discrete(n=config.view_distance_steps),
        )
    )


def q_sgym_mapping(cfg: sgym.SEnvConfig) -> sgym.SEnvMapping:
    return sgym.SEnvMapping(
        act_space=get_q_act_space,
        obs_space=get_q_obs_space,
        map_act=curry_q_act_to_diff_drive(cfg),
        map_sensor=map_q_sensor_to_obs,
    )


def curry_q_act_to_diff_drive(
    config: sgym.SEnvConfig,
) -> Callable[[any, sgym.SEnvConfig], sr.DiffDriveValues]:
    velo_from_index = _curry_velo_from_index(
        config.max_wheel_speed, config.wheel_speed_steps
    )

    def inner(a_space: int, _config: sgym.SEnvConfig) -> sr.DiffDriveValues:
        return velo_from_index(a_space)

    return inner


def map_q_sensor_to_obs(
    sensor: sr.CombiSensor, config: sgym.SEnvConfig
) -> tuple[int, int, int, int]:
    def discrete(distance: float) -> int:
        return hlp.cont_to_discrete(
            distance, 0.0, config.max_view_distance, config.view_distance_steps
        )

    return (
        sr.sector_mapping(sensor.opponent_in_sector),
        discrete(sensor.left_distance),
        discrete(sensor.front_distance),
        discrete(sensor.right_distance),
    )


def _curry_velo_from_index(
    max_velo: float, velo_steps: int
) -> Callable[[int], sr.DiffDriveValues]:
    velos = hlp.cont_values(-max_velo, max_velo, velo_steps + 1)
    n = len(velos)
    diff_drives = []
    for i in range(n):
        for j in range(n):
            diff_drives.append(sr.DiffDriveValues(velos[i], velos[j]))

    def inner(index: int) -> sr.DiffDriveValues:
        return diff_drives[index]

    return inner
