from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import training.helper as helper
import training.sgym.core as sgym
import training.sgym.qlearn as sgym_qlearn
import training.sgym.sample as sgym_sample
import training.simrunner as sr
from training.simrunner import DiffDriveValues

_config_a = sgym.SEnvConfig(
    max_wheel_speed=100.0,
    wheel_speed_steps=10,
    max_view_distance=200.0,
    view_distance_steps=20,
    max_simulation_steps=1000,
)

_as_a = sgym_sample.cont_act_space(_config_a)
_os_a = sgym_sample.cont_obs_space(_config_a)

value4 = [[20, 33, 66]]
value5 = [[0.0, 100.1, 150.0]]
sensor_to_observation_space_a_testdata = [
    (
        sr.CombiSensor(
            left_distance=20,
            front_distance=33,
            right_distance=66,
            opponent_in_sector=sr.SectorName.UNDEF,
        ),
        {
            "view": 0,
            "border": np.array(value4, dtype=_config_a.dtype),
        },
    ),
    (
        sr.CombiSensor(
            left_distance=0.0,
            front_distance=100.1,
            right_distance=150.0,
            opponent_in_sector=sr.SectorName.UNDEF,
        ),
        {
            "view": 0,
            "border": np.array(value5, dtype=_config_a.dtype),
        },
    ),
]


@pytest.mark.parametrize("sensor, expected", sensor_to_observation_space_a_testdata)
def test_sensor_to_observation_space_a(sensor: sr.CombiSensor, expected: dict):
    result = sgym_sample.map_cont_sensor_to_obs(sensor, _config_a)
    assert _os_a.contains(result)
    for k in result:
        np.testing.assert_equal(result[k], expected[k], err_msg=f"Comparing {k}")


value = [[-0.1, 0.5]]
value1 = [[-35, 45]]
value2 = [[0.1, -0.5]]
value3 = [[-35, -45]]
value6 = [[35, 45]]
value7 = [[0.1, 0.5]]
action_space_to_diff_drive_a_testdata = [
    (np.array(value7, dtype=_config_a.dtype), sr.DiffDriveValues(0.1, 0.5)),
    (
        np.array(value, dtype=_config_a.dtype),
        sr.DiffDriveValues(-0.1, 0.5),
    ),
    (
        np.array(value2, dtype=_config_a.dtype),
        sr.DiffDriveValues(0.1, -0.5),
    ),
    (np.array(value6, dtype=_config_a.dtype), sr.DiffDriveValues(35, 45)),
    (np.array(value1, dtype=_config_a.dtype), sr.DiffDriveValues(-35, 45)),
    (np.array(value3, dtype=_config_a.dtype), sr.DiffDriveValues(-35, -45)),
]


@pytest.mark.parametrize("space, expected", action_space_to_diff_drive_a_testdata)
def test_action_space_to_diff_drive_a(space: Any, expected: sr.DiffDriveValues):
    result = sgym_sample.map_cont_act_to_diff_drive(space, sgym.default_senv_config)
    assert _as_a.contains(space)
    assert result == expected


continuous_to_discrete_testdata = [
    (-10.0, 10.0, 3, (-100.0001, 0)),
    (-10.0, 10.0, 3, (-10.0001, 0)),
    (-10.0, 10.0, 3, (-10.0, 0)),
    (-10.0, 10.0, 3, (-3.334, 0)),
    (-10.0, 10.0, 3, (-3.332, 1)),
    (-10.0, 10.0, 3, (0.0, 1)),
    (-10.0, 10.0, 3, (3.332, 1)),
    (-10.0, 10.0, 3, (3.334, 2)),
    (-10.0, 10.0, 3, (4, 2)),
    (-10.0, 10.0, 3, (10.0, 2)),
    (-10.0, 10.0, 3, (10.0001, 2)),
    (-10.0, 10.0, 3, (100.0001, 2)),
    (0.0, 700.0, 10, (631.8, 9)),
    (0.0, 700.0, 10, (44.73, 0)),
    (0.0, 700.0, 10, (88.73, 1)),
    (0.0, 700.0, 10, (200.6, 2)),
    (0.0, 700.0, 10, (-13.3, 0)),
    (600.0, 700.0, 10, (555.3, 0)),
    (600.0, 700.0, 10, (665.3, 6)),
    (600.0, 700.0, 10, (645.3, 4)),
]


@pytest.mark.parametrize(
    "min_value, max_value, n, expected", continuous_to_discrete_testdata
)
def test_continuous_to_discrete(
    min_value: float, max_value: float, n: int, expected: tuple
):
    v, i = expected
    result = helper.cont_to_discrete(v, min_value, max_value, n)
    print(f"### i:{i} r:{result}")
    assert i == result


cont_values_testdata = [
    (-2.0, 2.0, 5, [-2.0, -1.0, 0.0, 1.0, 2.0]),
    (0.0, 2.0, 5, [0.0, 0.5, 1.0, 1.5, 2.0]),
    (-2.0, 2.0, 9, [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]),
]


@pytest.mark.parametrize("min_value, max_value,n, expected", cont_values_testdata)
def test_cont_values(min_value: float, max_value: float, n: int, expected: list[float]):
    result = helper.cont_values(min_value, max_value, n)
    assert result == expected


# Define some index velo mappers
velo_from_index_a = sgym_qlearn._curry_velo_from_index(max_velo=3.0, velo_steps=2)
velo_from_index_b = sgym_qlearn._curry_velo_from_index(max_velo=2.0, velo_steps=4)
velo_from_index_c = sgym_qlearn._curry_velo_from_index(max_velo=2.0, velo_steps=3)

index_to_velos_testdata = [
    (0, velo_from_index_a, sr.DiffDriveValues(-3, -3)),
    (2, velo_from_index_a, sr.DiffDriveValues(-3, 3)),
    (5, velo_from_index_a, sr.DiffDriveValues(0, 3)),
    (7, velo_from_index_a, sr.DiffDriveValues(3, 0)),
    (8, velo_from_index_a, sr.DiffDriveValues(3, 3)),
    (0, velo_from_index_b, sr.DiffDriveValues(-2, -2)),
    (10, velo_from_index_b, sr.DiffDriveValues(0, -2)),
    (18, velo_from_index_b, sr.DiffDriveValues(1, 1)),
    (0, velo_from_index_c, sr.DiffDriveValues(-2, -2)),
    (1, velo_from_index_c, sr.DiffDriveValues(-2, -0.666666)),
    (3, velo_from_index_c, sr.DiffDriveValues(-2, 2)),
    (4, velo_from_index_c, sr.DiffDriveValues(-0.66666, -2)),
    (5, velo_from_index_c, sr.DiffDriveValues(-0.66666, -0.666666)),
    (6, velo_from_index_c, sr.DiffDriveValues(-0.66666, 0.666666)),
    (7, velo_from_index_c, sr.DiffDriveValues(-0.66666, 2)),
]


@pytest.mark.parametrize("index, f, expected", index_to_velos_testdata)
def test_velo_from_index(
    index: int, f: Callable[[int], sr.DiffDriveValues], expected: sr.DiffDriveValues
):
    result = f(index)
    assert f1(result) == f1(expected)


def f1(v: DiffDriveValues) -> str:
    return f"r:{v.right_velo:.3f} l:{v.left_velo:.3f}"
