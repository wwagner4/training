from typing import Any

import numpy as np
import pytest

import training.sgym as sgym
import training.simrunner as sr

_config_a = sgym.SEnvConfig(
    max_wheel_speed=100.0, max_view_distance=200.0, max_simulation_steps=1000
)

_as_a = sgym.crete_action_space(_config_a)
_os_a = sgym.create_observation_space(_config_a)

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
            "border": sgym._create_numpy_array([[20, 33, 66]], _config_a),
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
            "border": sgym._create_numpy_array([[0.0, 100.1, 150.0]], _config_a),
        },
    ),
]


@pytest.mark.parametrize("sensor, expected", sensor_to_observation_space_a_testdata)
def test_sensor_to_observation_space_a(sensor: sr.CombiSensor, expected: dict):
    result = sgym.mapping_sensor_to_observation_space(sensor, _config_a)
    assert _os_a.contains(result)
    for k in result:
        np.testing.assert_equal(result[k], expected[k], err_msg=f"Comparing {k}")


action_space_to_diff_drive_a_testdata = [
    (sgym._create_numpy_array([[0.1, 0.5]], _config_a), sr.DiffDriveValues(0.1, 0.5)),
    (sgym._create_numpy_array([[-0.1, 0.5]], _config_a), sr.DiffDriveValues(-0.1, 0.5)),
    (sgym._create_numpy_array([[0.1, -0.5]], _config_a), sr.DiffDriveValues(0.1, -0.5)),
    (sgym._create_numpy_array([[35, 45]], _config_a), sr.DiffDriveValues(35, 45)),
    (sgym._create_numpy_array([[-35, 45]], _config_a), sr.DiffDriveValues(-35, 45)),
    (sgym._create_numpy_array([[-35, -45]], _config_a), sr.DiffDriveValues(-35, -45)),
]


@pytest.mark.parametrize("space, expected", action_space_to_diff_drive_a_testdata)
def test_action_space_to_diff_drive_a(
    space: Any, expected: sr.DiffDriveValues
) -> sr.DiffDriveValues:
    result = sgym.mapping_action_space_to_diff_drive(space)
    assert _as_a.contains(space)
    assert result == expected
