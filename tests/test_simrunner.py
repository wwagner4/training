import training.controller as ctr
import training.simrunner as sr


def test_format_command_start():
    data = sr.format_command(sr.StartCommand())
    assert data == "A|"


def test_format_diff_drive():
    r1 = ctr.DiffDriveValues(1.5, 4.3)
    r2 = ctr.DiffDriveValues(-1.2, 44.0)
    data = sr.format_command(sr.DiffDriveCommand(r1, r2, 123))
    assert data == "C|4.3000;1.5000#44.0000;-1.2000#123"


def test_parse_sensor():
    expected = sr.SensorCommand(
        robot1_sensor=sr.SensorDto(
            pos_dir=sr.PosDir(-80.0, 0.0, 0.1396),
            combi_sensor=ctr.CombiSensor(
                left_distance=357.6570,
                front_distance=506.4382,
                right_distance=398.3340,
                opponent_in_sector=ctr.SectorName.RIGHT,
            ),
        ),
        robot2_sensor=sr.SensorDto(
            pos_dir=sr.PosDir(-80.0, 0.0, 0.0),
            combi_sensor=ctr.CombiSensor(
                left_distance=385.7849,
                front_distance=462.9790,
                right_distance=401.1228,
                opponent_in_sector=ctr.SectorName.CENTER,
            ),
        ),
    )
    data = (
        "B|-80.0000;0.0000;0.1396;357.6570;506.4382;398.3340;RIGHT#"
        "-80.0000;0.0000;0.0000;385.7849;462.9790;401.1228;CENTER"
    )
    cmd = sr.parse_command(data)
    assert cmd == expected


def test_parse_finished_ok():
    data = "D|#winner!true"
    cmd = sr.parse_command(data)
    assert cmd == sr.FinishedOkCommand([], [("winner", "true")])
