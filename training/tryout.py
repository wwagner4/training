import socket
from pathlib import Path

import training.reward_util as rwu


def write1():
    time = "2024-09-14 13:17:06"
    nr = "01"
    rwu.util_sims_to_json_files(time, nr)


def create_simruns():
    bd = Path.home() / "tmp"
    od = bd / "ssout02"
    host = socket.gethostname()[0:4]
    rwu.util_create_simruns(n=5, m=4, host=host, id="rw", base_dir=bd, out_dir=od)


def visualize_from_files():
    rwu.visualize_from_files(
        "8f72cf",
        Path("/home/wwagner4/prj/SUMOSIM/training/tests/data"),
        5,
        4,
        Path.home() / "tmp" / "ssout01",
    )


def print_distances():
    filename = "wall-rw-7506e4-07_stay-in-field_stay-in-field.json"
    rwu.print_distances(
        Path("/home/wwagner4/prj/SUMOSIM/training/tests/data/") / filename
    )


def print_test_templates():
    id = "6b1d"
    dir = Path(__file__).parent.parent / "tests" / "data"
    print(dir)
    files = dir.iterdir()
    sorted_files = sorted(files, key=lambda f: f.name)
    for file in sorted_files:
        name = file.name
        if id in name and name.endswith("json"):
            print(f"""    (
        "{name}",
        rw.SimEvents(
            rw.RobotEvents(
                result=RobotEventsResult.DRAW,
                push_collision_count=0,
                is_pushed_collision_count=0,
                end=RobotEventsEnd.NONE,
            ),
            rw.RobotEvents(
                result=RobotEventsResult.DRAW,
                push_collision_count=0,
                is_pushed_collision_count=0,
                end=RobotEventsEnd.NONE,
            ),
        ),
    ),""")


def main():
    # create_simruns()
    # visualize_from_files()
    print_test_templates()
