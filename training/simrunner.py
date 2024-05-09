from dataclasses import dataclass
from enum import Enum

import training.simdb as db
import training.udp as udp
import training.util as util


class SectorName(Enum):
    UNDEF = "undef"
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


class SendCommand:
    pass


class ReceiveCommand:
    pass


@dataclass
class PosDir:
    xpos: float
    ypos: float
    direction: float


@dataclass
class CombiSensor:
    left_distance: float
    front_distance: float
    right_distance: float
    opponent_in_sector: SectorName


@dataclass
class SensorDto:
    pos_dir: PosDir
    combi_sensor: CombiSensor


@dataclass
class DiffDriveValues:
    right_velo: float
    left_velo: float


@dataclass
class StartCommand(SendCommand):
    pass


@dataclass
class SensorCommand(ReceiveCommand):
    robot1_sensor: SensorDto
    robot2_sensor: SensorDto


@dataclass
class DiffDriveCommand(SendCommand):
    robot1_diff_drive_values: DiffDriveValues
    robot2_diff_drive_values: DiffDriveValues


@dataclass
class FinishedOkCommand(ReceiveCommand):
    robot1_rewards: list[(str, str)]
    robot2_rewards: list[(str, str)]


@dataclass
class FinishedErrorCommand(ReceiveCommand):
    message: str


def start(port: int):
    with db.create_client() as client:

        def check_running():
            running_sim = db.find_running(client, "running", port)
            print(f"--- Found running for {port} {running_sim}")
            if running_sim:
                raise RuntimeError(f"Baseport {port} is currently running")

        def insert_new_sim() -> str:
            sim = db.Simulation(
                port=port,
            )
            _obj_id = db.insert(client, sim.to_dict())
            print(f"--- Wrote to database id:{_obj_id} sim:{sim}")
            return _obj_id

        def send_command_and_wait(command: SendCommand) -> ReceiveCommand:
            send_str = format_command(command)
            print(f"---> Sending {command} - {send_str}")
            resp_str = udp.send_and_wait(send_str, port, 10)
            resp = parse_command(resp_str)
            print(f"<--- Result {resp} {resp_str}")
            return resp

        check_running()
        obj_id = insert_new_sim()
        try:
            command = StartCommand()
            cnt = 0
            while True:
                cnt += 1
                print("")
                response: ReceiveCommand = send_command_and_wait(command)
                match response:
                    case SensorCommand(r1, r2):
                        print("sensors", r1, r2)
                        r1 = DiffDriveValues(0.5, 0.4)
                        r2 = DiffDriveValues(0.3, 0.4)
                        command = DiffDriveCommand(r1, r2)
                    case FinishedOkCommand(r1, r2):
                        events_dict = {"r1": r1, "r2": r2}
                        db.update_status_finished(client, obj_id, events_dict)
                        print(f"Finished with OK: {obj_id} {events_dict}")
                        break
                    case FinishedErrorCommand(msg):
                        db.update_status_error(client, obj_id, msg)
                        print(f"Finished with ERROR: {obj_id} {msg}")
                        break

        except BaseException as ex:
            msg = util.message(ex)
            print(f"ERROR: {msg}")
            db.update_status_error(client, obj_id, msg)


def format_command(cmd: SendCommand) -> str:
    def format_float(value: float) -> str:
        return f"{value:.4f}"

    def format_diff_drive_values(values: DiffDriveValues) -> str:
        return f"{format_float(values.left_velo)};{format_float(values.right_velo)}"

    match cmd:
        case StartCommand():
            return "A|"
        case DiffDriveCommand(r1, r2):
            return f"C|{format_diff_drive_values(r1)}#{format_diff_drive_values(r2)}"
        case _:
            raise NotImplementedError(f"format_command {cmd}")


def parse_command(data: str) -> ReceiveCommand:
    def parse_sensor_dto(sensor_data: str) -> SensorDto:
        ds = sensor_data.split(";")
        return SensorDto(
            pos_dir=PosDir(float(ds[0]), float(ds[1]), float(ds[2])),
            combi_sensor=CombiSensor(
                left_distance=float(ds[3]),
                front_distance=float(ds[4]),
                right_distance=float(ds[5]),
                opponent_in_sector=SectorName[ds[6]],
            ),
        )

    def parse_finished(data: str) -> SensorDto:
        print(f"--- data '{data}'")
        if data:
            ds = data.split(";")
            return [(d.split("!")[0], d.split("!")[1]) for d in ds]
        else:
            return []

    (h, d) = data.split("|")
    match h:
        case "E":
            return FinishedErrorCommand(d)
        case "B":
            (r1, r2) = d.split("#")
            return SensorCommand(parse_sensor_dto(r1), parse_sensor_dto(r2))
        case "D":
            (r1, r2) = d.split("#")
            return FinishedOkCommand(parse_finished(r1), parse_finished(r2))
        case _:
            raise NotImplementedError(f"parse_command {data}")
