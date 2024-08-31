import importlib
import traceback
from dataclasses import dataclass
from enum import Enum

from dataclasses_json import dataclass_json

import training.simdb as db
import training.udp as udp
import training.util as util


class SendCommand:
    pass


class ReceiveCommand:
    pass


class ControllerName(str, Enum):
    FAST_CIRCLE = ("fast-circle",)
    SLOW_CIRCLE = ("slow-circle",)
    STAY_IN_FIELD = ("stay-in-field",)
    TUMBLR = ("tumblr",)
    BLIND_TUMBLR = ("blind-tumblr",)
    TEST_TURN = ("test-turn",)


class SectorName(Enum):
    UNDEF = "undef"
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


@dataclass
class CombiSensor:
    left_distance: float
    front_distance: float
    right_distance: float
    opponent_in_sector: SectorName


@dataclass
class DiffDriveValues:
    right_velo: float
    left_velo: float


@dataclass_json
@dataclass
class PosDir:
    xpos: float
    ypos: float
    direction: float


@dataclass
class SensorDto:
    pos_dir: PosDir
    combi_sensor: CombiSensor


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
    stepsCount: int


@dataclass
class FinishedOkCommand(ReceiveCommand):
    robot1_rewards: list[(str, str)]
    robot2_rewards: list[(str, str)]


@dataclass_json
@dataclass
class SimulationState:
    robot1: PosDir
    robot2: PosDir


@dataclass
class FinishedErrorCommand(ReceiveCommand):
    message: str


# noinspection PyUnresolvedReferences
def start(
    port: int,
    sim_name: str,
    controller_name1: ControllerName,
    controller_name2: ControllerName,
):
    controller1 = ControllerProvider.get(controller_name1)
    controller2 = ControllerProvider.get(controller_name2)

    simulation_states = []

    with db.create_client() as client:

        def check_running():
            running_sim = db.find_running(client, "running", port)
            print(f"--- Found running for {port} {running_sim}")
            if running_sim:
                raise RuntimeError(f"Baseport {port} is currently running")

        def insert_new_sim() -> str:
            sim_robot1 = db.SimulationRobot(
                name=controller1.name(),
                description=controller1.description(),
            )
            sim_robot2 = db.SimulationRobot(
                name=controller2.name(),
                description=controller2.description(),
            )
            sim = db.Simulation(
                port=port,
                name=sim_name,
                robot1=sim_robot1,
                robot2=sim_robot2,
            )
            _obj_id = db.insert(client, sim.to_dict())
            print(f"--- Wrote to database id:{_obj_id} sim:{sim}")
            return _obj_id

        def send_command_and_wait(cmd: SendCommand) -> ReceiveCommand:
            send_str = format_command(cmd)
            # print(f"---> Sending {cmd} - {send_str}")
            resp_str = udp.send_and_wait(send_str, port, 10)
            resp = parse_command(resp_str)
            # print(f"<--- Result {resp} {resp_str}")
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
                    case SensorCommand(s1, s2):
                        # print("sensors", s1, s2)
                        state = SimulationState(s1.pos_dir, s2.pos_dir)
                        simulation_states.append(state)

                        r1 = controller1.take_step(s1.combi_sensor)
                        r2 = controller2.take_step(s2.combi_sensor)
                        # print(
                        #   "## sensor",
                        #   s1.combi_sensor.front_distance,
                        #   s2.combi_sensor.front_distance)

                        command = DiffDriveCommand(r1, r2, cnt)
                    case FinishedOkCommand(r1, r2):
                        events_dict = {"r1": r1, "r2": r2}
                        dicts = [s.to_dict() for s in simulation_states]
                        db.update_status_finished(client, obj_id, events_dict, dicts)
                        print(
                            f"Finished with OK: {obj_id} {events_dict}"
                            f"{simulation_states[:5]}..."
                        )
                        break
                    case FinishedErrorCommand(msg):
                        db.update_status_error(client, obj_id, msg)
                        print(f"Finished with ERROR: {obj_id} {msg}")
                        break

        except BaseException as ex:
            msg = util.message(ex)
            print(traceback.format_exc())
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
        case DiffDriveCommand(r1, r2, cnt):
            return (
                f"C|{format_diff_drive_values(r1)}#{format_diff_drive_values(r2)}#{cnt}"
            )
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

    def parse_finished(finished_data: str) -> list:
        if finished_data:
            ds = finished_data.split(";")
            return [(d.split("!")[0], d.split("!")[1]) for d in ds]
        else:
            return []

    (head, body) = data.split("|")
    match head:
        case "E":
            return FinishedErrorCommand(body)
        case "B":
            (r1, r2) = body.split("#")
            return SensorCommand(parse_sensor_dto(r1), parse_sensor_dto(r2))
        case "D":
            (r1, r2) = body.split("#")
            return FinishedOkCommand(parse_finished(r1), parse_finished(r2))
        case _:
            raise NotImplementedError(f"parse_command {data}")


class Controller:
    def take_step(self, sensor: CombiSensor) -> DiffDriveValues:
        pass

    def name(self) -> str:
        pass

    def description(self) -> dict:
        pass


class ControllerProvider:
    @staticmethod
    def get(name: ControllerName) -> Controller:
        match name:
            case ControllerName.FAST_CIRCLE:
                module = importlib.import_module(
                    "training.controller.circle_controller"
                )
                class_ = module.FastCircleController
                return class_()
            case ControllerName.SLOW_CIRCLE:
                module = importlib.import_module(
                    "training.controller.circle_controller"
                )
                class_ = module.SlowCircleController
                return class_()
            case ControllerName.TUMBLR:
                module = importlib.import_module(
                    "training.controller.tumblr_controller"
                )
                class_ = module.TumblrController
                return class_()
            case ControllerName.BLIND_TUMBLR:
                module = importlib.import_module(
                    "training.controller.blind_tumblr_controller"
                )
                class_ = module.BlindTumblrController
                return class_()
            case ControllerName.STAY_IN_FIELD:
                module = importlib.import_module(
                    "training.controller.stay_in_field_controller"
                )
                class_ = module.StayInFieldController
                return class_()
            case ControllerName.TEST_TURN:
                module = importlib.import_module(
                    "training.controller.test_turn_controller"
                )
                class_ = module.TestTurnController
                return class_()
            case _:
                raise RuntimeError(f"Unknown controller {name}")
