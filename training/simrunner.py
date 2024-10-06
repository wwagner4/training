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
class CombiSensorDto:
    pos_dir: PosDir
    combi_sensor: CombiSensor


@dataclass
class StartCommand(SendCommand):
    pass


@dataclass
class CombiSensorCommand(ReceiveCommand):
    robot1_sensor: CombiSensorDto
    robot2_sensor: CombiSensorDto


@dataclass
class DiffDriveCommand(SendCommand):
    robot1_diff_drive_values: DiffDriveValues
    robot2_diff_drive_values: DiffDriveValues
    stepsCount: int


@dataclass
class FinishedOkCommand(ReceiveCommand):
    robot1_finish_properties: list[(str, str)]
    robot2_finish_properties: list[(str, str)]


@dataclass_json
@dataclass
class SimulationState:
    robot1: PosDir
    robot2: PosDir


@dataclass
class FinishedErrorCommand(ReceiveCommand):
    message: str


class Controller:
    def take_step(self, sensor: CombiSensor) -> DiffDriveValues:
        pass

    def name(self) -> str:
        pass

    def description(self) -> dict:
        pass


@dataclass(frozen=True)
class Response:
    def is_finished(self) -> bool:
        pass


@dataclass(frozen=True)
class ActionResponse(Response):
    simulation_states: list[SimulationState]
    sensor1: CombiSensor
    sensor2: CombiSensor
    obj_id: str | None
    cnt: int

    def is_finished(self) -> bool:
        return False


@dataclass(frozen=True)
class ErrorResponse(Response):
    message: str

    def is_finished(self) -> bool:
        return True


@dataclass(frozen=True)
class FinishedResponse(Response):
    message: str

    def is_finished(self) -> bool:
        return True


@dataclass(frozen=True)
class ObservationRequest:
    diffDrive1: DiffDriveValues
    diffDrive2: DiffDriveValues
    simulation_states: list[SimulationState]
    obj_id: str
    cnt: int


def reset(
    port: int,
    sim_name: str,
    name1: str,
    desc1: dict,
    name2: str,
    desc2: dict,
    record: bool,
) -> Response:
    obj_id = _insert_new_sim(name1, desc1, name2, desc2, port, sim_name, record)
    return _step(StartCommand(), [], port, obj_id, 0)


def start(
    port: int,
    sim_name: str,
    controller_name1: ControllerName,
    controller_name2: ControllerName,
    record: bool,
):
    controller1 = ControllerProvider.get(controller_name1)
    controller2 = ControllerProvider.get(controller_name2)

    def create_request(response: Response) -> ObservationRequest:
        # print(f"### create_request {response}")
        match response:
            case ActionResponse(
                simulation_states=simulation_states,
                sensor1=sensor1,
                sensor2=sensor2,
                obj_id=obj_id,
                cnt=cnt,
            ):
                diff_drive1 = controller1.take_step(sensor1)
                diff_drive2 = controller2.take_step(sensor2)
                return ObservationRequest(
                    diffDrive1=diff_drive1,
                    diffDrive2=diff_drive2,
                    cnt=cnt + 1,
                    obj_id=obj_id,
                    simulation_states=simulation_states,
                )
            case _:
                raise ValueError(f"Unknown response {response}")

    response: Response = reset(
        port,
        sim_name,
        controller1.name(),
        controller1.description(),
        controller2.name(),
        controller2.description(),
        record,
    )
    while True:
        if response.is_finished():
            return
        request: ObservationRequest = create_request(response)
        response = step(request, port)


def step(request: ObservationRequest, port: int) -> Response:
    """

    :rtype: object
    """
    cmd = DiffDriveCommand(
        robot1_diff_drive_values=request.diffDrive1,
        robot2_diff_drive_values=request.diffDrive2,
        stepsCount=request.cnt,
    )
    return _step(
        command=cmd,
        simulation_states=request.simulation_states,
        obj_id=request.obj_id,
        port=port,
        cnt=request.cnt,
    )


def _step(
    command: SendCommand,
    simulation_states: list[SimulationState],
    port: int,
    obj_id: str | None,
    cnt: int,
) -> Response:
    try:
        response: ReceiveCommand = _send_command_and_wait(command, port)
        match response:
            case CombiSensorCommand(s1, s2):
                # print("sensors", s1, s2)
                state = SimulationState(s1.pos_dir, s2.pos_dir)
                simulation_states.append(state)

                return ActionResponse(
                    simulation_states=simulation_states,
                    sensor1=s1.combi_sensor,
                    sensor2=s2.combi_sensor,
                    obj_id=obj_id,
                    cnt=cnt,
                )
            case FinishedOkCommand(r1, r2):
                events_dict = {"r1": r1, "r2": r2}
                dicts = [s.to_dict() for s in simulation_states]
                if obj_id:
                    with db.create_client() as client:
                        db.update_status_finished(client, obj_id, events_dict, dicts)
                # print(
                #     f"Finished with OK: steps: {cnt} db_id: {obj_id} {events_dict}"
                #     f"{simulation_states[:5]}..."
                # )
                msg = f"Finished with OK: steps: {cnt} db_id: {obj_id}"
                return FinishedResponse(
                    message=msg,
                )
            case FinishedErrorCommand(msg):
                if obj_id:
                    with db.create_client() as client:
                        db.update_status_error(client, obj_id, msg)
                msg = f"Finished with ERROR: steps: {cnt} db_id: {obj_id} {msg}"
                # print(msg)
                return ErrorResponse(
                    message=msg,
                )
    except BaseException as ex:
        return _handle_exception(ex, obj_id)


def _handle_exception(ex: BaseException, obj_id: str | None) -> Response:
    msg = util.message(ex)
    print(traceback.format_exc())
    print(f"ERROR: {msg}")
    if obj_id:
        with db.create_client() as client:
            db.update_status_error(client, obj_id, msg)
    return ErrorResponse(msg)


def _insert_new_sim(
    name1: str,
    desc1: dict,
    name2: str,
    desc2: dict,
    port: int,
    sim_name: str,
    record: bool,
) -> str:
    _obj_id = None
    if record:
        sim_robot1 = db.SimulationRobot(
            name=name1,
            description=desc1,
        )
        sim_robot2 = db.SimulationRobot(
            name=name2,
            description=desc2,
        )
        sim = db.Simulation(
            port=port,
            name=sim_name,
            robot1=sim_robot1,
            robot2=sim_robot2,
        )
        # print("--- Writing to database")
        # pprint.pprint(sim)
        with db.create_client() as client:
            _obj_id = db.insert(client, sim.to_dict())
        # print(f"--- Wrote to database id:{_obj_id} sim:{sim}")
    return _obj_id


def _send_command_and_wait(cmd: SendCommand, port: int) -> ReceiveCommand:
    send_str = _format_command(cmd)
    # print(f"---> Sending {cmd} - {send_str}")
    resp_str = udp.send_and_wait(send_str, port, 10)
    resp = _parse_command(resp_str)
    # print(f"<--- Result {resp} {resp_str}")
    return resp


def _format_command(cmd: SendCommand) -> str:
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


def _parse_command(data: str) -> ReceiveCommand:
    def parse_sensor_dto(sensor_data: str) -> CombiSensorDto:
        ds = sensor_data.split(";")
        return CombiSensorDto(
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
            return CombiSensorCommand(parse_sensor_dto(r1), parse_sensor_dto(r2))
        case "D":
            (r1, r2) = body.split("#")
            return FinishedOkCommand(parse_finished(r1), parse_finished(r2))
        case _:
            raise NotImplementedError(f"parse_command {data}")


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
