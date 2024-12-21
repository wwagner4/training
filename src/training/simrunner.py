import datetime as dt
import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from dataclasses_json import dataclass_json

import training.simdb as db
import training.udp as udp


@dataclass_json
@dataclass
class SimInfo:
    name1: str
    desc1: dict
    name2: str
    desc2: dict
    port: int
    sim_name: str
    max_simulation_steps: int


@dataclass_json
@dataclass
class PosDir:
    xpos: float
    ypos: float
    direction: float


class SendCommand:
    pass


class ReceiveCommand:
    pass


class ControllerName(str, Enum):
    STAND_STILL = ("stand-still",)
    STAY_IN_FIELD = ("stay-in-field",)
    TUMBLR = ("tumblr",)
    BLIND_TUMBLR = ("blind-tumblr",)
    TEST_TURN = ("test-turn",)
    SGYM_SAMPLE = ("sgym-sample",)


class RewardHandlerName(str, Enum):
    END_CONSIDER_ALL = "end-consider-all"
    CONTINUOUS_CONSIDER_ALL = "continuous-consider-all"


class SectorName(Enum):
    UNDEF = "undef"
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


@dataclass_json
@dataclass
class SimulationState:
    robot1: PosDir
    robot2: PosDir


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
    stop: bool


@dataclass
class FinishedOkCommand(ReceiveCommand):
    robot1_finish_properties: list[(str, str)]
    robot2_finish_properties: list[(str, str)]


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


class Response:
    pass


@dataclass(frozen=True)
class SensorResponse(Response):
    simulation_states: list[SimulationState]
    sensor1: CombiSensor
    sensor2: CombiSensor
    reward1: float
    reward2: float
    cnt: int


@dataclass(frozen=True)
class ErrorResponse(Response):
    message: str


@dataclass(frozen=True)
class FinishedResponse(Response):
    reward1: float
    reward2: float
    message: str


@dataclass(frozen=True)
class ActionRequest:
    diffDrive1: DiffDriveValues
    diffDrive2: DiffDriveValues
    simulation_states: list[SimulationState]
    cnt: int


class RewardHandler(ABC):
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def calculate_reward(self, state: SimulationState) -> (float, float):
        pass

    @abstractmethod
    def calculate_end_reward(
        self,
        states: list[SimulationState],
        properties1: list[list],
        properties2: list[list],
        max_simulation_count: int,
    ) -> (float, float):
        pass


def reset(
    port: int, max_simulation_steps: int, reward_handler: RewardHandler
) -> Response:
    return _step(
        StartCommand(),
        reward_handler,
        [],
        port,
        0,
        max_simulation_steps,
        None,
    )


def step(
    request: ActionRequest,
    reward_handler: RewardHandler,
    port: int,
    stop: bool,
    max_simulation_steps: int,
    sim_info: SimInfo | None,
) -> Response:
    cmd = DiffDriveCommand(
        robot1_diff_drive_values=request.diffDrive1,
        robot2_diff_drive_values=request.diffDrive2,
        stepsCount=request.cnt,
        stop=stop,
    )
    return _step(
        command=cmd,
        reward_handler=reward_handler,
        simulation_states=request.simulation_states,
        port=port,
        cnt=request.cnt,
        max_simulation_steps=max_simulation_steps,
        sim_info=sim_info,
    )


def calculate_reward(reward_handler: RewardHandler, state: SimulationState) -> float:
    return reward_handler.calculate_reward(state)


def calculate_end_reward(
    reward_handler: RewardHandler,
    states: list[SimulationState],
    properties1: list[list],
    properties2: list[list],
    max_simulation_steps: int,
) -> (float, float):
    return reward_handler.calculate_end_reward(
        states, properties1, properties2, max_simulation_steps
    )


def _step(
    command: SendCommand,
    reward_handler: RewardHandler,
    simulation_states: list[SimulationState],
    port: int,
    cnt: int,
    max_simulation_steps: int,
    sim_info: SimInfo | None,
) -> Response:
    response: ReceiveCommand = _send_command_and_wait(command, port)
    match response:
        case CombiSensorCommand(s1, s2):
            state = SimulationState(s1.pos_dir, s2.pos_dir)
            simulation_states.append(state)
            reward1, reward2 = calculate_reward(reward_handler, state)
            return SensorResponse(
                simulation_states=simulation_states,
                sensor1=s1.combi_sensor,
                sensor2=s2.combi_sensor,
                reward1=reward1,
                reward2=reward2,
                cnt=cnt,
            )
        case FinishedOkCommand(properties1, properties2):
            reward1, reward2 = calculate_end_reward(
                reward_handler,
                simulation_states,
                properties1,
                properties2,
                max_simulation_steps,
            )
            obj_id = None
            if sim_info is not None:
                with db.create_client() as client:
                    obj_id = _db_insert_new_sim(
                        client=client,
                        sim_info=sim_info,
                        properties1=properties1,
                        properties2=properties2,
                        simulation_states=simulation_states,
                        reward_handler=reward_handler.name(),
                        reward1=reward1,
                        reward2=reward2,
                        step_count=cnt,
                    )
            msg = f"steps: {cnt} obj_id: {obj_id}"
            return FinishedResponse(
                reward1=reward1,
                reward2=reward2,
                message=msg,
            )
        case FinishedErrorCommand(msg):
            message = f"Finished with ERROR: steps:{cnt} msg:{msg}"
            return ErrorResponse(
                message=message,
            )
        case _:
            raise RuntimeError(f"Unknown response '{response}'")


def _db_insert_new_sim(
    client: any,
    sim_info: SimInfo,
    properties1: list[list[str]],
    properties2: list[list[str]],
    simulation_states: list[SimulationState],
    reward_handler: str,
    reward1: float,
    reward2: float,
    step_count: int,
) -> str:
    events_object = {
        "r1": properties1,
        "r2": properties2,
    }
    state_objects = [s.to_dict() for s in simulation_states]
    robot1_object = {
        "name": sim_info.name1,
        "description": sim_info.desc1,
    }
    robot2_object = {
        "name": sim_info.name2,
        "description": sim_info.desc2,
    }
    step_count_relative = float(step_count) / sim_info.max_simulation_steps
    sim = {
        "started_at": dt.datetime.now(),
        "status": "finished",
        "port": sim_info.port,
        "name": sim_info.sim_name,
        "robot1": robot1_object,
        "robot2": robot2_object,
        "events": events_object,
        "states": state_objects,
        "rewardhandler": reward_handler,
        "reward1": reward1,
        "reward2": reward2,
        "stepcount": step_count_relative,
    }
    return db.insert(client, sim)


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

    def format_bool(value: bool) -> str:
        return "1" if value else "0"

    match cmd:
        case StartCommand():
            return "A|"
        case DiffDriveCommand(r1, r2, cnt, stop):
            v1 = format_diff_drive_values(r1)
            v2 = format_diff_drive_values(r2)
            s = format_bool(stop)
            return f"C|{v1}#{v2}#{cnt}#{s}"
        case _:
            raise NotImplementedError(f"format_command {cmd}")


def _parse_command(data: str) -> ReceiveCommand:
    def parse_sensor_dto(sensor_data: str) -> CombiSensorDto:
        ds = sensor_data.split(";")
        # noinspection PyTypeChecker
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
            case ControllerName.STAND_STILL:
                module = importlib.import_module(
                    "training.controller.stand_still_controller"
                )
                class_ = module.StandStillController
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
            case ControllerName.SGYM_SAMPLE:
                module = importlib.import_module(
                    "training.controller.sgym_sample_controller"
                )
                class_ = module.SGymSampleController
                return class_()
            case _:
                raise RuntimeError(f"Unknown controller {name}")


class RewardHandlerProvider:
    @staticmethod
    def get(name: RewardHandlerName) -> RewardHandler:
        match name:
            case RewardHandlerName.END_CONSIDER_ALL:
                module = importlib.import_module("training.reward")
                class_ = module.EndConsiderAllRewardHandler
                return class_()
        match name:
            case RewardHandlerName.CONTINUOUS_CONSIDER_ALL:
                module = importlib.import_module("training.reward")
                class_ = module.ConsiderAllRewardHandler
                return class_()
            case _:
                raise RuntimeError(f"Unknown reward handler {name}")
