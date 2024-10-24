import itertools as it
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from subprocess import run

import matplotlib.pyplot as plt
import pandas as pd

import training.consts
import training.simrunner as sr
from training.simrunner import SimInfo
from training.sumosim_helper import row_col


class CombinationType(str, Enum):
    WITH_REPLACEMENT = "with-replacement"
    WITHOUT_REPLACEMENT = "without-replacement"


@dataclass
class Result:
    combinations: list[(str, str)]
    data: list[(str, str, float, float)]


def start(
    port: int,
    name: str,
    controller_names: list[sr.ControllerName],
    reward_handler_name: sr.RewardHandlerName,
    combination_type: CombinationType.WITHOUT_REPLACEMENT,
    max_simulation_steps: int,
    epoch_count: int,
    record: bool,
):
    if max_simulation_steps > training.consts.MAX_STEPS:
        raise ValueError(
            f"The simulator runs maximum {training.consts.MAX_STEPS} steps. "
            f"Defining max-simulation-steps greater than that makes no sense"
        )

    out_dir = Path.home() / "tmp" / "sumosim"
    out_dir.mkdir(exist_ok=True, parents=True)
    for f in out_dir.iterdir():
        if name in f.name:
            raise RuntimeError(f"Name '{name}' was already used")

    combinations = _tournament_combinations(controller_names, combination_type)
    combinations_count = len(combinations)
    i = 1
    result_data = []
    for c1, c2 in combinations:
        for j in range(epoch_count):
            r1, r2, msg = run_epoch(
                port,
                name,
                max_simulation_steps,
                i,
                j,
                c1,
                c2,
                reward_handler_name,
                record,
            )
            print(
                f"Finished epoch c:{i}/{combinations_count} "
                f"e:{j + 1}/{epoch_count} "
                f"r1:{r1:10.2f} r2:{r2:10.2f} {msg}"
            )
            result_data.append((c1.value, c2.value, r1, r2))
            j += 1
        i += 1
    df = pd.DataFrame(result_data, columns=["c1", "c2", "r1", "r2"])
    write_tournament_data(out_dir, name, df)

    controller_names_str = ", ".join([c.value for c in controller_names])
    desc = {
        "name": name,
        "controller": controller_names_str,
        "reward handler": reward_handler_name.value,
        "max sim steps": max_simulation_steps,
        "epoch count": epoch_count,
    }
    line_keys = [[0, 3, 4], [1], [2]]
    _suptitle = suptitle(desc, line_keys)
    img_path = plot_rewards(out_dir, name, _suptitle, df)
    print(f"Wrote results for {name} to {out_dir}")


def run_epoch(
    port: int,
    name: str,
    max_simulation_steps: int,
    combination_number: int,
    epoch_number: int,
    controller_name1: sr.ControllerName,
    controller_name2: sr.ControllerName,
    reward_handler_name: sr.RewardHandlerName,
    record: bool,
) -> (float, float):
    sim_name = f"{name}-{combination_number:03d}-{epoch_number:04d}"
    controller1 = sr.ControllerProvider.get(controller_name1)
    controller2 = sr.ControllerProvider.get(controller_name2)
    reward_handler = sr.RewardHandlerProvider.get(reward_handler_name)

    def apply_policies(sensor_response: sr.SensorResponse) -> sr.ActionRequest:
        diff_drive1 = controller1.take_step(sensor_response.sensor1)
        diff_drive2 = controller2.take_step(sensor_response.sensor2)
        return sr.ActionRequest(
            diffDrive1=diff_drive1,
            diffDrive2=diff_drive2,
            cnt=cnt + 1,
            simulation_states=simulation_states,
        )

    sim_info = None
    if record:
        sim_info = SimInfo(
            sim_name=sim_name,
            port=port,
            name1=controller1.name(),
            desc1=controller1.description(),
            name2=controller2.name(),
            desc2=controller2.description(),
            max_simulation_steps=max_simulation_steps,
        )
    try:
        response: sr.Response = sr.reset(
            port,
            sim_name,
            max_simulation_steps,
            reward_handler,
        )
        cnt = 0
        cumulative_reward1 = 0.0
        cumulative_reward2 = 0.0
        while True:
            match response:
                case sr.FinishedResponse(reward1, reward2, msg):
                    cumulative_reward1 += reward1
                    cumulative_reward2 += reward2
                    return cumulative_reward1, cumulative_reward2, msg
                case sr.ErrorResponse(message=msg):
                    error_msg = f"ERROR running {sim_name}: {msg}"
                    print(error_msg)
                    raise RuntimeError(error_msg)
                case sr.SensorResponse(
                    reward=reward,
                    simulation_states=simulation_states,
                    cnt=cnt,
                ):
                    reward1, reward2 = reward
                    cumulative_reward1 += reward1
                    cumulative_reward2 += reward2
                    # print(f"### {epoch_name} {cnt} reward:{reward}")
                    # noinspection PyTypeChecker
                    request = apply_policies(response)
                    stop = cnt + 1 >= max_simulation_steps
                    response = sr.step(
                        request,
                        reward_handler,
                        port,
                        stop,
                        max_simulation_steps,
                        sim_info,
                    )
    except BaseException as ex:
        print(f"### Error running {sim_name} {ex}")
        raise ex


def _tournament_combinations(
    names: list, combination_type: CombinationType
) -> list[(any, any)]:
    match combination_type:
        case CombinationType.WITH_REPLACEMENT:
            return list(it.combinations_with_replacement(names, 2))
        case CombinationType.WITHOUT_REPLACEMENT:
            return list(it.combinations(names, 2))


def suptitle(desc: dict, line_keys: list[list[int]]) -> str:
    k = dict([x for x in enumerate(desc)])

    def elem(i: int) -> str:
        key = k[i]
        value = desc[key]
        return f"{key}:{value}"

    def line(index: list[int]) -> str:
        return " ".join([elem(i) for i in index])

    return "\n".join([line(a) for a in line_keys])


def write_tournament_data(out_dir: Path, name: str, df: pd.DataFrame):
    filename = f"{name}.json"
    file = out_dir / filename
    df.to_json(file, indent=2)


def plot_rewards(
    out_dir: Path, name: str, _suptitle: str, result: pd.DataFrame
) -> Path:
    groups = result.groupby(["c1", "c2"])
    num_groups = groups.ngroups

    n_rows, n_cols = row_col(num_groups)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(12, 15))

    gl = list(groups)
    for i in range(n_cols):
        for j in range(n_rows):
            index = j * n_cols + i
            if n_cols == 1 and n_rows == 1:
                ax = axes
            elif n_cols == 1:
                ax = axes[j]
            elif n_rows == 1:
                ax = axes[i]
            else:
                ax = axes[j][i]
            if index < len(gl):
                key, grouped_data = gl[j * n_cols + i]
                name1, name2 = key
                title = f"{name1}  -  {name2}"
                ax.boxplot(grouped_data[["r1", "r2"]], labels=(name1, name2))
                ax.set_title(title)
                ax.set_ylim([-300, 300])
            else:
                ax.axis("off")
    plt.suptitle(_suptitle, y=0.98)
    filename = f"{name}.png"
    file_path = out_dir / filename
    fig.savefig(file_path)
    plt.close(fig)
    return file_path
