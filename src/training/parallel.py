import subprocess as sp
import socket
import itertools as it
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

import training.helper as hlp

_port = 4444


class ParallelConfig(Enum):
    Q_TRYOUT = "q-tryout"


@dataclass(frozen=True)
class TrainConfig:
    name: str
    values: dict


def create_train_configs1(
    parallel_config: ParallelConfig, max_parallel: int
) -> list[list[TrainConfig]]:
    match parallel_config:
        case ParallelConfig.Q_TRYOUT:
            values_dict = {
                "L": [0.1, 0.01, 0.001],
                "E": [0.1, 0.05, 0.01],
                "D": [0.99, 0.95, 0.5],
            }
            return create_train_configs(values_dict, max_parallel)
        case _:
            raise ValueError(f"Invalid Parallel Config {parallel_config.value}")


def create_train_configs(
    values_dict: dict, max_parallel: int
) -> list[list[TrainConfig]]:
    def to_dict(keys: list, batch_values: list) -> dict:
        return [dict(zip(keys, value)) for value in batch_values]

    def to_ids(dicts: list[dict]) -> list[str]:
        def to_id(dictionary: dict) -> str:
            return "".join([f"{key}{index}" for key, index in dictionary.items()])

        return [to_id(dictionary) for dictionary in dicts]

    def create_batched_dicts(
        keys: list, lists: list, max_parallel: 20
    ) -> list[list[dict]]:
        prod_values = list(it.product(*lists))
        batch_size = (len(prod_values) + max_parallel - 1) // max_parallel
        batched_values = it.batched(prod_values, batch_size)
        return [to_dict(keys, values) for values in batched_values]

    _keys = values_dict.keys()
    _values = [values_dict[k] for k in _keys]
    batched_dicts = create_batched_dicts(_keys, _values, max_parallel)

    index_values = [range(len(list(value))) for value in _values]
    batched_index_dicts = create_batched_dicts(_keys, index_values, max_parallel)
    ids = [to_ids(d) for d in batched_index_dicts]

    double_zipped = [zip(a, b) for a, b in (zip(batched_dicts, ids))]
    return [
        [TrainConfig(values=values, name=name) for values, name in zipped]
        for zipped in double_zipped
    ]


def call(command: list[str]) -> str:
    process = sp.Popen(command, stdout=sp.PIPE, stderr=sp.PIPE)
    b_out, b_err = process.communicate()
    if b_err:
        cmd_str = " ".join(command)
        msg = f"ERROR: calling '{cmd_str}'. \n{b_err.decode()}"
        raise RuntimeError(msg)
    return b_out.decode()


def create_network(name: str) -> str:
    try:
        return call(["docker", "network", "create", name])
    except RuntimeError as er:
        if "exists" not in str(er).lower():
            raise er
        return "<network exists>"


def start_simulator(sim_name: str, network_name: str) -> str:
    try:
        return call(
            [
                "docker",
                "run",
                "-d",
                "--rm",
                "--name",
                sim_name,
                "--network",
                network_name,
                "sumo",
                "sumo",
                "udp",
                "--port",
                str(_port),
            ]
        )
    except RuntimeError as er:
        if "is already in use by container" not in str(er).lower():
            raise er
        return "<already running>"


def parse_parallel_indexes(
    parallel_indexes: str, parallel_config: ParallelConfig, max_parallel: int
) -> list[int]:
    configs = create_train_configs1(parallel_config, max_parallel)
    if parallel_indexes.lower() == "all":
        return list(range(len(configs)))
    indexes = hlp.parse_integers(parallel_indexes)
    max_index = len(configs) - 1
    for index in indexes:
        if index > max_index:
            raise ValueError(
                f"ERROR: Cannot start simulation for parallel index {index} "
                f"of {parallel_config.value} "
                f"with max_parallel {max_parallel}. Max index is {max_index}"
            )
    return indexes


def start_training(
    name: str,
    parallel_config: ParallelConfig,
    max_parallel: int,
    parallel_index: int,
    sim_name: str,
    network_name: str,
    epoch_count: int,
    db_host: str,
    db_port: str,
    keep_container: bool,
    record: bool,
    out_dir: Path,
) -> str:
    out_dir_str = str(out_dir.absolute())
    user = call(["id", "-u"]).strip()
    group = call(["id", "-g"]).strip()
    db_host_ip = socket.gethostbyname(db_host)
    train_name = f"sumo-train{parallel_index:02d}"
    cmd = [
        "docker",
        "run",
        "-d",
        None if keep_container else "--rm",
        "-e",
        "PYTHONUNBUFFERED=True",
        "--name",
        train_name,
        "--network",
        network_name,
        "--user",
        f"{user}:{group}",
        "-v",
        f"{out_dir_str}:/tmp",
        "sumot",
        "uv",
        "run",
        "sumot",
        "qconfig",
        "--name",
        name,
        "--record" if record else None,
        "--parallel-config",
        parallel_config.value,
        "--max-parallel",
        str(max_parallel),
        "--parallel-index",
        str(parallel_index),
        "--epoch-count",
        str(epoch_count),
        "--sim-port",
        str(_port),
        "--sim-host",
        sim_name,
        "--out-dir",
        "/tmp",
        "--db-host",
        db_host_ip,
        "--db-port",
        str(db_port),
    ]
    cmd = [x for x in cmd if x is not None]
    print(f"Start training using: '{' '.join(cmd)}'")
    return call(cmd)


def start_training_configuration(
    name: str,
    parallel_config: ParallelConfig,
    max_parallel: int,
    parallel_index: int,
    epoch_count: int,
    db_host: str,
    db_port: int,
    keep_container: bool,
    record: bool,
    out_dir: Path,
):
    out_dir.mkdir(exist_ok=True, parents=True)
    network_name = f"sumo{parallel_index:02d}"
    sim_name = f"sumo{parallel_index:02d}"

    network_id = create_network(network_name)
    print(f"Created network {network_name} {network_id}")

    sim_run_id = start_simulator(sim_name, network_name)
    print(f"Started simulator {sim_name} {sim_run_id}")

    training_run_id = start_training(
        name=name,
        parallel_config=parallel_config,
        max_parallel=max_parallel,
        parallel_index=parallel_index,
        sim_name=sim_name,
        network_name=network_name,
        epoch_count=epoch_count,
        db_host=db_host,
        db_port=db_port,
        keep_container=keep_container,
        record=record,
        out_dir=out_dir,
    )
    print(f"Started training {name} {training_run_id}")


def subdir_exists(work_dir: Path, prefix: str) -> bool:
    for x in work_dir.iterdir():
        if x.name.startswith(prefix):
            return True
    return False


def parallel_main(
    name: str,
    parallel_config: ParallelConfig,
    max_parallel: int,
    parallel_indexes: str,
    epoch_count: str,
    db_host: str,
    db_port: int,
    keep_container: bool,
    record: bool,
    out_dir: str,
):
    print("Started parallel")
    out_path = Path(out_dir)
    out_path.mkdir(exist_ok=True, parents=True)
    if subdir_exists(out_path, name):
        raise RuntimeError(f"Output directory '{out_path}' {name} already exists")
    for parallel_index in parse_parallel_indexes(
        parallel_indexes, parallel_config, max_parallel
    ):
        start_training_configuration(
            name,
            parallel_config,
            max_parallel,
            parallel_index,
            epoch_count,
            db_host,
            db_port,
            keep_container,
            record,
            out_path,
        )
