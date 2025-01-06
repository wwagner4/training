from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import gymnasium.spaces as gyms
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sbn

import training.helper as hlp
import training.sgym.core as sgym
import training.simrunner as sr
import training.parallel as parallel


@dataclass(frozen=True)
class QLearnConfig:
    learning_rate: float
    initial_epsilon: float
    epsilon_decay: float
    final_epsilon: float
    discount_factor: float


default_env_config = sgym.SEnvConfig(
    max_wheel_speed=7,
    wheel_speed_steps=10,
    max_view_distance=700,
    view_distance_steps=3,
    max_simulation_steps=1000,
    dtype=np.float32,
)

default_q_learn_config = QLearnConfig(
    learning_rate=0.01,
    initial_epsilon=0.01,
    epsilon_decay=0.001,
    final_epsilon=0.05,
    discount_factor=0.95,
)


def q_config(
    name: str,
    record: bool,
    parallel_config: parallel.ParallelConfig,
    max_parallel: int,
    parallel_index: int,
    sim_host: str,
    sim_port: int,
    db_host: str,
    db_port: int,
    epoch_count: int,
    out_dir: str,
):
    def call_q_train_with_config(parallel_config: parallel.ParallelConfig):
        q_learn_config = default_q_learn_config
        parallel_config_values: dict = parallel_config.values
        if parallel_config_values.get("L") is not None:
            q_learn_config = replace(
                q_learn_config, learning_rate=parallel_config_values["L"]
            )
        if parallel_config_values.get("E") is not None:
            q_learn_config = replace(
                q_learn_config,
                initial_epsilon=parallel_config_values["E"],
                final_epsilon=parallel_config_values["E"],
                epsilon_decay=0.0,
            )
        if parallel_config_values.get("D") is not None:
            q_learn_config = replace(
                q_learn_config, discount_factor=parallel_config_values["D"]
            )

        _q_train(
            name=f"{name}-{parallel_config.name}",
            auto_naming=False,
            epoch_count=epoch_count,
            sim_host=sim_host,
            sim_port=sim_port,
            db_host=db_host,
            db_port=db_port,
            opponent_name=sr.ControllerName.STAND_STILL,
            reward_handler_name=sr.RewardHandlerName.CONTINUOUS_CONSIDER_ALL,
            record=record,
            plot_q_values_full=False,
            out_dir=out_dir,
            q_learn_env_config=default_env_config,
            q_learn_config=q_learn_config,
        )

    configs = parallel.create_train_configs1(parallel_config, max_parallel)
    if parallel_index >= len(configs):
        raise ValueError(
            f"Cannot run 'q_config' because the parallel index {parallel_index} exceeds the maximum index for parallel_config {parallel_config.value}. Max index is {len(configs) - 1}"
        )
    _configs = configs[parallel_index]
    for c in _configs:
        call_q_train_with_config(c)
    print(f"Finished parallel training n:{name}")


def q_train(
    name: str,
    auto_naming: bool,
    epoch_count: int,
    sim_host: str,
    sim_port: int,
    db_host: str,
    db_port: int,
    opponent_name: sr.ControllerName,
    reward_handler_name: sr.RewardHandlerName,
    record: bool,
    plot_q_values_full: bool,
    out_dir: str,
) -> int:
    return _q_train(
        name=name,
        auto_naming=auto_naming,
        epoch_count=epoch_count,
        sim_host=sim_host,
        sim_port=sim_port,
        db_host=db_host,
        db_port=db_port,
        opponent_name=opponent_name,
        reward_handler_name=reward_handler_name,
        record=record,
        plot_q_values_full=plot_q_values_full,
        out_dir=out_dir,
        q_learn_env_config=default_env_config,
        q_learn_config=default_q_learn_config,
    )


def _q_train(
    name: str,
    auto_naming: str,
    epoch_count: int,
    sim_host: str,
    sim_port: int,
    db_host: str,
    db_port: int,
    opponent_name: sr.ControllerName,
    reward_handler_name: sr.RewardHandlerName,
    record: bool,
    plot_q_values_full: bool,
    out_dir: str,
    q_learn_env_config: sgym.SEnvConfig,
    q_learn_config: QLearnConfig,
) -> int:
    reward_handler = sr.RewardHandlerProvider.get(reward_handler_name)
    results = []
    start_time = datetime.now()
    if auto_naming:
        name = f"{name}-{hlp.unique()}"
    out_path = Path(out_dir) / name
    if out_path.exists():
        raise FileExistsError(f"{name} was already used {out_path}. Choose another one")
    out_path.mkdir(parents=True)
    loop_name = "q"

    doc_interval = calc_doc_interval(epoch_count)
    doc_duration = calc_doc_duration(doc_interval)
    record_count = calc_record_count(epoch_count)

    record_interval = max(1, epoch_count // record_count)
    print(
        f"Started name:{name} loop-name:{loop_name} epoch-count:{epoch_count} sim-host:{sim_host} sim-port:{sim_port} "
        f"opponent:{opponent_name.value} reward-handler:{reward_handler_name.value} "
        f"doc-interval:{doc_interval} doc-duration:{doc_duration}  record-count:{record_count} record:{record} out-path:{out_path.absolute()}"
    )
    opponent = sr.ControllerProvider.get(opponent_name)
    env = sgym.SEnv(
        senv_config=q_learn_env_config,
        senv_mapping=q_sgym_mapping(q_learn_env_config),
        sim_host=sim_host,
        sim_port=sim_port,
        db_host=db_host,
        db_port=db_port,
        opponent=opponent,
        reward_handler=reward_handler,
    )
    agent = QAgent(
        env=env,
        reward_handler=reward_handler_name,
        learning_rate=q_learn_config.learning_rate,
        initial_epsilon=q_learn_config.initial_epsilon,
        epsilon_decay=q_learn_config.epsilon_decay,
        final_epsilon=q_learn_config.final_epsilon,
        discount_factor=q_learn_config.discount_factor,
    )
    for epoch_nr in range(epoch_count):
        sim_name = f"{name}-{epoch_nr:06d}"
        sim_info = None
        if record and (
            epoch_nr % record_interval == 0 or is_last(epoch_count, epoch_nr)
        ):
            sim_info = sr.SimInfo(
                name1=f"{loop_name}-agent",
                desc1={"info": f"Agent with {loop_name} actions"},
                name2=opponent.name(),
                desc2=opponent.description(),
                port=sim_port,
                sim_name=sim_name,
                max_simulation_steps=sgym.default_senv_config.max_simulation_steps,
            )
        obs, _info = env.reset(sim_info, sim_name)
        sim_nr = 0
        episode_over = False
        cuml_reward = 0.0
        while not episode_over:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            # print(f"# obs:{obs} a:{action} next_obs:{next_obs}")
            agent.update(obs, action, reward, terminated, next_obs)
            cuml_reward += reward
            episode_over = terminated or truncated
            obs = next_obs
            if plot_q_values_full:
                document_q_values(
                    name, agent, epoch_nr, sim_nr, out_path, q_learn_env_config
                )
            sim_nr += 1

        results.append(
            {
                "sim_steps": sim_nr,
                "max_sim_steps": q_learn_env_config.max_simulation_steps,
                "epoch_nr": epoch_nr,
                "max_epoch_nr": epoch_count,
                "reward": cuml_reward,
            }
        )
        if epoch_nr % (max(1, doc_interval // 10)) == 0:
            progr = hlp.progress_str(epoch_nr, epoch_count, start_time)
            print(f"Finished epoch {name} {progr} " f"reward:{cuml_reward:15.5f}")
        if do_plot_q_values(
            epoch_nr, doc_interval, doc_duration, plot_q_values_full
        ) or is_last(epoch_count, epoch_nr):
            document_q_values(
                name, agent, epoch_nr, sim_nr, out_path, q_learn_env_config
            )
        if (epoch_nr % doc_interval == 0 and epoch_nr > 0) or is_last(
            epoch_count, epoch_nr
        ):
            document(name, results, epoch_nr, q_learn_config, out_path)
    env.close()
    print(f"Finished training {name} {loop_name} p:{sim_port}")
    return sim_port


def calc_doc_interval(epoch_count: int) -> int:
    if epoch_count < 50:
        return 1
    elif epoch_count < 1000:
        return 10
    elif epoch_count < 10000:
        return 100
    else:
        return 1000


def calc_doc_duration(doc_interval: int) -> int:
    if doc_interval <= 100:
        return doc_interval
    elif doc_interval <= 1000:
        return doc_interval // 10
    else:
        return doc_interval // 100


def calc_record_count(epoch_count: int) -> int:
    if epoch_count < 20:
        return epoch_count
    else:
        return 10


def is_last(epoch_count, epoch_nr):
    return epoch_nr == (epoch_count - 1)


def get_q_act_space(config: sgym.SEnvConfig) -> gym.Space:
    n = (config.wheel_speed_steps + 1) * (config.wheel_speed_steps + 1)
    return gyms.Discrete(n)


def get_q_obs_space(config: sgym.SEnvConfig) -> gym.Space:
    return gyms.Tuple(
        (
            gyms.Discrete(n=4),
            gyms.Discrete(n=config.view_distance_steps),
            gyms.Discrete(n=config.view_distance_steps),
            gyms.Discrete(n=config.view_distance_steps),
        )
    )


def map_q_sensor_to_obs(
    sensor: sr.CombiSensor, config: sgym.SEnvConfig
) -> tuple[int, int, int, int]:
    def discrete(distance: float) -> int:
        return hlp.cont_to_discrete(
            distance, 0.0, config.max_view_distance, config.view_distance_steps
        )

    return (
        sr.sector_mapping(sensor.opponent_in_sector),
        discrete(sensor.left_distance),
        discrete(sensor.front_distance),
        discrete(sensor.right_distance),
    )


def curry_q_act_to_diff_drive(
    config: sgym.SEnvConfig,
) -> Callable[[any, sgym.SEnvConfig], sr.DiffDriveValues]:
    velo_from_index = _curry_velo_from_index(
        config.max_wheel_speed, config.wheel_speed_steps
    )

    def inner(a_space: int, _config: sgym.SEnvConfig) -> sr.DiffDriveValues:
        return velo_from_index(a_space)

    return inner


def q_sgym_mapping(cfg: sgym.SEnvConfig) -> sgym.SEnvMapping:
    return sgym.SEnvMapping(
        act_space=get_q_act_space,
        obs_space=get_q_obs_space,
        map_act=curry_q_act_to_diff_drive(cfg),
        map_sensor=map_q_sensor_to_obs,
    )


def initial_rewards(n: int) -> list[float]:
    return list(
        np.random.rand(
            n,
        )
        * 0.001
    )


def calc_next_q_value(
    reward: float,
    terminated: float,
    next_obs_q_values: list[float],
    current_q_value: float,
    discount_factor: float,
    learning_rate: float,
) -> tuple[float, float]:
    future_q_value = (not terminated) * np.max(next_obs_q_values)
    temporal_difference = reward + discount_factor * future_q_value - current_q_value
    q_value = current_q_value + learning_rate * temporal_difference
    return temporal_difference, q_value


def adjust_end(reward: float) -> float:
    min_value = -150.0
    max_value = 220.0
    _r = (reward - min_value) / (max_value - min_value)
    return min(max(0.0, _r), 1.0)


def adjust_cont(reward: float) -> float:
    min_value = -160.0
    max_value = 660.0
    _r = (reward - min_value) / (max_value - min_value)
    return min(max(0.0, _r), 1.0)


class QAgent:
    def __init__(
        self,
        env: gym.Env,
        reward_handler: sr.RewardHandlerName,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            env: The training environment
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """

        self.env = env
        self.q_values = defaultdict(lambda: initial_rewards(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.reward_handler = reward_handler

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, obs: tuple) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple,
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple,
    ):
        """Updates the Q-value of an action."""

        match self.reward_handler:
            case sr.RewardHandlerName.END_CONSIDER_ALL:
                reward = adjust_end(reward)
            case sr.RewardHandlerName.CONTINUOUS_CONSIDER_ALL:
                reward = adjust_cont(reward)
            case _:
                raise ValueError(f"Unknown reward handler {self.reward_handler}")

        temporal_difference, self.q_values[obs][action] = calc_next_q_value(
            reward,
            terminated,
            self.q_values[next_obs],
            self.q_values[obs][action],
            self.discount_factor,
            self.lr,
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


def _curry_velo_from_index(
    max_velo: float, velo_steps: int
) -> Callable[[int], sr.DiffDriveValues]:
    velos = hlp.cont_values(-max_velo, max_velo, velo_steps + 1)
    n = len(velos)
    diff_drives = []
    for i in range(n):
        for j in range(n):
            diff_drives.append(sr.DiffDriveValues(velos[i], velos[j]))

    def inner(index: int) -> sr.DiffDriveValues:
        return diff_drives[index]

    return inner


def do_plot_q_values(
    n: int, interval: int, duration: int, plot_q_values_full: bool
) -> bool:
    return not plot_q_values_full and bool(n % interval < duration)


def document_q_values(
    name: str,
    agent: QAgent,
    epoch_nr: int,
    sim_nr: int,
    work_dir: Path,
    q_learn_env_config: sgym.SEnvConfig,
):
    work_dir = work_dir / "v"
    work_dir.mkdir(parents=True, exist_ok=True)
    plot_q_values(agent, epoch_nr, sim_nr, name, work_dir, q_learn_env_config)


def document(
    name: str, results: list[dict], epoch_nr: int, config: QLearnConfig, work_dir: Path
):
    data_path = work_dir / f"{name}.json"
    df = pd.DataFrame(results)
    df.to_json(data_path, indent=2)
    plot_boxplot(df, name, config, work_dir)
    plot_all(df, name, config, work_dir)
    plain_dir = work_dir / "plain"
    plain_dir.mkdir(parents=True, exist_ok=True)
    plot_plain(df, name, epoch_nr, config, plain_dir)
    print(f"Wrote plots for {name} to {work_dir.absolute()}")


def plot_q_values(
    agent: QAgent,
    epoch_nr: int,
    sim_nr: int,
    name: str,
    work_dir: Path,
    q_learn_env_config: sgym.SEnvConfig,
) -> Path:
    matplotlib.use("agg")

    def all_obs() -> list:
        all = []
        for i0 in range(4):
            for i1 in range(q_learn_env_config.view_distance_steps):
                for i2 in range(q_learn_env_config.view_distance_steps):
                    for i3 in range(q_learn_env_config.view_distance_steps):
                        all.append((i0, i1, i2, i3))
        return all

    _all = all_obs()
    obs_action_data = []
    for obs in _all:
        values = agent.q_values[obs]
        obs_action_data.append(values)
    obs_action_matrix = np.matrix(obs_action_data, dtype=q_learn_env_config.dtype)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))
    sbn.heatmap(obs_action_matrix, vmin=0.0, vmax=1.0, ax=ax)
    ax.set_title(f"Q Values for {name}")
    ax.set_xlabel("action")
    ax.set_ylabel("observation")

    out_path = work_dir / f"{name}-{epoch_nr:08d}{sim_nr:08d}-heat.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def title(column: str, name: str, config: QLearnConfig) -> str:
    lines = [
        f"{column} {name}",
        f"discount:{config.discount_factor} learning-rate:{config.learning_rate}",
        f"initial-epsilon:{config.initial_epsilon} final-epsilon:{config.final_epsilon} epsilon-decay:{config.epsilon_decay}",
    ]
    return "\n".join(lines)


def plot_boxplot(
    data: pd.DataFrame, name: str, config: QLearnConfig, work_dir: Path
) -> Path:
    matplotlib.use("agg")
    column = "reward"
    y = data[column]

    def split_data(data: list[float], n: int) -> list[list[str], list[list[float]]]:
        data_len = len(data)
        if data_len < 10 * n:
            # Less than 10 data per boxplot
            return [str(data_len)], [data]
        if data_len <= n:
            return range(data_len), data
        d = np.array(data)
        cropped = (data_len // n) * n
        split = np.split(d[0:cropped], n)
        diff = cropped // n
        xs = range(0, cropped, diff)
        xs_str = [str(x) for x in xs]
        return xs_str, split

    x1, y1 = split_data(y, 15)
    try:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))
        ax.boxplot(y1, labels=x1)
        ax.set_title(title(column, name, config))
        ax.set_ylim(ymin=-20, ymax=300)
        ax.set_ylabel(column)
        ax.set_xlabel("epoch nr")
        ax.tick_params(axis="x", rotation=45)
        out_path = work_dir / f"{name}-boxplot.png"
        fig.savefig(out_path)
        plt.close(fig)
        return out_path
    except ValueError as ve:
        print(f"x1:{x1}")
        print(f"y1:{y1}")
        raise ve


def plot_all(
    data: pd.DataFrame, name: str, config: QLearnConfig, work_dir: Path
) -> Path:
    matplotlib.use("agg")
    column = "reward"
    y = data[column]
    window_size = 1
    y1 = np.convolve(y, np.ones(window_size) / window_size, mode="valid")
    window_size = 10
    y2 = np.convolve(y, np.ones(window_size) / window_size, mode="valid")

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))
    ax.plot(y1, label="reward")
    ax.plot(y2, label="reward (flat)")
    ax.set_title(title(column, name, config))
    out_path = work_dir / f"{name}-all.png"

    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_plain(
    data: pd.DataFrame, name: str, epoch_nr: int, config: QLearnConfig, work_dir: Path
) -> Path:
    matplotlib.use("agg")
    column = "reward"
    y = data[column]

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.plot(y, label="reward")
    ax.set_title(title(column, name, config))
    out_path = work_dir / f"{name}-{epoch_nr:05d}plain.png"

    fig.savefig(out_path)
    plt.close(fig)
    return out_path
