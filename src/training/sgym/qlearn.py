from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import gymnasium.spaces as gyms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbn

import training.helper as hlp
import training.sgym.core as sgym
import training.simrunner as sr
from training.simrunner import DiffDriveValues


@dataclass(frozen=True)
class QLearnConfig:
    learning_rate: float
    initial_epsilon: float
    epsilon_decay: float
    final_epsilon: float
    discount_factor: float
    doc_interval: int
    doc_duration: int


q_learn_env_config = sgym.SEnvConfig(
    max_wheel_speed=7,
    wheel_speed_steps=10,
    max_view_distance=700,
    view_distance_steps=3,
    max_simulation_steps=1000,
    dtype=np.float32,
)

q_learn_config = QLearnConfig(
    learning_rate=0.01,
    initial_epsilon=0.01,
    epsilon_decay=0.001,
    final_epsilon=0.05,
    discount_factor=0.95,
    doc_interval=1000,
    doc_duration=100,
)

def q_train(
    name: str,
    epoch_count: int,
    port: int,
    opponent_name: sr.ControllerName,
    reward_handler_name: sr.RewardHandlerName,
):
    loop_name = "q-train"

    reward_handler = sr.RewardHandlerProvider.get(reward_handler_name)
    print(
        f"Started {loop_name} e:{epoch_count} p:{port} "
        f"o:{opponent_name.value} rh:{reward_handler_name.value}"
    )

    run_id = hlp.time_id()

    record_count = 10
    record_nr = max(1, epoch_count // record_count)

    results = []
    start_time = datetime.now()
    training_name = f"Q-{name}-{run_id}"
    agent = None
    for epoch_nr in range(epoch_count):
        sim_name = f"{training_name}-{epoch_nr:06d}"
        opponent = sr.ControllerProvider.get(opponent_name)

        sim_info = None
        if epoch_nr % record_nr == 0:
            sim_info = sr.SimInfo(
                name1=f"{loop_name}-agent",
                desc1={"info": f"Agent with {loop_name} actions"},
                name2=opponent.name(),
                desc2=opponent.description(),
                port=port,
                sim_name=sim_name,
                max_simulation_steps=sgym.default_senv_config.max_simulation_steps,
            )

        env = sgym.SEnv(
            senv_config=q_learn_env_config,
            senv_mapping=q_sgym_mapping(q_learn_env_config),
            port=port,
            sim_name=sim_name,
            opponent=opponent,
            reward_handler=reward_handler,
            sim_info=sim_info,
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

        obs, _info = env.reset()
        cnt = 0
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
            cnt += 1

        if epoch_nr % (q_learn_config.doc_interval // 10) == 0 and epoch_nr > 0:
            progr = hlp.progress_str(epoch_nr, epoch_count, start_time)
            print(
                f"Finished epoch {training_name} {progr} " f"reward:{cuml_reward:10.2f}"
            )
        results.append(
            {
                "sim_steps": cnt,
                "max_sim_steps": q_learn_env_config.max_simulation_steps,
                "epoch_nr": epoch_nr,
                "max_epoch_nr": epoch_count,
                "reward": cuml_reward,
            }
        )
        env.close()
        work_dir = Path.home() / "tmp" / "sumosim" / "q" / training_name
        work_dir.mkdir(exist_ok=True, parents=True)

        if do_plot_q_values(
            epoch_nr, q_learn_config.doc_interval, q_learn_config.doc_duration
        ):
            document_q_values(training_name, agent, epoch_nr, work_dir)
        if epoch_nr % q_learn_config.doc_interval == 0 and epoch_nr > 0:
            document(training_name, results, work_dir)


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
    def view_mapping() -> int:
        match sensor.opponent_in_sector:
            case sr.SectorName.UNDEF:
                return 0
            case sr.SectorName.LEFT:
                return 1
            case sr.SectorName.CENTER:
                return 2
            case sr.SectorName.RIGHT:
                return 3
            case _:
                raise ValueError(f"Wrong sector name {sensor.opponent_in_sector}")

    def discrete(distance: float) -> int:
        return hlp.cont_to_discrete(
            distance, 0.0, config.max_view_distance, config.view_distance_steps
        )

    return (
        view_mapping(),
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
            diff_drives.append(DiffDriveValues(velos[i], velos[j]))

    def inner(index: int) -> sr.DiffDriveValues:
        return diff_drives[index]

    return inner


def do_plot_q_values(n: int, interval: int, duration: int) -> bool:
    return bool(n % interval < duration)


def document_q_values(name: str, agent: QAgent, epoch_nr: int | None, work_dir: Path):
    work_dir = work_dir / "v"
    work_dir.mkdir(parents=True, exist_ok=True)
    plot_q_values(agent, epoch_nr, name, work_dir)


def document(name: str, results: list[dict], work_dir: Path):
    data_path = work_dir / f"{name}.json"
    df = pd.DataFrame(results)
    df.to_json(data_path, indent=2)
    out = plot_boxplot(df, name, work_dir)
    print(f"--- wrote boxplot to {out}")


def plot_mean(data: pd.DataFrame, name: str, work_dir: Path) -> Path:
    y = data["reward"]
    x1, y1 = hlp.compress_means(y, 100)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))
    ax.plot(x1, y1, label="reward")
    ax.set_title(name)
    ax.set_ylim(ymin=-100, ymax=100)
    out_path = work_dir / f"{name}-mean.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_q_values(agent: QAgent, epoch_nr: int, name: str, work_dir: Path) -> Path:
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
    ax.set_title(f"Q Values for {name} epoch {epoch_nr}")
    ax.set_xlabel("action")
    ax.set_ylabel("observation")

    out_path = work_dir / f"{name}-{epoch_nr:010d}-heat.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_boxplot(data: pd.DataFrame, name: str, work_dir: Path) -> Path:
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
        croped = (data_len // n) * n
        split = np.split(d[0:croped], n)
        diff = croped // n
        xs = range(0, croped, diff)
        xs_str = [str(x) for x in xs]
        return xs_str, split

    x1, y1 = split_data(y, 15)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))
    ax.boxplot(y1, labels=x1)
    ax.set_title(f"{column} {name}")
    ax.set_ylim(ymin=-200, ymax=200)
    ax.set_ylabel(column)
    ax.set_xlabel("epoch nr")
    ax.tick_params(axis="x", rotation=45)
    out_path = work_dir / f"{name}-boxplot.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_all(data: pd.DataFrame, name: str, work_dir: Path) -> Path:
    y = data["reward"]
    window_size = 1
    y1 = np.convolve(y, np.ones(window_size) / window_size, mode="valid")
    window_size = 10
    y2 = np.convolve(y, np.ones(window_size) / window_size, mode="valid")

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))
    ax.plot(y1, label="reward")
    ax.plot(y2, label="reward (flat)")
    ax.set_title(name)
    out_path = work_dir / f"{name}-all.png"

    fig.savefig(out_path)
    plt.close(fig)
    return out_path
