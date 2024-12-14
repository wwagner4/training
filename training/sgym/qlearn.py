from collections import defaultdict
from collections.abc import Callable
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

q_train_config = sgym.SEnvConfig(
    max_wheel_speed=7,
    wheel_speed_steps=10,
    max_view_distance=700,
    view_distance_steps=3,
    max_simulation_steps=1000,
    dtype=np.float32,
)


def q_train(
    name: str,
    epoch_count: int,
    port: int,
    opponent_name: sr.ControllerName,
    reward_handler_name: sr.RewardHandlerName,
):
    loop_name = "q-train"
    doc_interval = 100

    reward_handler = sr.RewardHandlerProvider.get(reward_handler_name)
    print(
        f"### sgym {loop_name} e:{epoch_count} p:{port} "
        f"o:{opponent_name.value} rh:{reward_handler_name.value}"
    )

    run_id = hlp.time_id()

    record_count = 10
    record_nr = max(1, epoch_count // record_count)

    results = []
    start_time = datetime.now()
    training_name = f"Q-{name}-{run_id}"
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
            senv_config=q_train_config,
            senv_mapping=q_sgym_mapping(q_train_config),
            port=port,
            sim_name=sim_name,
            opponent=opponent,
            reward_handler=reward_handler,
            sim_info=sim_info,
        )

        agent = QAgent(
            env=env,
            learning_rate=0.01,
            initial_epsilon=0.01,
            epsilon_decay=0.001,
            final_epsilon=0.05,
            discount_factor=0.95,
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

        progr = hlp.progress_str(epoch_nr, epoch_count, start_time)
        print(
            f"### finished epoch {training_name} {progr} " f"reward:{cuml_reward:10.2f}"
        )
        results.append(
            {
                "sim_steps": cnt,
                "max_sim_steps": q_train_config.max_simulation_steps,
                "epoch_nr": epoch_nr,
                "max_epoch_nr": epoch_count,
                "reward": cuml_reward,
            }
        )
        env.close()
        if epoch_nr % doc_interval == 0 and epoch_nr > 0:
            document(training_name, results, agent, epoch_nr)
    document(training_name, results, agent, None)


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
    return (
        np.random.rand(
            n,
        )
        * 0.1
    )


class QAgent:
    def __init__(
        self,
        env: gym.Env,
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
        # Make reward a positive value
        reward = (reward + 150) * 0.1
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        current_q_value = self.q_values[obs][action]
        next_q_value = max(0.0, current_q_value + self.lr * temporal_difference)
        # print(f"## change q value for {obs} {action} :
        # {current_q_value:5.4f} -> {next_q_value:5.4f}")
        self.q_values[obs][action] = next_q_value
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


def document(name: str, results: list[dict], agent: QAgent, epoch_nr: int | None):
    work_dir = Path.home() / "tmp" / "sumosim"
    work_dir.mkdir(exist_ok=True, parents=True)

    heat_path = plot_q_values(agent, epoch_nr, name, work_dir)
    print(f"Wrote heatmap to {heat_path}")

    data_path = work_dir / f"{name}.json"
    df = pd.DataFrame(results)
    df.to_json(data_path, indent=2)
    print(f"Wrote data to {data_path}")

    plot_path = plot_boxplot(df, name, work_dir)
    print(f"Wrote plot to {plot_path}")


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


def plot_q_values(
    agent: QAgent, epoch_nr: int | None, name: str, work_dir: Path
) -> Path:
    def all_obs() -> list:
        all = []
        for i0 in range(4):
            for i1 in range(q_train_config.view_distance_steps):
                for i2 in range(q_train_config.view_distance_steps):
                    for i3 in range(q_train_config.view_distance_steps):
                        all.append((i0, i1, i2, i3))
        return all

    all = all_obs()
    mv = []
    for obs in all:
        values = agent.q_values[obs]
        mv.append(values)
    matrix = np.matrix(mv, dtype=q_train_config.dtype)
    print(f"matrix {matrix.shape}")
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))
    sbn.heatmap(matrix, vmin=0.0, vmax=1.0, ax=ax)

    out_path = work_dir / f"{name}-heat.png"
    if epoch_nr:
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
