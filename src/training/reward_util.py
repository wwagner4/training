import json
import random
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from time import sleep

import matplotlib.pyplot as plt

import training.reward as rw
import training.simdb as sd
import training.simrunner as sr


@dataclass
class Sim:
    states: list[sr.SimulationState]
    properties1: list[list]
    properties2: list[list]


def util_sims_to_json_files(
    sim_names: list[str], base_dir: Path, db_host: str, db_port: int
) -> list[Path]:
    out_files = []
    with sd.create_client(db_host, db_port) as c:
        all_sims = sd.find_all(c)
        for sim in all_sims:
            if sim["name"] in sim_names:
                of = db_sim_to_json_file(sim, base_dir)
                out_files.append(of)
    return out_files


def db_sim_to_json_file(db_sim: dict, base_dir: Path) -> Path:
    name = db_sim["name"]
    robot_name1 = db_sim["robot1"]["name"]
    robot_name2 = db_sim["robot2"]["name"]
    sim_name = f"{name}_{robot_name1}_{robot_name2}"
    file_name = f"{sim_name}.json"
    file_path = base_dir / file_name
    with file_path.open("w", encoding="UTF-8") as f:
        sim_dict = {
            "name": sim_name,
            "winner": db_sim["events"],
            "states": db_sim["states"],
        }
        json.dump(sim_dict, f, indent=2)
    return file_path


def can_see_200(r1: sr.PosDir, r2: sr.PosDir, h: float) -> float | None:
    v = rw.can_see(r1, r2)
    if v is not None and v < 200:
        return h
    return None


def print_distances(file: Path):
    sim = read_sim_from_file(file)
    dists = [rw.dist(s) for s in sim.states]
    for d in dists:
        if d < 101:
            print(d)


def _visualize_collisions(
    simulation_files: list[Path], run_name: str, rows: int, columns: int, out_dir: Path
):
    lineWidth = 5.0
    scale = 2.0
    w = 11.69 * scale
    h = 8.25 * scale
    fig = plt.figure(figsize=(w, h), facecolor="white")
    for k, file in enumerate(simulation_files):
        sim = read_sim_from_file(file)
        dists = [rw.dist(s) for s in sim.states]
        # pprint(dists)
        r1_can_see = [can_see_200(s.robot1, s.robot2, 150) for s in sim.states]
        r2_can_see = [can_see_200(s.robot2, s.robot1, 130) for s in sim.states]
        ax = fig.add_subplot(columns, rows, k + 1)
        ax.plot(
            range(len(dists)),
            dists,
            color="tab:blue",
            label="distance",
            dashes=[4, 1],
            linewidth=2.0,
        )
        ax.plot(
            range(len(r1_can_see)),
            r1_can_see,
            color="tab:red",
            label="r1 see opp.",
            linewidth=lineWidth,
        )
        ax.plot(
            range(len(r2_can_see)),
            r2_can_see,
            color="tab:orange",
            label="r2 see opp.",
            linewidth=lineWidth,
        )
        ax.plot(
            [],
            [],
            " ",
            label=f"winner is {sim.winner}",
        )
        ax.legend()
        ax.set_title(sim.name)
    if not out_dir.exists():
        out_dir.mkdir()
    file_name = f"{run_name}.pdf"
    file_path = out_dir / file_name
    fig.savefig(file_path, dpi=300)
    print(f"saved color image to {file_path}")
    file_name_gray = f"{run_name}-gray.pdf"
    file_path_gray = out_dir / file_name_gray
    subprocess.run(["convert", "-grayscale", "average", file_path, file_path_gray])
    print(f"saved grayscale image to {file_path}")


def visualize_collisions(id: str, in_dir: Path, rows: int, cols: int, out_dir: Path):
    def match(file: Path):
        return file.is_file() and file.name.find(id) >= 0 and file.suffix == ".json"

    files = [f for f in in_dir.iterdir() if match(f)]
    sorted_files = sorted(files, key=lambda f: f.stem)
    print(f"### found {len(sorted_files)} files for id: {id}")
    if rows * cols != len(sorted_files):
        raise (ValueError(f"Found {len(sorted_files)} for n, m: {rows}, {cols}"))
    _visualize_collisions(sorted_files, id, rows, cols, out_dir)


def create_sims_visualize_collisions(
    n: int, m: int, host: str, id: str, base_dir: Path, out_dir: Path
):
    def create(i: int, run_name: str) -> str:
        sim_name = f"{run_name}-{i:02d}"
        c1 = random.choice([sr.ControllerName.TUMBLR, sr.ControllerName.STAY_IN_FIELD])
        c2 = random.choice(
            [sr.ControllerName.BLIND_TUMBLR, sr.ControllerName.STAY_IN_FIELD]
        )
        sr.start(
            sim_port=4444,
            sim_name=sim_name,
            controller_name1=c1,
            controller_name2=c2,
        )
        sleep(0.5)
        return sim_name

    n1 = n * m
    run_id = str(uuid.uuid4())[0:6]
    run_name = f"{host}-{id}-{run_id}"
    names = []
    for i in range(n1):
        nam = create(i, run_name)
        names.append(nam)
    print(f"## created {n} simulations. run_name: {run_name}")
    out_files = util_sims_to_json_files(sim_names=names, base_dir=base_dir)
    _visualize_collisions(
        simulation_files=out_files,
        run_name=run_name,
        rows=n,
        columns=m,
        out_dir=out_dir,
    )


def read_sim_from_file(file: Path) -> Sim:
    with file.open() as f:
        data_dict = json.load(f)
        states_object = data_dict["states"]
        states = [sr.SimulationState.from_dict(s) for s in states_object]
        properties1 = data_dict["winner"]["r1"]
        properties2 = data_dict["winner"]["r2"]
        return Sim(states, properties1, properties2)


def reward_analysis():
    @dataclass
    class RoboEventsReward:
        robo_events: rw.RobotEndEvents
        reward: float

    def read_events_reward(file: Path) -> (RoboEventsReward, RoboEventsReward):
        sim = read_sim_from_file(file)
        sim_events = rw.end_events_from_simulation_states(
            sim.states, sim.properties1, sim.properties2
        )
        rew1, rew2 = reward_handler.calculate_end_reward(
            sim.states, sim.properties1, sim.properties2
        )
        rer1 = RoboEventsReward(sim_events.robot1, rew1)
        rer2 = RoboEventsReward(sim_events.robot2, rew2)
        return rer1, rer2

    def print_header():
        print()
        print(
            f"{'name':10}"
            f"{'push':>10}"
            f"{'is_pushed':>15}"
            f"{'end':>15}"
            f"{'steps':>10}"
            f"{'reward':>10}"
        )
        print("-" * 70)

    def print_events_reward(events_reward: RoboEventsReward):
        # print(pformat(events_reward))
        print(
            f"{events_reward.robo_events.result.name:10}"
            f"{events_reward.robo_events.push_collision_count:10d}"
            f"{events_reward.robo_events.is_pushed_collision_count:15d}"
            f"{events_reward.robo_events.end.name:>15}"
            f"{events_reward.robo_events.steps_count_relative:10.3f}"
            f"{events_reward.reward:10.3f}"
        )

    def print_result(
        event_rewards: list[RoboEventsReward], result: rw.RobotEventsResult
    ):
        filtered = [e for e in event_rewards if e.robo_events.result == result]
        for er in filtered:
            print_events_reward(er)

    reward_handler = rw.EndConsiderAllRewardHandler()

    good_bad = """
    What is good or bad for
    
    WINNER:
    end_push + reward for fast winning -> Highest reward
    end_none -> No reward. The opponent waked out
    
    LOOSER:
    end_was_pushed -> Low minus. Do not blame too much a good robot for being pushed out 
        (eventually same reward/penalty for push/is_pushed as for draw)
    end_none + penalty for fast walk out -> High penalty 
        (eventually same reward/penalty for push/is_pushed as for draw)
    
    DRAW:
    reward per push
    penalty per is_pushed (think about how high the reward/penalty should be)
    
    Think about two situations:
    - Trained robot is bad compared to the opponent
    - Trained robot is almost as good as trained robot
    
    """  # noqa: E501

    data_dir = Path(__file__).parent.parent / "tests" / "data"
    files = [file for file in data_dir.iterdir() if file.suffix == ".json"]
    event_rewards = [
        read_events_reward(file) for file in files if file.suffix == ".json"
    ]
    flat_event_rewards = list(sum(event_rewards, ()))
    print(good_bad)

    print_header()
    print_result(flat_event_rewards, rw.RobotEventsResult.WINNER)
    print_header()
    print_result(flat_event_rewards, rw.RobotEventsResult.LOOSER)
    print_header()
    print_result(flat_event_rewards, rw.RobotEventsResult.DRAW)
