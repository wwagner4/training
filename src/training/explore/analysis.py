import training.sgym.qlearn as ql
import training.helper as hlp
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sb

from enum import Enum


class AnalysisName(Enum):
    NEXT_Q_VALUE = "next-q-value"
    ADJUST_REWARD = "adjust-reward"


class AnalysisReportName(Enum):
    VIDEOS = "videos"


def analysis_report_main(
    analysis_report_name: AnalysisReportName, base_dir: str, prefix: str
):
    match analysis_report_name:
        case AnalysisReportName.VIDEOS:
            videos(base_dir, prefix)
        case _:
            raise ValueError(f"Unknown analysis name {analysis_report_name}")


def videos(base_dir: str, prefix: str):
    def video(name: str, video_dir: Path, out_dir: Path) -> Path:
        infiles = f"{str(video_dir)}/*.png"
        outfile = f"{str(out_dir)}/{name}.mp4"
        print(
            f"name: {name} vd: {video_dir} od: {out_dir} ifs: {infiles}  of: {outfile}"
        )
        cmd = [
            "ffmpeg",
            "-framerate",
            "200",
            "-pattern_type",
            "glob",
            "-i",
            f"{str(infiles)}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(outfile),
        ]
        print(f"calling: '{' '.join(cmd)}'")
        hlp.call(cmd, ignore_stderr=True)
        print(f"Created video: {outfile}")

    base_path = Path(base_dir)
    if not base_path.exists():
        raise ValueError(f"Base directory does not exist {base_path.absolute()}")
    files = [p for p in base_path.iterdir() if p.name.startswith(prefix)]
    files_sorted = sorted(files, key=lambda p: p.name)
    for f in files_sorted:
        video(f.name, f / "v", base_dir)


def analysis_main(analysis_name: AnalysisName):
    match analysis_name:
        case AnalysisName.NEXT_Q_VALUE:
            next_q_value()
        case AnalysisName.ADJUST_REWARD:
            adjust_reward()
        case _:
            raise ValueError(f"Unknown analysis name {analysis_name}")


def adjust_reward():
    from training.sgym.qlearn import adjust_end
    from training.sgym.qlearn import adjust_cont

    rewards = hlp.create_values(100, -200, 800)
    ends = [adjust_end(x) for x in rewards]
    conts = [adjust_cont(x) for x in rewards]

    fig, ax = plt.subplots(figsize=(20, 20))
    sb.lineplot(x=rewards, y=rewards, ax=ax)
    sb.lineplot(x=rewards, y=ends, ax=ax)
    sb.lineplot(x=rewards, y=conts, ax=ax)
    ax.set_ylim(-0.1, 1.1)

    work_dir = Path.home() / "tmp" / "sumosim" / "adjust-reward"
    work_dir.mkdir(parents=True, exist_ok=True)
    file = work_dir / "adjust-reward-001.png"
    fig.savefig(file)
    print(f"Wrote analysis 'adjust-reward to {file}'")


def next_q_value():
    n_rows = 3
    n_cols = 3
    work_dir = Path.home() / "tmp" / "sumosim" / "tryout"
    discount_factor = 0.95
    learning_rate = 0.1

    def _next(reward: float, current_q: float, next_obs_q: float) -> float:
        _, _next_q = ql.calc_next_q_value(
            reward=reward,
            terminated=False,
            next_obs_q_values=[next_obs_q],
            current_q_value=current_q,
            discount_factor=discount_factor,
            learning_rate=learning_rate,
        )
        return _next_q

    def plot_next_obs_q(axs: list[list[plt.Axes]]):
        def data(next_obs_q: float):
            _rewards = hlp.create_values(100, 0.0, 1.0)
            _curr_values = hlp.create_values(5, 0.0, 1.0)
            result = []
            for cv in _curr_values:
                for r in _rewards:
                    next_q = _next(reward=r, current_q=cv, next_obs_q=next_obs_q)
                    result.append({"curr_q": cv, "next_q": next_q, "reward": r})
            data = pd.DataFrame(result)
            return data.pivot(index="reward", columns="curr_q", values="next_q")

        next_obs_qs = hlp.create_values(9, 0.0, 0.0)
        k = 0
        for i in range(n_rows):
            for j in range(n_cols):
                if k < len(next_obs_qs):
                    ax = axs[i][j]
                    nq = next_obs_qs[k]
                    _data = data(next_obs_q=nq)
                    ax = sb.lineplot(_data, ax=ax)
                    ax.set_title(f"next obs q {nq:.3f}")
                    ax.set_ylabel("next_q")
                    ax.set_xlim(0.0, 1.0)
                    ax.set_ylim(0.0, 1.0)
                    ax.grid()
                    k += 1

    work_dir.mkdir(exist_ok=True, parents=True)

    fig, axs = plt.subplots(
        nrows=n_rows, ncols=n_cols, figsize=(20, 20), constrained_layout=True
    )
    plot_next_obs_q(axs)
    file = work_dir / "next_q_value.png"
    fig.savefig(file)
    print(f"Wrote next q value analysis to {file}")
