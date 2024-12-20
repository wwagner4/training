import pandas as pd
from pathlib import Path
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np


def reward_values():
    work_dir = Path.home() / "tmp" / "sumosim" / "start"
    file_c = work_dir / "MM-C-20977.json"
    file_e = work_dir / "MM-E-52085.json"
    df1 = pd.read_json(file_c)
    df2 = pd.read_json(file_e)

    def concat_rewards(df: pd.DataFrame) -> list[float]:
        r1 = df["r1"]
        r2 = df["r2"]
        a1 = list([float(x) for x in r1])
        a2 = list([float(x) for x in r2])
        return list(a1) + list(a2)

    data = {"continuous": concat_rewards(df1), "end": concat_rewards(df2)}

    fig, ax = plt.subplots(figsize=(5, 10))
    sb.boxplot(data, ax=ax)
    ax.grid(which="both", color="black", linestyle="dotted", linewidth=0.1)
    ax.yaxis.set_ticks(np.arange(-200, 700, 20))
    ax.set_title("Reward Values Distribution")
    ax.set_xlabel("reward handler")
    out_path = work_dir / "mm.png"
    fig.savefig(out_path)
    print(f"Wrote plot to {out_path}")


def main():
    reward_values()
