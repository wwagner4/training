import time
from datetime import datetime
from pathlib import Path

import pandas as pd

import training.explore.blackjack as bj
import training.helper as hlp
import training.sgym.qlearn as sgym_qlearn


def tryout01():
    s = datetime.now()
    count = 20
    for i in range(count):
        time.sleep(0.287362)
        print(f"{hlp.progress_str(i, count, s)}")

    print(f"finished at:{datetime.now()}")


def tryout03():
    bj.main()


def tryout02():
    n = 121
    x = sgym_qlearn.initial_rewards(n)
    for i, v in enumerate(x):
        print(f"-- {i} {v}")


def tryout():
    name = "Q-A-34476-131000"
    work_dir = Path.home() / "tmp" / "sumosim"
    data_file = f"{name}.json"
    data_path = work_dir / data_file
    data = pd.read_json(data_path)
    out_file = sgym_qlearn.plot_boxplot(data, name, work_dir)
    print(f"wrote plot {out_file}")
