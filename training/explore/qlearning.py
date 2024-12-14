import time
from datetime import datetime

import numpy as np

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
    x = (
        np.random.rand(
            30,
        )
        * 0.1
    )
    print(x)
