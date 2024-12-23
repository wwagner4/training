import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
import training.sgym.qlearn as ql
import random
import pandas as pd
from pathlib import Path

def main():
    print("-- tryout")
    data = [-100 + x for x in range(200)]
    df = pd.DataFrame(data, columns=["reward"])
    work_dir = Path.home() / "tmp" / "sumosim" / "tryout"
    print(work_dir.exists())
    pl = ql.plot_boxplot(df, "LINEAR", ql.default_q_learn_config, work_dir)
    print(pl)
