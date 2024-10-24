import time
from pathlib import Path

import pandas as pd

import training.simrunner_tournament as srt
from training.sgym_training import tid


def test_tid():
    for i in range(20):
        t = tid()
        print(f"### tid {i:3d} {t}")
        time.sleep(0.111111)


def test_display_results():
    dir = Path.home() / "tmp" / "sumosim"
    filename = "COMBI-06.json"
    file = dir / filename
    result = pd.read_json(file)
    srt.plot_rewards(dir, filename, result)


def main():
    test_display_results()
