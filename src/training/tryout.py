from pathlib import Path
import math

import matplotlib.pyplot as plt


def main():
    work_dir = Path.home() / "tmp"

    def func(x: float, g: float) -> float:
        f = math.pow(10, g)
        return math.pow(f, x - 1) + (math.pow(f, - 1) * (x - 1))

    dist = 100
    n = 4
    gs = [0, 0.5, 1, 1.5, 2]
    x = list([a / float(dist - 1) for a in range(dist)])

    x1 = [a / (n - 1) for a in range(n)]
    for g in gs:
        y1 = [func(x, g) for x in x1]
        y1_str = ", ".join([f"{y:5.3f}" for y in y1])
        print(f"{g:6.1f} {y1_str}")

    fig, ax = plt.subplots(figsize=(15, 15))
    for g in gs:
        y1 = [func(a, g) for a in x]
        ax.plot(x, y1)
    out_path = work_dir / "a.png"
    fig.savefig(out_path)
    plt.close(fig)
