import math

import training.sgym as sgym


def continuous_to_discrete(value: float, max_value: float, step_count: int) -> int:
    d = 2.0 * max_value / step_count
    i = int(math.floor((value + max_value) / d))
    return min(max(0, i), (step_count - 1))


def tryout():
    print("Exploring qlearning")

    config = sgym.default_senv_config

    print(f"### config {config}")
    _action_space = sgym.crete_action_space(sgym.default_senv_config)
    print(_action_space)

    for x in [-12, -10, -3.334, -3.332, 0, 3.332, 3.334, 4, 6, 10, 11, 1000]:
        _n = continuous_to_discrete(x, 10, 3)
        print(f"{x:10.4f} -->> {_n:<10d}")

    # for i  in range(5):
    #     action = action_space.sample()
    #     print(f"### action {i} - {type(action)} {action}")
