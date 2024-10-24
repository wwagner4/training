import time

import training.sgym as sgym
import training.simrunner as sr


def tid() -> str:
    return f"{int(time.time() * 10) % 86400:05d}"


def main():
    port = 4444

    opponent_name = sr.ControllerName.TUMBLR
    reward_handler = sr.RewardHandlerProvider.get(sr.RewardHandlerName.FIRST)
    record = True
    epoch_count = 100

    print(
        f"### sgym tryout port:{port} " f"opponent:{opponent_name.name} record:{record}"
    )

    run_id = tid()
    for n in range(epoch_count):
        sim_name = f"SGYM-001-{run_id}-{n:03d}"
        opponent = sr.ControllerProvider.get(opponent_name)
        env = sgym.SEnv(
            senv_config=sgym.default_senv_config,
            port=port,
            sim_name=sim_name,
            opponent=opponent,
            reward_handler=reward_handler,
            record=record,
        )
        observation, info = env.reset()
        cnt = 0
        episode_over = False
        cuml_reward = 0.0
        while not episode_over:
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            cuml_reward += reward
            # if cnt > 0 and cnt % 50 == 0:
            #     print(f"### {cnt} reward:{reward} info:{info}")
            episode_over = terminated or truncated
            cnt += 1

        print(f"### finished {sim_name} {cuml_reward:10.2f}")
        env.close()
