import os
import sys

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("環境変数の設定をしてください")


from stable_baselines3.dqn.dqn import DQN
from custom_env import CustomSumoEnvironment
import numpy as np

def create_env(num_seconds:int, pattern:list, pattern_num:int):
    return CustomSumoEnvironment(
        net_file=f"/Users/chashu/Desktop/dev/sumorl-venv/app/data/4road_intersection/net/{pattern[pattern_num-1]}.net.xml",
        route_file=f"/Users/chashu/Desktop/dev/sumorl-venv/app/data/4road_intersection/rou/{pattern[pattern_num-1]}.rou.xml",
        out_csv_name=f"/Users/chashu/Desktop/dev/sumorl-venv/app/outputs/4road_intersection_pattern{str(int(pattern_num))}",
        use_gui=True,
        begin_time=0,
        num_seconds=num_seconds,
        time_to_teleport=-1,
        yellow_time=3,
        delta_time=5,
        min_green=10,
        # reward_fn=my_reward_fn,
        single_agent=True,
        fixed_ts=True,
    )

def evaluation_baseline(env, pattern_num):
    obs, info = env.reset()
    done = False

    while not done:
        step_result = env.step(None)
        
        obs, rewards, terminated, truncated, info = step_result
        done = truncated
        current_step = info.get("step")
        
        percent = (current_step / num_seconds) * 100
        if percent % 10 == 0:
            print(f"{int(percent)}% completed")

        if done:
            print("シミュレーション終了.")
            env.save_csv(f"/Users/chashu/Desktop/dev/sumorl-venv/app/outputs/4road_intersection_pattern{str(int(pattern_num))}_bl", 0)
            break

if __name__ == "__main__":
    num_seconds:int = 50000
    pattern_num:int = 4
    pattern:list = ["2way", "2way_right-arrow", "2way_right-lane", "3way"]

    env = create_env(num_seconds, pattern, pattern_num)
    evaluation_baseline(env, pattern_num)

