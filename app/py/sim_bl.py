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

def create_env(num_seconds:int, net_name: str, route_type: str):
    return CustomSumoEnvironment(
        net_file=f"app/data/{net_name}/{net_name}.net.xml",
        route_file=f"app/data/{net_name}/rou_{route_type}/{net_name}_{route_type}.rou.xml",
        out_csv_name=f"results/{date}/{net_name}_{route_type}/{net_name}_{route_type}_bl",
        use_gui=True,
        begin_time=0,
        num_seconds=num_seconds,
        time_to_teleport=-1,
        yellow_time=4,
        delta_time=5,
        min_green=10,
        single_agent=True,
        fixed_ts=True,
    )

def evaluation_baseline(env):
    obs, info = env.reset()
    done = False

    while not done:
        step_result = env.step(None)
        
        obs, rewards, terminated, truncated, info = step_result
        current_step = info.get("step")
        
        if current_step >= timesteps:
            done = True

        if done:
            print("シミュレーション終了.")
            env.save_csv(f"results/{date}/{net_name}_{route_type}/{net_name}_{route_type}_bl", 0)
            break

if __name__ == "__main__":
    timesteps = 100000
    num_seconds:int = timesteps
    date = "1_13"
    net_name = "tanimachi9"
    route_type = "c"

    env = create_env(num_seconds, net_name, route_type)
    evaluation_baseline(env)

