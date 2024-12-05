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

def my_reward_fn(traffic_tignal):
    waiting_time_per_lane = traffic_tignal.get_accumulated_waiting_time_per_lane()
    total_waiting_time = sum(waiting_time_per_lane)
    return -total_waiting_time

def create_env(num_seconds:int, net_name: str, route_type: str):
    return CustomSumoEnvironment(
        net_file=f"/Users/chashu/Desktop/dev/sumorl-venv/app/data/{net_name}/{net_name}.net.xml",
        route_file=f"/Users/chashu/Desktop/dev/sumorl-venv/app/data/{net_name}/rou_{route_type}/{net_name}_{route_type}.rou.xml",
        out_csv_name=f"/Users/chashu/Desktop/dev/sumorl-venv/app/outputs/{net_name}_{route_type}",
        use_gui=True,
        begin_time=0,
        num_seconds=num_seconds,
        time_to_teleport=800,
        yellow_time=3,
        delta_time=5,
        min_green=10,
        # reward_fn=my_reward_fn,
        single_agent=True,
    )

def create_model(env):
    return DQN(
        env=env,
        policy="MlpPolicy",
        learning_rate=1e-2,
        learning_starts=1000,
        buffer_size=478864,
        train_freq=3,
        target_update_interval=7395,
        exploration_fraction=0.12,
        exploration_initial_eps=0.22,
        exploration_final_eps=0.07,
        verbose=1,
    )

def evaluation_model(env, net_name: str, route_type: str):
    epi_rewards = []
    obs, info = env.reset()
    done = False

    while not done:
        action, _states = model.predict(obs)
        step_result = env.step(action)
        
        obs, rewards, terminated, truncated, info = step_result
        current_step = info.get("step")

        if current_step >= 72000:
            done = True

        if done:
            print("シミュレーション終了.")
            env.save_csv(f"/Users/chashu/Desktop/dev/sumorl-venv/app/outputs/{net_name}_{route_type}", 0)
            break

if __name__ == "__main__":
    timesteps = 72000
    num_seconds:int = timesteps * 5
    net_name = "ehira"
    route_type = "b"

    env = create_env(num_seconds, net_name, route_type)
    model = create_model(env)

    model.learn(total_timesteps=timesteps)

    evaluation_model(env, net_name, route_type)

    

