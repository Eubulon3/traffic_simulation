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

def get_name(reward_fn):
    if callable(reward_fn):
        return reward_fn.__name__
    else:
        return reward_fn

def total_waiting_time(traffic_tignal):
    reward = sum(traffic_tignal.get_accumulated_waiting_time_per_lane()) / 100
    return -reward

def create_env(num_seconds:int, net_name: str, route_type: str, reward: str):
    return CustomSumoEnvironment(
        net_file=f"app/data/{net_name}/{net_name}.net.xml",
        route_file=f"app/data/{net_name}/rou_{route_type}/{net_name}_{route_type}.rou.xml",
        out_csv_name=f"results/{date}/{net_name}_{route_type}/{net_name}_{route_type}_{reward}",
        use_gui=True,
        begin_time=0,
        num_seconds=num_seconds,
        time_to_teleport=-1,
        yellow_time=4,
        delta_time=5,
        min_green=10,
        reward_fn=reward_fn,
        single_agent=True,
        additional_sumo_cmd=f"--gui-settings-file app/data/{net_name}/{net_name}_decals.xml --delay 30"
    )

def create_model(env, lerning_rate, exploration_fraction):
    return DQN(
        env=env,
        policy="MlpPolicy",
        learning_rate=lerning_rate,
        buffer_size=100000,
        batch_size=200,
        exploration_fraction=exploration_fraction,
    )

def evaluation_model(env, net_name: str, route_type: str):
    epi_rewards = []
    obs, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _states = model.predict(obs)
        step_result = env.step(action)
        
        obs, rewards, terminated, truncated, info = step_result
        current_step = info.get("step")
        total_reward += rewards

        if current_step >= 100000:
            done = True

        if done:
            print("done.")
            print(f"累計報酬:{total_reward}")
            env.save_csv(f"results/{date}/{net_name}_{route_type}/{net_name}_{route_type}_{reward}", 0)
            break

if __name__ == "__main__":
    timesteps = 100000
    num_seconds:int = 100000
    date = "1_15"
    net_name = "ehira"
    route_type = "a"
    reward_fn = total_waiting_time
    lerning_rate = 60e-4
    exploration_fraction = 0.08
    reward = get_name(reward_fn)

    env = create_env(num_seconds, net_name, route_type, reward)
    model = create_model(env, lerning_rate, exploration_fraction)

    model.learn(total_timesteps=timesteps)

    evaluation_model(env, net_name, route_type)

    

