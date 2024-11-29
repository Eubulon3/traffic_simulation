import os
import sys

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("環境変数の設定をしてください")


from stable_baselines3.dqn.dqn import DQN
from sumo_rl import SumoEnvironment
from custom_env import CustomSumoEnvironment
import numpy as np


num_seconds = 10000


def my_reward_fn(traffic_tignal):
    waiting_time_per_lane = traffic_tignal.get_accumulated_waiting_time_per_lane()
    total_waiting_time = sum(waiting_time_per_lane)
    return -total_waiting_time

file_name = ["osm", "tanimachikyuchome"]

env = CustomSumoEnvironment(
    net_file=f"./app/data/nets/{file_name[0]}.net.xml",
    route_file=f"./app/data/rou/{file_name[0]}.rou.xml",
    out_csv_name=f"./app/outputs/{file_name[0]}",
    single_agent=True,
    use_gui=True,
    begin_time=0,
    num_seconds=num_seconds,
    time_to_teleport=500,
    yellow_time=4,
    delta_time=5,
    min_green=10,
    reward_fn=my_reward_fn,
)
env_2 = CustomSumoEnvironment(
    net_file=f"./app/data/nets/{file_name[1]}.net.xml",
    route_file=f"./app/data/rou/{file_name[1]}.rou.xml",
    out_csv_name=f"./app/outputs/{file_name[1]}",
    single_agent=True,
    use_gui=True,
    begin_time=0,
    num_seconds=num_seconds,
    time_to_teleport=500,
    yellow_time=4,
    delta_time=5,
    min_green=10,
    reward_fn=my_reward_fn,
)


model = DQN(
    env=env,
    policy="MlpPolicy",
    learning_rate=3e-2,
    learning_starts=1000,
    buffer_size=50000,
    train_freq=1,
    target_update_interval=2700,
    exploration_fraction=0.24,
    exploration_initial_eps=0.83,
    exploration_final_eps=0.01,
    verbose=1,
)
model.learn(total_timesteps=num_seconds)


epi_rewards = []

obs, info = env_2.reset()
print("初期観測値:", obs)
done = False

# 学習後
while not done:
    action, _states = model.predict(obs)
    step_result = env_2.step(action)
    
    obs, rewards, terminated, truncated, info = step_result
    done = truncated

    current_step = info.get("step")
    epi_rewards.append(rewards)

    current_time = info.get("step")
    
    percent = (current_step / num_seconds) * 100
    if percent % 10 == 0:
        print(f"{int(percent)}% completed")

    if done:
        print("シミュレーション終了.")
        avg_reward = np.mean(epi_rewards)
        print(f"平均車両待ち時間: {avg_reward}")
        break
