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


num_seconds = 5000

pattern_num = 3
pattern = ["2way", "2way_right-arrow", "2way_right-lane", "3way"]



def my_reward_fn(traffic_tignal):
    waiting_time_per_lane = traffic_tignal.get_accumulated_waiting_time_per_lane()
    total_waiting_time = sum(waiting_time_per_lane)
    return -total_waiting_time

env_0 = CustomSumoEnvironment(
    net_file=f"/Users/chashu/Desktop/dev/sumorl-venv/app/data/4road_intersection/net/{pattern[0]}.net.xml",
    route_file=f"/Users/chashu/Desktop/dev/sumorl-venv/app/data/4road_intersection/rou/{pattern[0]}.rou.xml",
    out_csv_name=f"/Users/chashu/Desktop/dev/sumorl-venv/app/outputs/4road_intersection_pattern0",
    use_gui=True,
    begin_time=0,
    num_seconds=num_seconds,
    time_to_teleport=-1,
    yellow_time=3,
    delta_time=5,
    min_green=10,
    # reward_fn=my_reward_fn,
    single_agent=True,
)
env_1 = CustomSumoEnvironment(
    net_file=f"/Users/chashu/Desktop/dev/sumorl-venv/app/data/4road_intersection/net/{pattern[1]}.net.xml",
    route_file=f"/Users/chashu/Desktop/dev/sumorl-venv/app/data/4road_intersection/rou/{pattern[1]}.rou.xml",
    out_csv_name=f"/Users/chashu/Desktop/dev/sumorl-venv/app/outputs/4road_intersection_pattern1",
    use_gui=True,
    begin_time=0,
    num_seconds=num_seconds,
    time_to_teleport=-1,
    yellow_time=3,
    delta_time=5,
    min_green=10,
    # reward_fn=my_reward_fn,
    single_agent=True,
)
env_2 = CustomSumoEnvironment(
    net_file=f"/Users/chashu/Desktop/dev/sumorl-venv/app/data/4road_intersection/net/{pattern[2]}.net.xml",
    route_file=f"/Users/chashu/Desktop/dev/sumorl-venv/app/data/4road_intersection/rou/{pattern[2]}.rou.xml",
    out_csv_name=f"/Users/chashu/Desktop/dev/sumorl-venv/app/outputs/4road_intersection_pattern2",
    use_gui=True,
    begin_time=0,
    num_seconds=num_seconds,
    time_to_teleport=-1,
    yellow_time=3,
    delta_time=5,
    min_green=10,
    # reward_fn=my_reward_fn,
    single_agent=True,
)
env_3 = CustomSumoEnvironment(
    net_file=f"/Users/chashu/Desktop/dev/sumorl-venv/app/data/4road_intersection/net/{pattern[3]}.net.xml",
    route_file=f"/Users/chashu/Desktop/dev/sumorl-venv/app/data/4road_intersection/rou/{pattern[3]}.rou.xml",
    out_csv_name=f"/Users/chashu/Desktop/dev/sumorl-venv/app/outputs/4road_intersection_pattern3",
    use_gui=True,
    begin_time=0,
    num_seconds=num_seconds,
    time_to_teleport=-1,
    yellow_time=3,
    delta_time=5,
    min_green=10,
    # reward_fn=my_reward_fn,
    single_agent=True,
)


model = DQN(
    env=env_0,
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

obs, info = env_0.reset()
print("初期観測値:", obs)
done = False

# 学習後
while not done:
    action, _states = model.predict(obs)
    step_result = env_0.step(action)
    
    obs, rewards, terminated, truncated, info = step_result
    done = truncated

    current_step = info.get("step")
    epi_rewards.append(rewards)

    current_time = info.get("step")
    
    percent = (current_step / num_seconds) * 100
    if percent % 10 == 0:
        print(f"{int(percent)}% completed")

    if done:
        print("env_0: シミュレーション終了.")
        avg_reward = np.mean(epi_rewards)
        env_0.save_csv("/Users/chashu/Desktop/dev/sumorl-venv/app/outputs/env_0_result", 0)
        print(f"平均車両待ち時間: {avg_reward}")
        break

epi_rewards = []

obs, info = env_1.reset()
print("初期観測値:", obs)
done = False

# 学習後
while not done:
    action, _states = model.predict(obs)
    step_result = env_1.step(action)
    
    obs, rewards, terminated, truncated, info = step_result
    done = truncated

    current_step = info.get("step")
    epi_rewards.append(rewards)

    current_time = info.get("step")
    
    percent = (current_step / num_seconds) * 100
    if percent % 10 == 0:
        print(f"{int(percent)}% completed")

    if done:
        print("env_1: シミュレーション終了.")
        avg_reward = np.mean(epi_rewards)
        env_1.save_csv("/Users/chashu/Desktop/dev/sumorl-venv/app/outputs/env_1_result", 0)
        print(f"平均車両待ち時間: {avg_reward}")
        break

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
        print("env_2: シミュレーション終了.")
        avg_reward = np.mean(epi_rewards)
        env_2.save_csv("/Users/chashu/Desktop/dev/sumorl-venv/app/outputs/env_2_result", 0)
        print(f"平均車両待ち時間: {avg_reward}")
        break

epi_rewards = []

obs, info = env_3.reset()
print("初期観測値:", obs)
done = False

# 学習後
while not done:
    action, _states = model.predict(obs)
    step_result = env_3.step(action)
    
    obs, rewards, terminated, truncated, info = step_result
    done = truncated

    current_step = info.get("step")
    epi_rewards.append(rewards)

    current_time = info.get("step")
    
    percent = (current_step / num_seconds) * 100
    if percent % 10 == 0:
        print(f"{int(percent)}% completed")

    if done:
        print("env_3: シミュレーション終了.")
        avg_reward = np.mean(epi_rewards)
        env_3.save_csv("/Users/chashu/Desktop/dev/sumorl-venv/app/outputs/env_3_result", 0)
        print(f"平均車両待ち時間: {avg_reward}")
        break