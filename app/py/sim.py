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
    )

def create_model(env):
    return DQN(
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

def evaluation_model(env, pattern_num):
    epi_rewards = []
    obs, info = env.reset()
    done = False

    while not done:
        action, _states = model.predict(obs)
        step_result = env.step(action)
        
        obs, rewards, terminated, truncated, info = step_result
        done = truncated
        current_step = info.get("step")
        epi_rewards.append(rewards)
        
        percent = (current_step / num_seconds) * 100
        if percent % 10 == 0:
            print(f"{int(percent)}% completed")

        if done:
            print("シミュレーション終了.")
            avg_reward = np.mean(epi_rewards)
            env.save_csv(f"/Users/chashu/Desktop/dev/sumorl-venv/app/outputs/4road_intersection_pattern{str(int(pattern_num))}", 0)
            print(f"平均車両待ち時間: {avg_reward}")
            break

if __name__ == "__main__":
    num_seconds:int = 50000
    pattern_num:int = 4
    pattern:list = ["2way", "2way_right-arrow", "2way_right-lane", "3way"]

    env = create_env(num_seconds, pattern, pattern_num)
    model = create_model(env)

    model.learn(total_timesteps=num_seconds)

    evaluation_model(env, pattern_num)

    

