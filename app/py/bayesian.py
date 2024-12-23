import os
import sys
import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer
import matplotlib.pyplot as plt
from skopt.plots import plot_convergence, plot_objective
from stable_baselines3.dqn.dqn import DQN
from custom_env import CustomSumoEnvironment

#環境変数
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("環境変数の設定をしてください")

env = None

def get_name(reward_fn):
    if callable(reward_fn):
        return reward_fn.__name__
    else:
        return reward_fn

def total_waiting_time(traffic_tignal):
    waiting_time_per_lane = traffic_tignal.get_accumulated_waiting_time_per_lane()
    total_waiting_time = sum(waiting_time_per_lane)
    return -total_waiting_time


def create_env(num_seconds:int, net_name: str, route_type: str, reward_fn: str):
    return CustomSumoEnvironment(
        net_file=f"/Users/chashu/Desktop/dev/sumorl-venv/app/data/{net_name}/{net_name}.net.xml",
        route_file=f"/Users/chashu/Desktop/dev/sumorl-venv/app/data/{net_name}/rou_{route_type}/{net_name}_{route_type}.rou.xml",
        use_gui=False,
        begin_time=0,
        num_seconds=num_seconds,
        time_to_teleport=-1,
        yellow_time=4,
        delta_time=5,
        min_green=10,
        reward_fn=reward_fn,
        single_agent=True,
    )


#評価関数
def evaluate_hyperparameters(params, env):
    learning_rate, exploration_fraction = params

    model = DQN(
        env=env,
        policy="MlpPolicy",
        learning_rate=learning_rate,
        exploration_fraction=exploration_fraction,
        verbose=0
    )

    model.learn(total_timesteps=timesteps)

    #シミュレーション実行
    obs, info = env.reset()
    total_reward = 0
    done = False
    _n_calls = 0

    while not done:
        action, _states = model.predict(obs)
        step_result = env.step(action)
        
        obs, rewards, terminated, truncated, info = step_result
        # sim_step = info.get("step")
        total_reward += rewards
        done = terminated or truncated

        if done:
            _n_calls += 1
            print("done")
            break

    return -total_reward


if __name__ == "__main__":
    timesteps = 100000
    num_seconds:int = 100000
    net_name = "ehira"
    route_type = "a"
    reward_fn = "diff-waiting-time"
    reward = get_name(reward_fn)

    env = create_env(num_seconds, net_name, route_type, reward_fn)

    # 評価関数で環境を渡す
    def objective_function(params):
        return evaluate_hyperparameters(params, env)

    #ハイパーパラメーターの範囲を定義
    space = [
        Real(1e-5, 1e-2, prior="log-uniform", name="learning_rate"),
        Real(0.1, 0.5, name='exploration_fraction'),
    ]

    #ベイズ最適化
    res = gp_minimize(
        objective_function,
        space,
        n_calls=25, #最適化する呼び出し回数
        random_state=0,
    )
    #最適化結果
    print("最適なハイパーパラメータ:")
    print(f"学習率: {res.x[0]}")
    print(f"探索率: {res.x[1]}")
    print(f"最大化された報酬: {-res.fun}")

    try:
        # 最適化の収束過程をプロット
        plot_convergence(res)
        plt.title('Convergence Plot')  # タイトルを追加
        plt.savefig("convergence_plot.png")  # 保存
        plt.show()

        # 目的関数の探索結果をプロット
        plot_objective(res)
        plt.title('Objective Function Exploration')  # タイトルを追加
        plt.savefig("objective_function_plot.png")  # 保存
        plt.show()
    except Exception as e:
        print(f"Error occurred while plotting: {e}")