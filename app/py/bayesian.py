import os
import sys
import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer
from stable_baselines3.dqn.dqn import DQN
from custom_env import CustomSumoEnvironment

#環境変数
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("環境変数の設定をしてください")

env = None


def create_env(num_seconds:int, net_name: str, route_type: str):
    return CustomSumoEnvironment(
        net_file=f"/Users/chashu/Desktop/dev/sumorl-venv/app/data/{net_name}/{net_name}.net.xml",
        route_file=f"/Users/chashu/Desktop/dev/sumorl-venv/app/data/{net_name}/rou_{route_type}/{net_name}_{route_type}.rou.xml",
        out_csv_name=f"/Users/chashu/Desktop/dev/sumorl-venv/app/outputs/{net_name}_{route_type}",
        use_gui=True,
        begin_time=0,
        num_seconds=num_seconds,
        time_to_teleport=1000,
        yellow_time=4,
        delta_time=5,
        min_green=10,
        single_agent=True,
    )


#評価関数
def evaluate_hyperparameters(params, env):
    learning_rate, buffer_size, train_freq, target_update_interval, exploration_fraction, exploration_initial_eps, exploration_final_eps = params

    model = DQN(
        env=env,
        policy="MlpPolicy",
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=100,
        train_freq=(int(train_freq), "step"),
        target_update_interval=target_update_interval,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=exploration_initial_eps,
        exploration_final_eps=exploration_final_eps,
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
        sim_step = info.get("step")
        total_reward += rewards

        if sim_step >= 50000:
            done = True

        if done:
            _n_calls += 1
            print(f"{_n_calls}回完了")
            break

    return -total_reward


if __name__ == "__main__":
    timesteps = 36000
    num_seconds:int = timesteps * 5
    net_name = "ehira"
    route_type = "a"

    env = create_env(num_seconds, net_name, route_type)

    # 評価関数で環境を渡す
    def objective_function(params):
        return evaluate_hyperparameters(params, env)

    #ハイパーパラメーターの範囲を定義
    space = [
        Real(1e-5, 1e-1, name="learning_rate"),
        Integer(10000, 1000000, name='buffer_size'),
        Integer(1, 10, name='train_freq'),
        Integer(100, 10000, name='target_update_interval'),
        Real(0.01, 0.5, name='exploration_fraction'),
        Real(0.1, 1.0, name='exploration_initial_eps'),
        Real(0.01, 0.2, name='exploration_final_eps'),
    ]

    #ベイズ最適化
    res = gp_minimize(
        objective_function,
        space,
        n_calls=10, #最適化する呼び出し回数
        random_state=0,
    )
    #最適化結果
    print("最適なハイパーパラメータ:")
    print(f"学習率: {res.x[0]}")
    print(f"バッファサイズ: {res.x[1]}")
    print(f"学習頻度: {res.x[2]}")
    print(f"ターゲット更新間隔: {res.x[3]}")
    print(f"探索割合: {res.x[4]}")
    print(f"初期探索率: {res.x[5]}")
    print(f"最終探索率: {res.x[6]}")
    print(f"最大化された報酬: {res.fun}")