import os
import sys
import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer
from stable_baselines3.dqn.dqn import DQN
from sumo_rl import SumoEnvironment

#環境変数
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("環境変数の設定をしてください")


def create_env():
    num_seconds = 10000
    time_to_teleport = 300
    return SumoEnvironment(
        net_file="./app/data/nets/osm.net.xml.gz",
        route_file="./app/data/rou/osm.rou.xml",
        # out_csv_name="./app/data/results",
        single_agent=True,
        use_gui=True,
        begin_time=0,
        num_seconds=num_seconds,
        min_green=10,
        yellow_time=4,
        time_to_teleport=time_to_teleport,
    )


#評価関数
def evaluate_hyperparameters(params):
    learning_rate, buffer_size, target_update_interval, exploration_fraction, exploration_initial_eps, exploration_final_eps = params
    env = create_env()

    model = DQN(
        env=env,
        policy="MlpPolicy",
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=100,
        train_freq=4,
        target_update_interval=target_update_interval,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=exploration_initial_eps,
        exploration_final_eps=exploration_final_eps,
        verbose=0
    )

    model.learn(total_timesteps=10000)

    #シミュレーション実行
    obs, info = env.reset()
    total_reward = 0
    done = False
    _n_calls = 0
    current_time = 0
    max_time = 10000

    while not done:
        _n_calls += 1
        action, _states = model.predict(obs)
        obs, reward, done, additional_info, info_dict = env.step(action)
        total_reward += reward
        current_time += 1

        if current_time >= max_time:
            print(f"{_n_calls}実行完了")
            break
    
    return -total_reward

#ハイパーパラメーターの範囲を定義
space = [
    Real(1e-5, 1e-1, name="learning_rate"),
    Integer(10000, 1000000, name='buffer_size'),
    # Integer(1, 5, name='train_freq'),
    Integer(100, 10000, name='target_update_interval'),
    Real(0.01, 0.5, name='exploration_fraction'),
    Real(0.1, 1.0, name='exploration_initial_eps'),
    Real(0.01, 0.2, name='exploration_final_eps'),
]

#ベイズ最適化
res = gp_minimize(
    evaluate_hyperparameters,
    space,
    n_calls=50, #最適化する呼び出し回数
    random_state=0,
)
#最適化結果
print("最適なハイパーパラメータ:")
print(f"学習率: {res.x[0]}")
print(f"バッファサイズ: {res.x[1]}")
# print(f"学習頻度: {res.x[2]}")
print(f"ターゲット更新間隔: {res.x[2]}")
print(f"探索割合: {res.x[3]}")
print(f"初期探索率: {res.x[4]}")
print(f"最終探索率: {res.x[5]}")
print(f"最小化された報酬: {-res.fun}")

