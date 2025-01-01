import os
import sys
import numpy as np
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import matplotlib.pyplot as plt
from skopt.plots import plot_convergence, plot_objective
from stable_baselines3.dqn.dqn import DQN
from custom_env import CustomSumoEnvironment
from skopt.callbacks import CheckpointSaver
from sklearn.base import BaseEstimator, ClassifierMixin
from tqdm import tqdm

#環境変数
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("環境変数の設定をしてください")

def total_waiting_time(traffic_tignal):
    waiting_time_per_lane = traffic_tignal.get_accumulated_waiting_time_per_lane()
    total_waiting_time = sum(waiting_time_per_lane)
    return -total_waiting_time
    

def create_env(num_seconds:int, net_name: str, route_type: str, reward_fn: str):
    return CustomSumoEnvironment(
        net_file=f"app/data/{net_name}/{net_name}.net.xml",
        route_file=f"app/data/{net_name}/rou_{route_type}/{net_name}_{route_type}.rou.xml",
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

# シミュレーション用ラッパークラスを定義
class DQNSumoWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate=1e-3, exploration_fraction=0.1):
        self.learning_rate = learning_rate
        self.exploration_fraction = exploration_fraction
        self.env = None

    def fit(self, X, y=None):
        if self.env is None:
            self.env = create_env(
                num_seconds=100000,
                net_name="tanimachi9",
                route_type="a",
                reward_fn="diff-waiting-time",
            )
        
        self.model = DQN(
            policy="MlpPolicy",
            env = self.env,
            learning_rate=self.learning_rate,
            exploration_fraction=self.exploration_fraction,
            verbose=0,
        )
        self.model.learn(total_timesteps=100000)
        return self
    
    def score(self, X, y=None):
        obs, info = self.env.reset()
        total_reward = 0
        done = False
        while not done:
            action, _states = self.model.predict(obs)
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            done = terminated or truncated
        return total_reward

param_space = {
    "learning_rate": Real(1e-5, 1e-2),
    "exploration_fraction": Real(0.05, 0.5),
}

model = DQNSumoWrapper()

opt = BayesSearchCV(
    model,
    param_space,
    n_iter=25,
    cv=3,
    verbose=3,
    n_jobs=-1,
)

X_dummy = np.zeros((10,1))
y_dummy = np.zeros(10)

print("最適化を開始...")
opt.fit(X_dummy, y_dummy)

print("最適なハイパラ:", opt.best_params_)
print("最適なスコア:", opt.best_score_)

# 'dimensions' を正しいハイパーパラメータ名に置き換えてください
dimensions = ["learning_rate", "exploration_fraction"]

# 目的関数プロット
_ = plot_objective(opt.optimizer_results_[0],
                   dimensions=dimensions,
                   n_minimum_search=1000)  # 計算負荷を抑える
plt.title("Objective Function Exploration")
plt.savefig("objective_function_plot.png")  # 保存
plt.show()

