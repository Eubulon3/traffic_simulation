import pandas as pd
import json

# 入力データのパス設定
net_name = "tanimachi9"
route_type = "c"
reward_1 = "diff-waiting-time"
reward_2 = "total_waiting_time"

# ファイルパス
dqn_csv_1 = f"results/1_11/{net_name}_{route_type}/{net_name}_{route_type}_{reward_1}_conn0_ep0.csv"
dqn_csv_2 = f"results/1_12/{net_name}_{route_type}/{net_name}_{route_type}_{reward_2}_conn0_ep0.csv"
bl_csv_file = f"results/1_13/{net_name}_{route_type}/{net_name}_{route_type}_bl_conn0_ep0.csv"
output_json_path = f"results/1_13/{net_name}_{route_type}/{net_name}_{route_type}_analysis.json"

# CSVファイル読み込み
dqn_data_1 = pd.read_csv(dqn_csv_1)
dqn_data_2 = pd.read_csv(dqn_csv_2)
bl_data = pd.read_csv(bl_csv_file)

# 目的の列
target_columns = [
    "system_total_waiting_time",
    "system_mean_waiting_time",
    "system_mean_speed",
    "system_total_time_loss",
    "system_mean_time_loss",
    "system_total_depart_delay",
    "system_mean_depart_delay",
]

# Metrics を含めたリスト型の辞書に変更
result_dict = {
    "Metrics": target_columns,  # 指標名を列に追加
    "baseline": [],
    f"dqn_{reward_1}": [],
    f"dqn_{reward_2}": [],
    f"dqn_{reward_1}_vs_baseline": [],
    f"dqn_{reward_2}_vs_baseline": [],
}

for column in target_columns:
    # ベースラインの平均
    mean_bl = round(bl_data[column].mean(), 3)
    result_dict["baseline"].append(mean_bl)

    # 報酬1のDQNの平均
    mean_dqn_1 = round(dqn_data_1[column].mean(), 3)
    result_dict[f"dqn_{reward_1}"].append(mean_dqn_1)

    # 報酬2のDQNの平均
    mean_dqn_2 = round(dqn_data_2[column].mean(), 3)
    result_dict[f"dqn_{reward_2}"].append(mean_dqn_2)

    # 比較（ベースラインとの差分）
    rate_dqn_1 = round(((mean_bl - mean_dqn_1) / mean_bl) * 100, 3)
    rate_dqn_2 = round(((mean_bl - mean_dqn_2) / mean_bl) * 100, 3)
    result_dict[f"dqn_{reward_1}_vs_baseline"].append(rate_dqn_1)
    result_dict[f"dqn_{reward_2}_vs_baseline"].append(rate_dqn_2)


# 結果の出力
import json
with open(output_json_path, "w", encoding="utf-8") as file:
    json.dump(result_dict, file, ensure_ascii=False, indent=4)

# 必要に応じてDataFrameに変換することもできます
df_result = pd.DataFrame(result_dict)
df_result.to_csv(output_json_path.replace(".json", ".csv"), index=False)