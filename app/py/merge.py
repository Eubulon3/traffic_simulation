import pandas as pd

timesteps = 100000
date = "12_20"
net_name = "ehira"
route_type = "a"
reward = "diff-waiting-time"

# CSVファイルのリスト
# csv_files = [
#     f"/Users/chashu/Desktop/dev/sumorl-venv/results/4road_intersection_conn0_ep2.csv",
#     f"/Users/chashu/Desktop/dev/sumorl-venv/results/4road_intersection_conn0_ep4.csv",
#     f"/Users/chashu/Desktop/dev/sumorl-venv/results/4road_intersection_conn0_ep6.csv",
#     f"/Users/chashu/Desktop/dev/sumorl-venv/results/4road_intersection_conn0_ep8.csv",
#     f"/Users/chashu/Desktop/dev/sumorl-venv/results/4road_intersection_conn0_ep10.csv",
#     ]
csv_files = [
    f"/Users/chashu/Desktop/dev/sumorl-venv/results/{date}/{net_name}_{route_type}/{net_name}_{route_type}_{reward}_conn0_ep2.csv",
    f"/Users/chashu/Desktop/dev/sumorl-venv/results/{date}/{net_name}_{route_type}/{net_name}_{route_type}_{reward}_conn0_ep4.csv",
    f"/Users/chashu/Desktop/dev/sumorl-venv/results/{date}/{net_name}_{route_type}/{net_name}_{route_type}_{reward}_conn0_ep6.csv",
    f"/Users/chashu/Desktop/dev/sumorl-venv/results/{date}/{net_name}_{route_type}/{net_name}_{route_type}_{reward}_conn0_ep8.csv",
    f"/Users/chashu/Desktop/dev/sumorl-venv/results/{date}/{net_name}_{route_type}/{net_name}_{route_type}_{reward}_conn0_ep10.csv"
    ]
# csv_files = [
#     f"/Users/chashu/Desktop/dev/sumorl-venv/results/{date}/{net_name}_{route_type}/{net_name}_{route_type}_{reward}_conn0_ep2.csv",
#     f"/Users/chashu/Desktop/dev/sumorl-venv/results/{date}/{net_name}_{route_type}/{net_name}_{route_type}_{reward}_conn0_ep4.csv",
#     f"/Users/chashu/Desktop/dev/sumorl-venv/results/{date}/{net_name}_{route_type}/{net_name}_{route_type}_{reward}_conn0_ep6.csv",
#     f"/Users/chashu/Desktop/dev/sumorl-venv/results/{date}/{net_name}_{route_type}/{net_name}_{route_type}_{reward}_conn0_ep8.csv",
#     f"/Users/chashu/Desktop/dev/sumorl-venv/results/{date}/{net_name}_{route_type}/{net_name}_{route_type}_{reward}_conn0_ep10.csv",
#     f"/Users/chashu/Desktop/dev/sumorl-venv/results/{date}/{net_name}_{route_type}/{net_name}_{route_type}_{reward}_conn0_ep12.csv",
#     f"/Users/chashu/Desktop/dev/sumorl-venv/results/{date}/{net_name}_{route_type}/{net_name}_{route_type}_{reward}_conn0_ep14.csv",
#     f"/Users/chashu/Desktop/dev/sumorl-venv/results/{date}/{net_name}_{route_type}/{net_name}_{route_type}_{reward}_conn0_ep16.csv",
#     ]

# マージしたデータを保存するリスト
merged_data = []

for i, file in enumerate(csv_files):
    # CSVファイルを読み込む
    df = pd.read_csv(file)
    
    # step列に処理を適用 (1列目がstepと仮定)
    df.iloc[:, 0] = df.iloc[:, 0] + (i) * (timesteps + 5)
    
    # マージ用リストに追加
    merged_data.append(df)

# すべてのデータフレームを結合
result = pd.concat(merged_data, ignore_index=True)

# 結果をCSVに保存
result.to_csv(f"/Users/chashu/Desktop/dev/sumorl-venv/results/{date}/{net_name}_{route_type}/{net_name}_{route_type}_{reward}_conn0_dqn.csv", index=False)
# result.to_csv(f"/Users/chashu/Desktop/dev/sumorl-venv/results/4road_intersection_conn0_dqn.csv", index=False)