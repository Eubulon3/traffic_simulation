import pandas as pd
import json

sim_date = "12_11"
net_name = "ehira"
route_type = "b"
reward = "total_waiting_time"

dqn_csv_file = f"/Users/chashu/Desktop/dev/sumorl-venv/results/{sim_date}/{net_name}_{route_type}/{net_name}_{route_type}_{reward}_conn0_ep0.csv"
bl_csv_file = f"/Users/chashu/Desktop/dev/sumorl-venv/results/{sim_date}/{net_name}_{route_type}/{net_name}_{route_type}_bl_conn0_ep0.csv"
output_json_path = f"/Users/chashu/Desktop/dev/sumorl-venv/results/{sim_date}/{net_name}_{route_type}/{net_name}_{route_type}_{reward}_analysis.json"

dqn_data = pd.read_csv(dqn_csv_file)
bl_data = pd.read_csv(bl_csv_file)

target_colmun = [
    "system_total_stopped",
    "system_total_waiting_time",
    "system_mean_waiting_time",
    "system_mean_speed",
    "tl_stopped",
    "tl_accumulated_waiting_time",
    "tl_average_speed",
    "agents_total_stopped",
    "agents_total_accumulated_waiting_time",
    ]

rate_dict = {}
for i in range(len(target_colmun)):
    mean_dqn_data = dqn_data[target_colmun[i]].mean()
    mean_bl_data = bl_data[target_colmun[i]].mean()
    # print(f"mean_dqn: {mean_dqn_data}, mean_bl: {mean_bl_data}")
    rate_of_change = ((mean_bl_data - mean_dqn_data) / mean_bl_data) * 100
    rate_dict[target_colmun[i]] = rate_of_change

serializable_dict = {key: value.tolist() for key, value in rate_dict.items()}

with open(output_json_path, "w", encoding="utf-8") as file:
    json.dump(serializable_dict, file, ensure_ascii=False, indent=4)

