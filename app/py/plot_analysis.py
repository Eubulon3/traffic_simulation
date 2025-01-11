import glob
from itertools import cycle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# グラフスタイルの設定
sns.set_theme(
    style="darkgrid",
    rc={
        "figure.figsize": (10, 6),
        "text.usetex": False,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "legend.fontsize": 12,
    },
)
colors = sns.color_palette("colorblind", 4)
dashes_styles = cycle(["-", "-.", "--", ":"])
sns.set_palette(colors)
colors = cycle(colors)

# 移動平均を計算する関数
def moving_average(interval, window_size):
    if window_size == 1:
        return interval
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, "same")

# データをプロットする関数
def plot_df(df, color, xaxis, yaxis, ma=1, label=""):
    df[yaxis] = pd.to_numeric(df[yaxis], errors="coerce")  # NaN文字列を数値NaNに変換

    mean = df.groupby(xaxis).mean()[yaxis]
    std = df.groupby(xaxis).std()[yaxis]
    if ma > 1:
        mean = moving_average(mean, ma)
        std = moving_average(std, ma)

    x = df.groupby(xaxis)[xaxis].mean().keys().values
    plt.plot(x, mean, label=label, color=color, alpha=0.7)
    plt.fill_between(x, mean + std, mean - std, alpha=0.25, color=color, rasterized=True)

# メイン処理
labels = ["baseline", "difference waiting time", "total waiting time"]

# メイン処理
def analyze_and_plot(csv_files, xaxis, yaxis, ma=1, xlabel="", ylabel="", title="", output=None):
    plt.figure()

    for file, label in zip(csv_files, labels):  # ラベルをファイルごとに設定
        main_df = pd.DataFrame()
        for f in glob.glob(file + "*"):
            df = pd.read_csv(f)
            if main_df.empty:
                main_df = df
            else:
                main_df = pd.concat((main_df, df))

        plot_df(main_df, xaxis=xaxis, yaxis=yaxis, label=label, color=next(colors), ma=ma)

    # グラフの装飾
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()  # 凡例を表示
    plt.ylim([0, 1500000])

    # 保存または表示
    if output:
        plt.savefig(output, bbox_inches="tight", dpi=300)
    plt.show()

# 使用例
if __name__ == "__main__":
    net_name = "ehira"
    route_type = "c"
    # 使用するCSVファイルのリスト
    csv_files = [
        f"results/1_6/{net_name}_{route_type}/{net_name}_{route_type}_bl_conn0_ep0.csv",
        f"results/1_5/{net_name}_{route_type}/{net_name}_{route_type}_diff-waiting-time_conn0_ep0.csv",
        f"results/1_5/{net_name}_{route_type}/{net_name}_{route_type}_total_waiting_time_conn0_ep0.csv",
        ]

    # 分析とプロット
    analyze_and_plot(
        csv_files=csv_files,
        xaxis="step",
        yaxis="system_total_depart_delay",
        ma=10,
        xlabel="Time Step (seconds)",
        ylabel="Loss Time (seconds)",
        title="",
        output=f"results/1_7/DepartDelayTransitionComparison_{net_name}_{route_type}.jpg",  # 出力ファイル名（指定しない場合は画面に表示）
    )