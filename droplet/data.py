import numpy
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.cluster.vq import kmeans, vq

# グラフの設定
# ---------------------------------------------------------
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 11
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams["figure.dpi"] = 400
# ---------------------------------------------------------


# CSVファイルの読み込み
data = np.genfromtxt(
    r"./2ul_40mm_1-9_n1_analysed_6.csv",
    delimiter=',')

# xとyの配列にデータを分割
x = data[:, 0]
y = data[:, 1]

# ピーク検出
peaks, _ = find_peaks(y)

# ピーク間の範囲を取得し、各ピークのデータを分割
range_x_list = np.split(x, peaks)
range_y_list = np.split(y, peaks)

min_max_list = []

# 各ピークのデータをグラフにプロット
for i in range(len(range_x_list)):
    current_x_list = range_x_list[i]
    current_y_list = range_y_list[i]

    # 最大・最小を取得
    max_value_y = max(current_y_list)
    max_value_y_index = numpy.argmax(current_y_list)
    max_value_x = current_x_list[max_value_y_index]

    min_value_y = min(current_y_list)
    min_value_y_index = numpy.argmin(current_y_list)
    min_value_x = current_x_list[min_value_y_index]

    # 最大，最小をプロット
    plt.plot(max_value_x, max_value_y, marker="x", color='red')
    plt.plot(min_value_x, min_value_y, marker="x", color='blue')

    # 曲線プロット
    plt.plot(range_x_list[i], range_y_list[i])

    # 最大最小をリストに追加
    min_max_list.append([max_value_x, max_value_y])
    min_max_list.append([min_value_x, min_value_y])
    plt.plot(max_value_x, max_value_y, 'x')
    plt.plot(min_value_x, min_value_y, 'x')

min_max_list_np = np.array(min_max_list)



data = min_max_list_np
# K-meansアルゴリズムを使用してクラスタリング
k = 15
centroids, _ = kmeans(data, k)
idx, _ = vq(data, centroids)

for i in range(k):
    cluster_data = data[idx == i]
    # plt.scatter(cluster_data[:, 0], cluster_data[:, 1], alpha=0.5)
    plt.scatter(centroids[i, 0], centroids[i, 1], marker='*', s=100)

plt.show()