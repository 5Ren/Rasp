import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans, vq
from scipy.signal import find_peaks

# CSVファイルの読み込み
data = np.genfromtxt(
    r"2ul_40mm_1-9_n1_analysed_6.csv",
    delimiter=',')

# xとyの配列にデータを分割
x = data[:, 0]
y = data[:, 1]

# グラフのプロット
# plt.plot(x, y)

# ピーク検出
peaks, _ = find_peaks(y)

peak_list = []
for peak_point in range(len(peaks)):
    peak_list.append([peaks[peak_point], y[peaks[peak_point]]])

data = np.array(peak_list)
# print(data)

# K-meansアルゴリズムを使用してクラスタリング
k = 13
centroids, _ = kmeans(data, k)
idx, _ = vq(data, centroids)

for i in range(k):
    cluster_data = data[idx.astype(int) == i]
    # plt.scatter(cluster_data[:, 0], cluster_data[:, 1], alpha=0.5)
    plt.scatter(centroids[i, 0], centroids[i, 1], marker='*', s=100)
print(f'{cluster_data=}')
print(f'{centroids=}')
plt.show()
