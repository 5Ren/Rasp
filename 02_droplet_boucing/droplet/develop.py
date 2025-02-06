import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.cluster.vq import kmeans, vq

# CSVファイルの読み込み
data = np.genfromtxt(
    r"2ul_40mm_1-9_n1_analysed_6.csv",
    delimiter=',')

# xとyの配列にデータを分割
x = data[:, 0]
y = data[:, 1]

# グラフのプロット
plt.plot(x, y)

# ピーク検出
peaks, _ = find_peaks(y)


# ピーク位置とピーク値の出力
for i in range(len(peaks)):
    plt.plot(x[peaks[i]], y[peaks[i]], "x")

# グラフのタイトル、ラベル、凡例を設定
plt.title('2D Data Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend(['data'])

# グラフの表示
plt.show()
