import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# CSVファイルからデータを読み込む
data = np.loadtxt('1-9_n1_analysed_height.csv', delimiter=',')
time = data[:, 0]
time = time / 2000
height = data[:, 1]

# 時系列データのプロット
plt.figure(figsize=(10, 4))
plt.plot(time, height)
plt.xlabel('Time')
plt.ylabel('Height')
plt.title('Time series of droplet height')
plt.show()

# ピーク検出のためのパラメータ設定
threshold = 0.5  # ピークと見なす閾値
min_distance = 10  # ピーク同士の最小距離（インデックス単位）

# 信号のピーク検出
peaks, _ = find_peaks(height, height=threshold, distance=min_distance)

# 時系列データとピークのプロット
plt.figure(figsize=(10, 4))
plt.plot(time, height)
plt.plot(time[peaks], height[peaks], 'ro')  # ピークを赤い点で表示
plt.xlabel('Time')
plt.ylabel('Signal')
plt.title('Time series with peak detection')
plt.show()
