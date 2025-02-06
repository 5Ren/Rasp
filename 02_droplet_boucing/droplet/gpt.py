import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans, vq

# 二次元配列に格納されたデータ
data = np.array([[7, 13.418], [52, 1.6], [126, 7.491], [248, 3.4], [254, 3.455], [259, 3.473], [350, 2.891],
                  [355, 2.909], [360, 2.873], [436, 2.382], [442, 2.436], [447, 2.382], [488, 0.873], [507, 1.836],
                  [514, 1.982], [519, 2.036], [525, 2.036], [552, 1.018], [555, 0.818], [563, 0.964], [583, 1.655]])

# K-meansアルゴリズムを使用してクラスタリング
k = 13
centroids, _ = kmeans(data, k)
idx, _ = vq(data, centroids)

for i in range(k):
    cluster_data = data[idx == i]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], alpha=0.5)
    plt.scatter(centroids[i, 0], centroids[i, 1], marker='*', s=100)

plt.show()