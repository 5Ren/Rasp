import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# データ
INDX = [0, 1, 2, 3]
F = [30, 30, 40, 50]
n_0 = [3, 9, 6, 6]
n_l = [11, 11, 12, 12]

# Figure & Axes
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection='3d')

# 散布図
ax.scatter(F, n_0, n_l, s=60)

# 各点に index を表示
for i in range(len(INDX)):
    ax.text(
        F[i],
        n_0[i],
        n_l[i],
        f'ID {INDX[i]}',
        fontsize=10
    )

# 軸ラベル
ax.set_xlabel('F')
ax.set_ylabel('n_0')
ax.set_zlabel('n_l')

# 軸範囲
ax.set_xlim(20, 60)
ax.set_ylim(2, 10)
ax.set_zlim(10, 13)

# 表示
plt.tight_layout()
plt.show()
