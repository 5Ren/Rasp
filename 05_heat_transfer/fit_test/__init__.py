import numpy as np
from numpy.linalg import solve
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# データ (x, y, z, k)
# 未加工
# data = np.array([
#     [2.073085714, 0.12, 10.6812, 320.436],
#     [2.993142857, 0.12, 10.6812, 534.06],
#     [3.454285714, 0.12, 10.6812, 747.684]
# ])

# 化学処理
# data = np.array([
#     [2.389028571, 0.12, 10.6812, 320.436],
#     [3.354814286, 0.12, 10.6812, 534.06],
#     [3.972514286, 0.12, 10.6812, 747.684]
# ])

# 階層構造
data = np.array([
    [2.763942857, 0.12, 10.6812, 320.436],
    [3.868628571, 0.12, 10.6812, 534.06],
    [4.690028571, 0.12, 10.6812, 747.684]
])


x = data[:, 0]
y = data[:, 1]
z = data[:, 2]
k = data[:, 3]

# 線形モデル: Ax + B(-y + z)
X_mat = np.vstack([x, -y + z]).T  # ← 形状は (3, 2)

# 最小二乗法で係数を求める
coeffs, residuals, rank, s = np.linalg.lstsq(X_mat, k, rcond=None)
A, B = coeffs

# 予測値
k_pred = A * x + B * (-y + z)

# 評価指標
r2 = r2_score(k, k_pred)
rmse = mean_squared_error(k, k_pred, squared=False)
mae = mean_absolute_error(k, k_pred)

print(f"A = {A:.4f}, B = {B:.4f}")
print(f"R² = {r2:.4f}")
print(f"RMSE = {rmse:.4f}")
print(f"MAE = {mae:.4f}")